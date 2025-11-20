# traderbruh_web_dashboard_gh.py
# TraderBruh — Global Edition (ASX, US, INDIA)
# Architecture: "The Bible" Logic + Multi-Market Tabs + News Fix

from datetime import datetime, time
import os, re, glob, json
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
import zoneinfo

try:
    from pypdf import PdfReader
    HAVE_PYPDF = True
except Exception:
    HAVE_PYPDF = False

# ---------------- Config ----------------
# ✅ FIX: Restored SYD definition for local file parsing
SYD = zoneinfo.ZoneInfo('Australia/Sydney')

FETCH_DAYS          = 900
MINI_BARS           = 120
SPARK_DAYS          = 90
OUTPUT_DIR          = "docs"
OUTPUT_HTML         = os.path.join(OUTPUT_DIR, "index.html")
ANN_DIR             = 'announcements'
NEWS_WINDOW_DAYS    = 14
PATTERN_LOOKBACK    = 180
PIVOT_WINDOW        = 4
PRICE_TOL           = 0.03
PATTERNS_CONFIRMED_ONLY = True

# Technical Rules
RULES = {
    'buy':     {'rsi_min': 45, 'rsi_max': 70},
    'dca':     {'rsi_max': 45, 'sma200_proximity': 0.05},
    'avoid':   {'death_cross': True},
    'autodca': {'gap_thresh': -2.0, 'fill_req': 50.0}
}

BREAKOUT_RULES = {
    'atr_mult':   0.50,
    'vol_mult':   1.30,
    'buffer_pct': 0.003,
}
AUTO_UPGRADE_BREAKOUT = True

# Fundamental Rules (Buffett/Piotroski-lite)
FUNDY_THRESHOLDS = {
    'roe_high': 0.15, 'roe_mid': 0.10,
    'margin_high': 0.15, 'margin_mid': 0.08,
    'de_safe': 0.50, 'de_risky': 1.50,
    'curr_ratio': 1.5,
    'peg_fair': 2.0,
    'fcf_yield': 0.03
}

# ---------------- The Global Universe ----------------
MARKETS = {
    'AU': {
        'name': '🇦🇺 ASX',
        'tz': 'Australia/Sydney',
        'suffix': '.AX',
        'stocks': {
            'CSL': ('CSL Limited', 'Biotech: vaccines, plasma & specialty therapies.'),
            'COH': ('Cochlear', 'Implantable hearing devices leader.'),
            'BHP': ('BHP Group', 'Global Miner.'),
            'CBA': ('CommBank', 'Banking Major.'),
            'WTC': ('WiseTech', 'Logistics SaaS.'),
            'XRO': ('Xero', 'SMB cloud accounting.'),
            'PME': ('Pro Medicus', 'Cloud radiology/AI.'),
            'PLS': ('Pilbara Minerals', 'Hard-rock lithium.'),
            'FMG': ('Fortescue', 'Iron Ore & Green Energy.'),
            'TLS': ('Telstra', 'Telecom Infrastructure.'),
            'WOW': ('Woolworths', 'Consumer Staples.'),
            'REA': ('REA Group', 'Real Estate Portal.'),
            'STO': ('Santos', 'Energy/Oil/Gas.'),
            'NXT': ('NextDC', 'Data Centres.'),
            'DRO': ('DroneShield', 'Counter-UAS & EW.')
        }
    },
    'US': {
        'name': '🇺🇸 Wall St',
        'tz': 'America/New_York',
        'suffix': '', # US tickers usually have no suffix on YF
        'stocks': {
            'NVDA': ('NVIDIA', 'AI Hardware Leader.'),
            'MSFT': ('Microsoft', 'Cloud & AI.'),
            'AAPL': ('Apple', 'Consumer Tech ecosystem.'),
            'AMZN': ('Amazon', 'E-comm & AWS.'),
            'GOOGL': ('Alphabet', 'Search & AI.'),
            'META': ('Meta', 'Social & VR.'),
            'TSLA': ('Tesla', 'EV & Robotics.'),
            'PLTR': ('Palantir', 'Big Data Defence.'),
            'COIN': ('Coinbase', 'Crypto Infrastructure.'),
            'BRK-B': ('Berkshire', 'The Buffett Fortress.'),
            'AMD': ('AMD', 'Semiconductors.'),
            'NET': ('Cloudflare', 'Edge Cloud & Security.')
        }
    },
    'IN': {
        'name': '🇮🇳 India',
        'tz': 'Asia/Kolkata',
        'suffix': '.NS',
        'stocks': {
            'RELIANCE': ('Reliance Ind', 'Conglomerate (Oil/Telco/Retail).'),
            'TCS': ('TCS', 'IT Services Global.'),
            'HDFCBANK': ('HDFC Bank', 'Banking Leader.'),
            'INFY': ('Infosys', 'IT Consulting.'),
            'TATAMOTORS': ('Tata Motors', 'Auto (JLR + EV).'),
            'ICICIBANK': ('ICICI Bank', 'Private Banking.'),
            'BAJFINANCE': ('Bajaj Finance', 'Consumer Lending.'),
            'ITC': ('ITC', 'FMCG & Hotels.'),
            'ZOMATO': ('Zomato', 'Food Delivery Tech.'),
            'PAYTM': ('Paytm', 'Fintech (Risky).')
        }
    }
}

# ---------------- Data Fetching (TA + Deep FA) ----------------
def fetch_prices(symbol: str, timezone: str) -> pd.DataFrame:
    """Fetch OHLCV history (Timezone Aware)."""
    df = yf.download(symbol, period=f'{FETCH_DAYS}d', interval='1d', auto_adjust=False, progress=False, group_by='column', prepost=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.xs(symbol, axis=1, level=-1, drop_level=True)
        except: df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df['Date'] = pd.to_datetime(df[date_col], utc=True)

    # Intraday stitch (Timezone Aware)
    market_tz = zoneinfo.ZoneInfo(timezone)
    now_market = datetime.now(market_tz)
    last_date_market = df['Date'].dt.tz_convert(market_tz).dt.date.max()

    if (now_market.time() >= time(10, 0) and last_date_market < now_market.date()):
        try:
            intr = yf.download(symbol, period='5d', interval='60m', auto_adjust=False, progress=False, prepost=False, group_by='column')
            if intr is not None and not intr.empty:
                if isinstance(intr.columns, pd.MultiIndex):
                    try: intr = intr.xs(symbol, axis=1, level=-1, drop_level=True)
                    except: intr.columns = intr.columns.get_level_values(0)
                intr = intr.reset_index()
                intr['Date'] = pd.to_datetime(intr[intr.columns[0]], utc=True)
                last = intr.tail(1).iloc[0]
                top = pd.DataFrame([{
                    'Date': last['Date'], 'Open': float(last['Open']), 'High': float(last['High']),
                    'Low': float(last['Low']), 'Close': float(last['Close']), 'Volume': float(last['Volume']),
                }])
                df = pd.concat([df, top], ignore_index=True)
        except: pass

    df['Date'] = df['Date'].dt.tz_convert(market_tz).dt.tz_localize(None)
    return df.dropna(subset=['Close'])

def fetch_deep_fundamentals(symbol: str):
    """
    Performs a deep dive into Financial Statements (Income, Balance Sheet, Cash Flow).
    Calculates a 'Buffett/Piotroski' Quality Score (0-10).
    """
    try:
        tick = yf.Ticker(symbol)
        info = tick.info
        
        # Fetch Statements (DataFrame)
        try:
            bs = tick.balance_sheet  # Balance Sheet
            is_ = tick.income_stmt   # Income Statement
            cf = tick.cashflow       # Cash Flow
        except:
            bs, is_, cf = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # --- Helpers to safely get latest or 3Y avg ---
        def get_item(df, item_names, idx=0):
            if df.empty: return 0
            for name in item_names:
                if name in df.index:
                    try: return float(df.loc[name].iloc[idx])
                    except: return 0
            return 0

        def get_cagr(df, item_names, years=3):
            if df.empty or df.shape[1] < years: return 0
            curr = get_item(df, item_names, 0)
            past = get_item(df, item_names, years-1)
            if past <= 0 or curr <= 0: return 0
            return (curr / past)**(1/(years-1)) - 1

        # --- 1. Profitability & Efficiency (3 pts) ---
        roe_3y = 0
        try:
            if not is_.empty and not bs.empty:
                roes = []
                for i in range(min(3, len(is_.columns), len(bs.columns))):
                    ni = get_item(is_, ['Net Income'], i)
                    eq = get_item(bs, ['Stockholders Equity', 'Total Equity Gross Minority Interest'], i)
                    if eq > 0: roes.append(ni/eq)
                if roes: roe_3y = sum(roes) / len(roes)
        except: pass

        ocf = get_item(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
        net_inc = get_item(is_, ['Net Income'])
        high_quality_earnings = (ocf > net_inc)
        marg_curr = info.get('profitMargins', 0)
        
        score = 0
        if roe_3y > 0.15: score += 2
        elif roe_3y > 0.10: score += 1
        if marg_curr > 0.10: score += 1
        if high_quality_earnings: score += 0.5

        # --- 2. Balance Sheet Health (3 pts) ---
        curr_ratio = info.get('currentRatio', 0)
        debt_eq = info.get('debtToEquity', 999)
        if debt_eq > 50: debt_eq = debt_eq / 100.0

        cash = get_item(bs, ['Cash And Cash Equivalents', 'Cash Financial'])
        lt_debt = get_item(bs, ['Long Term Debt'])
        
        if cash > lt_debt: score += 1.5
        elif debt_eq < 0.5: score += 1
        
        if curr_ratio > 1.5: score += 1
        elif curr_ratio > 1.1: score += 0.5

        # --- 3. Capital Allocation (2 pts) ---
        shares_curr = get_item(bs, ['Share Issued', 'Ordinary Shares Number'], 0)
        shares_old = get_item(bs, ['Share Issued', 'Ordinary Shares Number'], 2)
        
        is_buyback = False
        if shares_old > 0:
            change = (shares_curr - shares_old) / shares_old
            if change < -0.01: score += 1.5; is_buyback = True
            elif change < 0.05: score += 1
        
        # --- 4. Growth & Valuation (2 pts) ---
        rev_cagr = get_cagr(is_, ['Total Revenue', 'Operating Revenue'], 3)
        if rev_cagr > 0.10: score += 1
        
        peg = info.get('pegRatio', 0)
        pe = info.get('trailingPE', 0)
        
        if (peg and 0 < peg < 2.0) or (pe and 0 < pe < 25): score += 1
        
        score = min(score, 10)
        tier = 'Fortress' if score >= 7 else ('Quality' if score >= 4 else ('Spec' if score >= 2 else 'Junk'))
        
        return {
            'score': round(score, 1), 'tier': tier, 'roe_3y': roe_3y, 'margins': marg_curr,
            'debt_eq': debt_eq, 'rev_cagr': rev_cagr, 'is_buyback': is_buyback,
            'fcf_yield': (info.get('freeCashflow', 0) / (info.get('marketCap', 1) or 1)) if info.get('marketCap') else 0,
            'pe': pe, 'cash': cash
        }

    except Exception as e:
        return {'score': 0, 'tier': 'Error', 'roe_3y':0, 'margins':0, 'debt_eq':0, 'rev_cagr':0, 'is_buyback':False, 'fcf_yield':0, 'pe':0, 'cash': 0}

# ---------------- TA Indicators ----------------
def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values('Date').reset_index(drop=True)
    x['SMA20']  = x['Close'].rolling(20).mean()
    x['SMA50']  = x['Close'].rolling(50).mean()
    x['SMA200'] = x['Close'].rolling(200).mean()
    x['EMA21']  = x['Close'].ewm(span=21, adjust=False).mean()
    x['High20'] = x['High'].rolling(20).max()
    x['High52W']= x['High'].rolling(252).max()
    x['Vol20']  = x['Volume'].rolling(20).mean()

    chg = x['Close'].diff()
    gains = chg.clip(lower=0).rolling(14).mean()
    losses = (-chg).clip(lower=0).rolling(14).mean()
    RS = gains / losses
    x['RSI14'] = 100 - (100 / (1 + RS))

    x['Dist_to_52W_High_%'] = (x['Close'] / x['High52W'] - 1) * 100.0
    x['Dist_to_SMA200_%']   = (x['Close'] / x['SMA200']  - 1) * 100.0

    x['H-L']  = x['High'] - x['Low']
    x['H-C']  = (x['High'] - x['Close'].shift(1)).abs()
    x['L-C']  = (x['Low']  - x['Close'].shift(1)).abs()
    x['TR']   = x[['H-L', 'H-C', 'L-C']].max(axis=1)
    x['ATR14']= x['TR'].rolling(14).mean()
    return x

def label_row(r: pd.Series) -> str:
    buy_ok = (
        (r['Close'] > r['SMA200']) and
        (r['Close'] > r['High20']) and
        (r['SMA50'] > r['SMA200']) and
        (RULES['buy']['rsi_min'] <= r['RSI14'] <= RULES['buy']['rsi_max'])
    )
    dca_ok = (
        (r['Close'] >= r['SMA200']) and
        (r['RSI14'] < RULES['dca']['rsi_max']) and
        (r['Close'] <= r['SMA200'] * (1 + RULES['dca']['sma200_proximity']))
    )
    avoid = (r['SMA50'] < r['SMA200']) if RULES['avoid']['death_cross'] else False
    if buy_ok: return 'BUY'
    if dca_ok: return 'DCA'
    if avoid: return 'AVOID'
    return 'WATCH'

# Auto DCA & Pattern Logic
def auto_dca_gate(ind: pd.DataFrame):
    if len(ind) < 3: return False, {}
    D0, D1, D2 = ind.iloc[-1], ind.iloc[-2], ind.iloc[-3]
    gap_pct = (D1['Open'] / D2['Close'] - 1) * 100.0
    if not np.isfinite(gap_pct) or gap_pct > RULES['autodca']['gap_thresh']:
        return False, {}
    gap_mid = (D1['High'] + D1['Low']) / 2.0
    reclaim_mid = bool(D0['Close'] > gap_mid)
    above_ema21 = bool(D0['Close'] > D0['EMA21'])
    gap_size = max(D2['Close'] - D1['Open'], 0.0)
    fill_pct = float(0.0 if gap_size == 0 else (D0['Close'] - D1['Open']) / gap_size * 100.0)
    filled50 = bool(fill_pct >= RULES['autodca']['fill_req'])
    flag = reclaim_mid and above_ema21 and filled50
    return flag, {'gap_pct': float(gap_pct), 'reclaim_mid': reclaim_mid, 'above_ema21': above_ema21, 'gap_fill_%': fill_pct}

def _pivots(ind, window=PIVOT_WINDOW):
    v = ind.tail(PATTERN_LOOKBACK).reset_index(drop=True).copy()
    ph = (v['High'] == v['High'].rolling(window * 2 + 1, center=True).max())
    pl = (v['Low']  == v['Low'].rolling(window * 2 + 1, center=True).min())
    v['PH'] = ph.fillna(False); v['PL'] = pl.fillna(False)
    return v

def _similar(a, b, tol=PRICE_TOL):
    m = (a + b) / 2.0
    return (abs(a - b) / m) <= tol

def detect_double_bottom(ind):
    v = _pivots(ind)
    lows = v.index[v['PL']].tolist()
    out = []
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            li, lj = lows[i], lows[j]
            if lj - li < 10: continue
            p1, p2 = float(v.loc[li, 'Low']), float(v.loc[lj, 'Low'])
            if not _similar(p1, p2): continue
            neck = float(v.loc[li:lj, 'High'].max())
            confirmed = bool(v['Close'].iloc[-1] > neck)
            conf = 0.6 + (0.2 if confirmed else 0.0)
            if np.isfinite(v['Vol20'].iloc[-1]) and confirmed and v['Volume'].iloc[-1] > 1.2 * v['Vol20'].iloc[-1]: conf += 0.2
            lines = [('h', v.loc[li, 'Date'], v.loc[lj, 'Date'], (p1 + p2) / 2.0), ('h', v.loc[li, 'Date'], v['Date'].iloc[-1], neck)]
            out.append({'name': 'Double Bottom', 'status': 'confirmed' if confirmed else 'forming', 'confidence': round(min(conf, 1.0), 2), 'levels': {'base': round((p1 + p2) / 2.0, 4), 'neckline': round(neck, 4)}, 'lines': lines})
            return out
    return out

def detect_double_top(ind):
    v = _pivots(ind)
    highs = v.index[v['PH']].tolist()
    out = []
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            hi, hj = highs[i], highs[j]
            if hj - hi < 10: continue
            p1, p2 = float(v.loc[hi, 'High']), float(v.loc[hj, 'High'])
            if not _similar(p1, p2): continue
            neck = float(v.loc[hi:hj, 'Low'].min())
            confirmed = bool(v['Close'].iloc[-1] < neck)
            conf = 0.6 + (0.2 if confirmed else 0.0)
            if np.isfinite(v['Vol20'].iloc[-1]) and confirmed and v['Volume'].iloc[-1] > 1.2 * v['Vol20'].iloc[-1]: conf += 0.2
            lines = [('h', v.loc[hi, 'Date'], v.loc[hj, 'Date'], (p1 + p2) / 2.0), ('h', v.loc[hi, 'Date'], v['Date'].iloc[-1], neck)]
            out.append({'name': 'Double Top', 'status': 'confirmed' if confirmed else 'forming', 'confidence': round(min(conf, 1.0), 2), 'levels': {'ceiling': round((p1 + p2) / 2.0, 4), 'neckline': round(neck, 4)}, 'lines': lines})
            return out
    return out

def detect_inverse_hs(ind):
    v = _pivots(ind)
    lows = v.index[v['PL']].tolist()
    out = []
    for i in range(len(lows) - 2):
        l1, h, l2 = lows[i], lows[i + 1], lows[i + 2]
        pL1, pH, pL2 = float(v.loc[l1, 'Low']), float(v.loc[h, 'Low']), float(v.loc[l2, 'Low'])
        if not (pH < pL1 * (1 - 0.04) and pH < pL2 * (1 - 0.04)): continue
        if not _similar(pL1, pL2): continue
        left_high, right_high = float(v.loc[l1:h, 'High'].max()), float(v.loc[h:l2, 'High'].max())
        confirmed = bool(v['Close'].iloc[-1] > min(left_high, right_high))
        conf = 0.6 + (0.2 if confirmed else 0.0)
        lines = [('seg', v.loc[l1, 'Date'], left_high, v.loc[l2, 'Date'], right_high)]
        out.append({'name': 'Inverse H&S', 'status': 'confirmed' if confirmed else 'forming', 'confidence': round(min(conf, 1.0), 2), 'levels': {'neck_left': round(left_high, 4), 'neck_right': round(right_high, 4)}, 'lines': lines})
        return out
    return out

def detect_hs(ind):
    v = _pivots(ind)
    highs = v.index[v['PH']].tolist()
    out = []
    for i in range(len(highs) - 2):
        l1, h, l2 = highs[i], highs[i + 1], highs[i + 2]
        pL1, pH, pL2 = float(v.loc[l1, 'High']), float(v.loc[h, 'High']), float(v.loc[l2, 'High'])
        if not (pH > pL1 * (1 + 0.04) and pH > pL2 * (1 + 0.04)): continue
        if not _similar(pL1, pL2): continue
        left_low, right_low = float(v.loc[l1:h, 'Low'].min()), float(v.loc[h:l2, 'Low'].min())
        confirmed = bool(v['Close'].iloc[-1] < max(left_low, right_low))
        conf = 0.6 + (0.2 if confirmed else 0.0)
        lines = [('seg', v.loc[l1, 'Date'], left_low, v.loc[l2, 'Date'], right_low)]
        out.append({'name': 'Head & Shoulders', 'status': 'confirmed' if confirmed else 'forming', 'confidence': round(min(conf, 1.0), 2), 'levels': {'neck_left': round(left_low, 4), 'neck_right': round(right_low, 4)}, 'lines': lines})
        return out
    return out

def detect_triangles(ind):
    v = _pivots(ind)
    tail = v.tail(120).copy()
    phs, pls = tail[tail['PH']], tail[tail['PL']]
    out = []
    if len(phs) >= 2 and len(pls) >= 2:
        ph_vals = phs['High'].values
        for i in range(len(ph_vals) - 1):
            if _similar(ph_vals[i], ph_vals[i + 1]):
                res = (ph_vals[i] + ph_vals[i + 1]) / 2.0
                slope = np.polyfit(np.arange(len(pls)), pls['Low'].values, 1)[0]
                if slope > 0:
                    confirmed = bool(tail['Close'].iloc[-1] > res)
                    conf = 0.55 + (0.25 if confirmed else 0.0)
                    lines = [('h', pls['Date'].iloc[0], tail['Date'].iloc[-1], res), ('seg', pls['Date'].iloc[0], pls['Low'].iloc[0], pls['Date'].iloc[-1], pls['Low'].iloc[-1])]
                    out.append({'name': 'Ascending Triangle', 'status': 'confirmed' if confirmed else 'forming', 'confidence': round(min(conf, 1.0), 2), 'levels': {'resistance': round(res, 4)}, 'lines': lines})
                    break
        pl_vals = pls['Low'].values
        for i in range(len(pl_vals) - 1):
            if _similar(pl_vals[i], pl_vals[i + 1]):
                sup = (pl_vals[i] + pl_vals[i + 1]) / 2.0
                slope = np.polyfit(np.arange(len(phs)), phs['High'].values, 1)[0]
                if slope < 0:
                    confirmed = bool(tail['Close'].iloc[-1] < sup)
                    conf = 0.55 + (0.25 if confirmed else 0.0)
                    lines = [('h', phs['Date'].iloc[0], tail['Date'].iloc[-1], sup), ('seg', phs['Date'].iloc[0], phs['High'].iloc[0], phs['Date'].iloc[-1], phs['High'].iloc[-1])]
                    out.append({'name': 'Descending Triangle', 'status': 'confirmed' if confirmed else 'forming', 'confidence': round(min(conf, 1.0), 2), 'levels': {'support': round(sup, 4)}, 'lines': lines})
                    break
    return out

def detect_flag(ind):
    if len(ind) < 60: return False, {}
    look = ind.tail(40)
    impulse = (look['Close'].max() / look['Close'].min() - 1) * 100
    if not np.isfinite(impulse) or impulse < 12: return False, {}
    win = 14
    tail = ind.tail(max(win, 8)).copy()
    x = np.arange(len(tail))
    hi, lo = np.polyfit(x, tail['High'].values, 1), np.polyfit(x, tail['Low'].values, 1)
    slope_pct = (hi[0] / tail['Close'].iloc[-1]) * 100
    ch = (np.polyval(hi, x[-1]) - np.polyval(lo, x[-1]))
    tight = ch <= max(0.4 * (look['Close'].max() - look['Close'].min()), 0.02 * tail['Close'].iloc[-1])
    gentle = (-0.006 <= slope_pct <= 0.002)
    return (tight and gentle), {'hi': hi.tolist(), 'lo': lo.tolist(), 'win': win}

def pattern_bias(name: str) -> str:
    if name in ("Double Bottom", "Inverse H&S", "Ascending Triangle", "Bull Flag"): return "bullish"
    if name in ("Double Top", "Head & Shoulders", "Descending Triangle"): return "bearish"
    return "neutral"

def breakout_ready_dt(ind: pd.DataFrame, pat: dict, rules: dict):
    if not pat or pat.get('name') != 'Double Top': return False, {}
    last = ind.iloc[-1]
    atr, vol, vol20 = float(last.get('ATR14', np.nan)), float(last.get('Volume', np.nan)), float(last.get('Vol20', np.nan))
    ceiling = float(pat.get('levels', {}).get('ceiling', np.nan))
    if not (np.isfinite(atr) and np.isfinite(vol) and np.isfinite(vol20) and np.isfinite(ceiling)): return False, {}
    close = float(last['Close'])
    ok_price = (close >= ceiling * (1.0 + rules['buffer_pct'])) and (close >= ceiling + rules['atr_mult'] * atr)
    ok_vol   = (vol20 > 0) and (vol >= rules['vol_mult'] * vol20)
    return bool(ok_price and ok_vol), {'ceiling': round(ceiling, 4), 'atr': round(atr, 4), 'stop': round(close - atr, 4)}

def mini_candle(ind, flag_info=None, pattern_lines=None):
    v = ind.tail(MINI_BARS).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=v['Date'], open=v['Open'], high=v['High'], low=v['Low'], close=v['Close'],
        hoverinfo='skip', showlegend=False, increasing_line_color='#4ade80', decreasing_line_color='#f87171'
    ))
    if 'SMA20' in v.columns:
        fig.add_trace(go.Scatter(x=v['Date'], y=v['SMA20'], mode='lines', line=dict(width=1.4, color='rgba(56,189,248,0.8)'), hoverinfo='skip', showlegend=False))
    if flag_info:
        t2 = ind.tail(max(flag_info.get('win', 14), 8)).copy()
        x = np.arange(len(t2))
        hi, lo = np.poly1d(flag_info['hi']), np.poly1d(flag_info['lo'])
        for line_data in [hi(x), lo(x)]:
            fig.add_trace(go.Scatter(x=t2['Date'], y=line_data, mode='lines', line=dict(width=2, dash='dash', color='rgba(167,139,250,0.95)'), hoverinfo='skip', showlegend=False))
    if pattern_lines:
        def _expand(lines):
            out = []
            if not lines: return out
            for ln in lines:
                if ln[0] == 'h': _, d_left, d_right, y = ln; out.append(('h', d_left, y, d_right, y))
                else: _, d1, y1, d2, y2 = ln; out.append(('seg', d1, y1, d2, y2))
            return out
        for (kind, x1, y1, x2, y2) in _expand(pattern_lines):
            c, d = ('rgba(34,197,94,0.95)', 'dot') if kind == 'h' else ('rgba(234,179,8,0.95)', 'solid')
            fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode='lines', line=dict(width=2, color=c, dash=d), hoverinfo='skip', showlegend=False))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=130, width=280,
        xaxis=dict(visible=False, fixedrange=True), yaxis=dict(visible=False, fixedrange=True),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'staticPlot': True})

# ---------------- News & Announcements ----------------
NEWS_TYPES = [
    ('Appendix 3Y', r'Appendix\s*3Y|Change of Director.?s? Interest Notice', 'director'),
    ('Appendix 2A', r'Appendix\s*2A|Application for quotation of securities', 'issue'),
    ('Cleansing Notice', r'Cleansing Notice', 'issue'),
    ('Price Query', r'Price Query|Aware Letter|Response to ASX Price Query', 'reg'),
    ('Share Price Movement', r'Share Price Movement', 'reg'),
    ('Trading Halt', r'Trading Halt', 'reg'),
    ('Withdrawal', r'withdrawn', 'reg'),
    ('Orders / Contracts', r'order|contract|sale|revenue|cash receipts', 'ops'),
]

def parse_3y_stats(text: str):
    act = 'Disposed' if re.search(r'\bDisposed\b', text, re.I) else ('Acquired' if re.search(r'\bAcquired\b', text, re.I) else None)
    shares, value = None, None
    m = re.search(r'(\d{1,3}(?:,\d{3}){1,3})\s+(?:ordinary|fully\s+paid|shares)', text, re.I)
    if m: shares = m.group(1)
    v = re.search(r'\$ ?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)', text)
    if v: value = v.group(1)
    parts = [p for p in [act, f"{shares} shares" if shares else None, f"A${value}" if value else None] if p]
    return ' • '.join(parts) if parts else None

def read_pdf_first_text(path: str):
    if not HAVE_PYPDF: return ''
    try: return re.sub(r'[ \t]+', ' ', PdfReader(path).pages[0].extract_text() or '')
    except: return ''

def parse_announcements():
    rows = []
    if not os.path.isdir(ANN_DIR): return pd.DataFrame(columns=['Date', 'Ticker', 'Type', 'Tag', 'Headline', 'Details', 'Path'])
    for fp in sorted(glob.glob(os.path.join(ANN_DIR, '*.pdf'))):
        fname = os.path.basename(fp)
        ticker = None
        m = re.match(r'([A-Z]{2,4})[_-]', fname)
        if m: ticker = m.group(1)
        text = read_pdf_first_text(fp)
        _type, tag = 'Announcement', 'gen'
        for label, pat, t in NEWS_TYPES:
            if re.search(pat, fname + " " + text, re.I):
                _type, tag = label, t
                break
                
        d = None
        if (md := re.search(r'(\d{1,2}\s+[A-Za-z]{3,9}\s+20\d{2})', text)):
            try: d = datetime.strptime(md.group(1), '%d %B %Y')
            except: 
                try: d = datetime.strptime(md.group(1), '%d %b %Y')
                except: pass
        if d is None: d = datetime.fromtimestamp(os.path.getmtime(fp), tz=SYD).replace(tzinfo=None)
        details = parse_3y_stats(text) if _type == 'Appendix 3Y' else None
        rows.append({'Date': d.date().isoformat(), 'Ticker': ticker or '', 'Type': _type, 'Tag': tag, 'Headline': _type, 'Details': details, 'Path': fp})
    return pd.DataFrame(rows)

# ---------------- Commentary Logic ----------------
def is_euphoria(r: pd.Series) -> bool:
    return (r['Dist_to_52W_High_%'] > -3.5) and (r['Dist_to_SMA200_%'] > 50.0) and (r['RSI14'] >= 70.0)

def comment_for_row(r: pd.Series) -> str:
    d200, d52, rsi, sig = r['Dist_to_SMA200_%'], r['Dist_to_52W_High_%'], r['RSI14'], str(r.get('Signal', '')).upper()
    f_score = r.get('Fundy_Score', 0)
    
    # --- THE MATRIX (Technicals x Fundamentals) ---
    # 1. BUY Signal
    if sig == 'BUY':
        if f_score >= 7:
            return f"<b>CORE BUY (High Conviction):</b> Strong technical uptrend backed by Fortress/Quality fundamentals ({f_score}/10). Good for sizing up."
        elif f_score <= 3:
            return f"<b>SPECULATIVE BUY (Trash Rally):</b> Uptrend in price, but weak fundamentals ({f_score}/10). Use tight stops; do not marry this trade."
        else:
            return f"Standard Buy: Uptrend intact (close > 200DMA). RSI {rsi:.0f} is constructive. Fundys OK ({f_score}/10)."

    # 2. DCA Signal
    elif sig == 'DCA':
        if f_score <= 3:
            return f"<b>AVOID (Falling Knife):</b> Technicals suggest a dip-buy, but fundamentals ({f_score}/10) are Junk. High risk of value trap."
        elif f_score >= 7:
            return f"<b>QUALITY DIP (Accumulate):</b> Fortress balance sheet ({f_score}/10) on sale near 200DMA. Ideal for measured adds."
        else:
            return f"DCA Zone: Trading near 200DMA (Δ {d200:.1f}%) with cooling RSI. Decent risk/reward for solid co."

    # 3. WATCH Signal
    elif sig == 'WATCH':
        if is_euphoria(r):
            if f_score <= 3:
                return f"<b>EXIT WARNING:</b> Junk rally in euphoria zone ({abs(d52):.1f}% off high). Take profits."
            return f"Euphoria Zone: Price extended. Quality hold, but trim or trail stops."
        else:
            if f_score >= 7:
                return f"<b>GOLDEN WATCHLIST:</b> High quality ({f_score}/10) near inflection. Wait for setup."
            return f"Watch: {abs(d52):.1f}% off highs. Momentum mixed."

    # 4. AVOID Signal
    elif sig == 'AVOID':
        if f_score >= 7:
            return f"<b>VALUE WATCH:</b> Great business ({f_score}/10) in a downtrend. Do not catch falling knife; wait for base."
        return f"Avoid: Weak trend (Δ 200DMA {d200:.1f}%) matches weak fundamentals. Sidelines."

    return "Neutral."

# ---------------- Main Execution Loop ----------------
all_market_html = ""
print("Parsing local news...")
news_df_all = parse_announcements()

for mkt_code, config in MARKETS.items():
    print(f"--- Processing {config['name']} ---")
    rows = []
    
    for ticker, meta in config['stocks'].items():
        symbol = ticker + config['suffix']
        print(f"Fetching {symbol}...")
        
        # 1. Price Data
        df = fetch_prices(symbol, config['tz'])
        if df.empty: continue
        
        # 2. Fundamental Data
        fundy = fetch_deep_fundamentals(symbol)
        
        # 3. Indicators
        ind = indicators(df).dropna(subset=['SMA200', 'SMA50', 'High20', 'RSI14', 'EMA21', 'Vol20', 'ATR14'])
        if ind.empty: continue
        last = ind.iloc[-1]
        sig = label_row(last)

        # 4. Pattern / Gate Logic
        flag_flag, flag_det = detect_flag(ind)
        pats = detect_double_bottom(ind) + detect_double_top(ind) + detect_inverse_hs(ind) + detect_hs(ind) + detect_triangles(ind)
        if PATTERNS_CONFIRMED_ONLY: pats = [p for p in pats if p.get('status') == 'confirmed']
        
        breakout_ready, breakout_info = False, {}
        if (dt_pats := [p for p in pats if p.get('name') == 'Double Top']):
            breakout_ready, breakout_info = breakout_ready_dt(ind, dt_pats[0], BREAKOUT_RULES)
        
        signal_auto = False
        if AUTO_UPGRADE_BREAKOUT and breakout_ready:
            sig = 'BUY'; signal_auto = True

        gate_flag, gate_det = auto_dca_gate(ind)
        pname = pats[0]['name'] if pats else ''
        palign = 'ALIGNED' if (pattern_bias(pname) in [str(sig).lower(), 'neutral'] or str(sig).lower() == 'neutral') else 'CONFLICT'

        # Build Row
        r = {
            'Ticker': ticker, 'Name': meta[0], 'Desc': meta[1],
            'LastDate': pd.to_datetime(last['Date']).strftime('%Y-%m-%d'), 'LastClose': float(last['Close']),
            'SMA20': float(last['SMA20']), 'SMA50': float(last['SMA50']), 'SMA200': float(last['SMA200']),
            'RSI14': float(last['RSI14']), 'High52W': float(last['High52W']),
            'Dist_to_52W_High_%': float(last['Dist_to_52W_High_%']), 'Dist_to_SMA200_%': float(last['Dist_to_SMA200_%']),
            'Signal': sig, 'SignalAuto': bool(signal_auto), 'Comment': None,
            'Flag': bool(flag_flag), '_flag_info': flag_det, '_pattern_lines': pats[0]['lines'] if pats else None,
            '_pattern_name': pname, '_pattern_status': pats[0]['status'] if pats else '', '_pattern_conf': pats[0]['confidence'] if pats else np.nan,
            '_pattern_align': palign, 'AutoDCA_Flag': bool(gate_flag), 'AutoDCA_Gap_%': float(gate_det.get('gap_pct', np.nan)),
            'AutoDCA_ReclaimMid': bool(gate_det.get('reclaim_mid', False)), 'AutoDCA_AboveEMA21': bool(gate_det.get('above_ema21', False)),
            'AutoDCA_Fill_%': float(gate_det.get('gap_fill_%', np.nan)), 'BreakoutReady': bool(breakout_ready),
            'Breakout_Level': float(breakout_info.get('ceiling', np.nan)), 'Breakout_Stop': float(breakout_info.get('stop', np.nan)),
            # Fundamentals
            'Fundy_Score': fundy['score'], 'Fundy_Tier': fundy['tier'], 'Fundy_ROE': fundy['roe_3y'],
            'Fundy_Margin': fundy['margins'], 'Fundy_PE': fundy['pe'], 'Fundy_PEG': fundy['peg'],
            'Fundy_Growth': fundy['rev_cagr'], 'Fundy_Cash': fundy['cash'], 'Fundy_Debt': 0 if fundy['debt_eq'] == 999 else fundy['debt_eq']
        }
        
        r['Comment'] = comment_for_row(r)
        if r.get('BreakoutReady', False): r['Comment'] += f" • BreakoutReady: cleared {r['Breakout_Level']:.2f}."
        
        # News Attach
        if not news_df_all.empty and mkt_code == 'AU':
            nd = news_df_all[news_df_all['Ticker'] == ticker]
            if not nd.empty:
                # Filter for recent
                try:
                    today_syd = datetime.now(SYD).date()
                    nd['IsRecent'] = nd.apply(lambda x: (today_syd - datetime.strptime(x['Date'], '%Y-%m-%d').date()).days <= NEWS_WINDOW_DAYS, axis=1)
                    recent_news = nd[nd['IsRecent']]
                    if not recent_news.empty:
                        top = recent_news.sort_values('Date').iloc[-1]
                        r['Comment'] += f" • 📰 News: {top['Type']} ({top['Date']})"
                except: pass

        r['_mini_candle'] = mini_candle(ind, r['_flag_info'] if r['Flag'] else None, r['_pattern_lines'])
        rows.append(r)

    df_mkt = pd.DataFrame(rows)
    if df_mkt.empty: continue
    
    # Rank & Sort
    buy = df_mkt[df_mkt.Signal == 'BUY'].sort_values(['Fundy_Score', 'Dist_to_52W_High_%'], ascending=[False, False])
    dca = df_mkt[df_mkt.Signal == 'DCA'].sort_values(['Fundy_Score', 'Dist_to_SMA200_%'], ascending=[False, True])
    watch = df_mkt[df_mkt.Signal == 'WATCH'].sort_values(['Fundy_Score', 'Dist_to_52W_High_%'], ascending=[False, False])
    avoid = df_mkt[df_mkt.Signal == 'AVOID'].sort_values('Fundy_Score', ascending=True)
    
    gate = df_mkt[df_mkt['AutoDCA_Flag'] == True].sort_values('AutoDCA_Fill_%', ascending=False)
    pats = df_mkt[df_mkt['_pattern_name'] != ''].sort_values(['_pattern_conf', 'Ticker'], ascending=[False, True])
    
    # Render HTML Components
    def render_card(r, badge_type):
        score = r['Fundy_Score']
        s_badge = "shield-high" if score >= 7 else ("shield-low" if score <= 3 else "buy") 
        s_icon = "💎" if score >= 9 else ("🛡️" if score >= 7 else ("⚠️" if score <= 3 else "⚖️"))
        fundy_html = f'<span class="badge {s_badge}" style="margin-left:6px">{s_icon} {score}/10 {r["Fundy_Tier"]}</span>'
        
        tags = ""
        if r['Flag']: tags += '<span class="badge news">🚩 FLAG</span>'
        if r['AutoDCA_Flag']: tags += '<span class="badge dca">GATE</span>'
        if r['_pattern_name']: tags += f'<span class="badge watch">{r["_pattern_name"]}</span>'
        
        return f"""
        <div class="card searchable-item">
            <div class="card-header">
                <div>
                    <a href="#" class="ticker-badge mono">{r['Ticker']}</a>
                    <span class="badge {badge_type}" style="margin-left:8px">{r['Signal']}</span>
                    {fundy_html} {tags}
                    <div style="font-size:12px; color:var(--text-muted); margin-top:4px">{r['Name']}</div>
                </div>
                <div class="price-block">
                    <div class="price-main mono">${r['LastClose']:.2f}</div>
                    <div class="price-sub">{r['LastDate']}</div>
                </div>
            </div>
            <div class="metrics-row">
                <div class="metric"><label>RSI(14)</label><span class="mono" style="color:{'#ef4444' if r['RSI14']>70 else '#10b981'}">{r['RSI14']:.0f}</span></div>
                <div class="metric"><label>vs 200DMA</label><span class="mono">{r['Dist_to_SMA200_%']:+.1f}%</span></div>
                <div class="metric"><label>Score</label><span class="mono">{r['Fundy_Score']}</span></div>
            </div>
            <div class="comment-box">{r['Comment']}</div>
            <div class="chart-container">{r['_mini_candle']}</div>
        </div>"""

    grid_html = ""
    for section, sub, b in [('BUY — Actionable', buy, 'buy'), ('DCA — Accumulate', dca, 'dca'), ('WATCH — Monitoring', watch, 'watch'), ('AVOID — Sidelines', avoid, 'avoid')]:
        if not sub.empty:
            grid_html += f"<h3 style='margin-top:30px; color:var(--text-muted)'>{section}</h3><div class='grid'>{''.join([render_card(r, b) for _, r in sub.iterrows()])}</div>"

    # Tables (Safe String Construction)
    def fmt_pe(x): return f"{x:.1f}" if x and x > 0 else "-"
    def fmt_pct(x): return f"{x*100:.1f}%" if x else "-"
    
    fundy_rows_list = []
    for _, r in df_mkt.sort_values('Fundy_Score', ascending=False).iterrows():
        score = r['Fundy_Score']
        b_cls = 'shield-high' if score >= 7 else ('shield-low' if score <= 3 else 'watch')
        fundy_rows_list.append(f"<tr class='searchable-item'><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td><span class='badge {b_cls}'>{score}/10 {r['Fundy_Tier']}</span></td><td class='mono'>{fmt_pct(r['Fundy_ROE'])}</td><td class='mono'>{fmt_pct(r['Fundy_Margin'])}</td><td class='mono'>{fmt_pe(r['Fundy_PE'])}</td><td class='mono'>{fmt_pct(r['Fundy_Growth'])}</td><td class='mono'>{r['Fundy_Debt']:.2f}</td></tr>")
    fundy_html = "".join(fundy_rows_list)

    dca_html = ""
    if not gate.empty:
        dca_list = []
        for _, r in gate.iterrows():
            dca_list.append(f"<tr class='searchable-item'><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td class='mono text-red'>{r['AutoDCA_Gap_%']:.1f}%</td><td class='mono'>{'Yes' if r['AutoDCA_ReclaimMid'] else 'No'}</td><td class='mono'>{'Yes' if r['AutoDCA_AboveEMA21'] else 'No'}</td><td class='mono'>{r['AutoDCA_Fill_%']:.1f}%</td><td>{r['_mini_candle']}</td></tr>")
        dca_html = "".join(dca_list)

    pat_html = ""
    if not pats.empty:
        pat_list = []
        for _, r in pats.iterrows():
            pat_list.append(f"<tr class='searchable-item'><td>{r['_pattern_name']}</td><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td class='mono'>{r['_pattern_status']}</td><td class='mono'>{r['_pattern_conf']:.2f}</td><td class='mono'>{r['_pattern_align']}</td><td>{r['_mini_candle']}</td></tr>")
        pat_html = "".join(pat_list)

    news_html = ""
    if not news_df_all.empty and mkt_code == 'AU':
        n_list = []
        for _, n in news_df_all.sort_values('Date', ascending=False).iterrows():
            n_list.append(f"<tr class='searchable-item'><td class='mono' style='color:var(--text-muted)'>{n['Date']}</td><td><b>{n['Ticker']}</b></td><td><span class='badge news'>{n['Type']}</span></td><td>{n['Headline']}</td></tr>")
        news_html = f"<h2 id='news' style='margin-top:40px'>Local Announcements</h2><div class='card'><div class='table-responsive'><table><thead><tr><th>Date</th><th>Ticker</th><th>Type</th><th>Detail</th></tr></thead><tbody>{''.join(n_list)}</tbody></table></div></div>"

    # Playbook
    playbook_html = """
    <div class="playbook">
        <b>The "TraderBruh Shield" (0-10 Score):</b><br>
        Separating "Compounders" from "Garbage" using Buffett/Quality principles.<br><br>
        <b>Metrics Tracked (Max 10 pts):</b><br>
        • <b>Profitability (3 pts):</b> ROE > 15% (Efficiency), Margins > 15% (Pricing Power).<br>
        • <b>Survival (3 pts):</b> Low Debt/Equity (<0.5), High Current Ratio (>1.5), Interest Coverage.<br>
        • <b>Cash/Capital (2 pts):</b> FCF Yield & Dividends.<br>
        • <b>Growth/Val (2 pts):</b> Rev Growth > 10%, PEG < 2.<br><br>
        <b>The Matrix:</b><br>
        💎 <b>High Quality (Score 7-10):</b> Fortress balance sheets. OK to DCA dips or size up on breakouts.<br>
        ⚠️ <b>Speculative/Junk (Score 0-3):</b> Weak fundamentals. If TA says "Buy", use tight stops. If TA says "DCA", <b>AVOID</b> (Falling Knife).
    </div>
    """

    # Assemble Tab Content
    all_market_html += f"""
    <div id="tab-{mkt_code}" class="market-tab-content" style="display:{'block' if mkt_code=='AU' else 'none'}">
        <div style="margin-bottom:20px"><h1 style="font-size:24px; margin:0">{config['name']} Overview</h1></div>
        {playbook_html}
        {grid_html}
        
        <h2 style="margin-top:40px">Fundamental Health Check</h2>
        <div class="card"><div class="table-responsive"><table><thead><tr><th>Ticker</th><th>Quality Score</th><th>ROE</th><th>Margin</th><th>P/E</th><th>Rev Growth</th><th>Debt/Eq</th></tr></thead><tbody>{fundy_html}</tbody></table></div></div>

        <h2 style="margin-top:40px">Auto-DCA Candidates</h2>
        <div class="card"><div class="table-responsive"><table><thead><tr><th>Ticker</th><th>Gap %</th><th>Reclaim Mid?</th><th>&gt; EMA21?</th><th>Gap-fill %</th><th>Trend</th></tr></thead><tbody>{dca_html if dca_html else "<tr><td colspan='6' style='text-align:center; color:var(--text-muted)'>No active setups today</td></tr>"}</tbody></table></div></div>
        
        <h2 style="margin-top:40px">Patterns & Structures</h2>
        <div class="card"><div class="table-responsive"><table><thead><tr><th>Pattern</th><th>Ticker</th><th>Status</th><th>Confidence</th><th>Alignment</th><th>Mini</th></tr></thead><tbody>{pat_html if pat_html else "<tr><td colspan='6' style='text-align:center; color:var(--text-muted)'>No patterns right now.</td></tr>"}</tbody></table></div></div>

        {news_html}
    </div>"""

# ---------------- CSS/JS & Export ----------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg: #0f172a; --surface-1: #1e293b; --surface-2: #334155; --primary: #3b82f6;
    --text-main: #f1f5f9; --text-muted: #94a3b8;
    --accent-green: #10b981; --accent-amber: #f59e0b; --accent-red: #ef4444; --accent-purple: #a855f7;
    --glass: rgba(30, 41, 59, 0.7); --border: rgba(148, 163, 184, 0.1);
}
* { box-sizing: border-box; -webkit-font-smoothing: antialiased; }
body {
    background: var(--bg); background-image: radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.1) 0px, transparent 50%), radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.1) 0px, transparent 50%);
    background-attachment: fixed; color: var(--text-main); font-family: 'Inter', sans-serif; margin: 0; padding-bottom: 60px; font-size: 14px;
}
.mono { font-family: 'JetBrains Mono', monospace; }
.text-green { color: var(--accent-green); } .text-red { color: var(--accent-red); } .text-amber { color: var(--accent-amber); }
.text-purple { color: var(--accent-purple); } .text-primary { color: var(--primary); } .hidden { display: none !important; }

.nav-wrapper { position: sticky; top: 0; z-index: 100; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border); padding: 10px 16px; }
.nav-tabs { display:flex; gap:10px; max-width:800px; margin:0 auto; overflow-x:auto; padding-bottom:4px; }
.tab-btn { background:rgba(255,255,255,0.05); border:none; color:var(--text-muted); padding:8px 16px; border-radius:20px; cursor:pointer; font-weight:600; transition:all 0.2s; white-space:nowrap; }
.tab-btn.active, .tab-btn:hover { background:var(--primary); color:white; }

.search-container { max-width: 1200px; margin: 16px auto 0; padding: 0 16px; }
.search-input { width: 100%; background: var(--glass); border: 1px solid var(--border); padding: 12px 16px; border-radius: 12px; color: white; font-family: 'Inter'; font-size: 15px; outline: none; transition: border-color 0.2s; }
.search-input:focus { border-color: var(--primary); }

.container { max-width: 1200px; margin: 0 auto; padding: 20px 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }
@media(max-width: 600px) { .grid { grid-template-columns: 1fr; } }

.card { background: var(--glass); backdrop-filter: blur(10px); border: 1px solid var(--border); border-radius: 16px; padding: 16px; overflow: hidden; position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
.card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
.ticker-badge { background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 15px; letter-spacing: 0.5px; color: white; text-decoration: none; display: inline-block; }
.price-block { text-align: right; }
.price-main { font-size: 18px; font-weight: 600; }
.price-sub { font-size: 11px; color: var(--text-muted); }

.metrics-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 8px; }
.metric { display: flex; flex-direction: column; }
.metric label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; }
.metric span { font-size: 13px; font-weight: 500; }

.comment-box { font-size: 13px; line-height: 1.5; color: #cbd5e1; margin-bottom: 12px; padding-top: 8px; border-top: 1px solid var(--border); }
.playbook { background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; color: #e2e8f0; line-height: 1.6; }
.playbook b { color: white; }

.badge { padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; text-transform: uppercase; display: inline-block; }
.badge.buy { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.2); }
.badge.dca { background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.2); }
.badge.watch { background: rgba(59, 130, 246, 0.15); color: var(--primary); border: 1px solid rgba(59, 130, 246, 0.2); }
.badge.avoid { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.2); }
.badge.news { background: rgba(168, 85, 247, 0.15); color: var(--accent-purple); }
.badge.shield-high { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.2); }
.badge.shield-low { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.2); }

.table-responsive { overflow-x: auto; border-radius: 12px; border: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 12px 16px; color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border); background: rgba(15, 23, 42, 0.5); white-space: nowrap; }
td { padding: 12px 16px; border-bottom: 1px solid var(--border); vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(255,255,255,0.02); }
.chart-container { margin-top: 10px; border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05); }
"""

JS = """
function init() {
    // Search
    const searchInput = document.getElementById('globalSearch');
    const searchableItems = document.querySelectorAll('.searchable-item');
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        searchableItems.forEach(item => {
            const text = item.innerText.toLowerCase();
            item.classList.toggle('hidden', !text.includes(query));
        });
    });
    // Sort
    document.querySelectorAll('th').forEach(th => {
        th.addEventListener('click', () => {
            const table = th.closest('table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const idx = Array.from(th.parentNode.children).indexOf(th);
            const asc = th.dataset.asc === 'true';
            rows.sort((a, b) => {
                const v1 = a.children[idx].innerText;
                const v2 = b.children[idx].innerText;
                const n1 = parseFloat(v1.replace(/[^0-9.-]/g, ''));
                const n2 = parseFloat(v2.replace(/[^0-9.-]/g, ''));
                if (!isNaN(n1) && !isNaN(n2)) return asc ? n1 - n2 : n2 - n1;
                return asc ? v1.localeCompare(v2) : v2.localeCompare(v1);
            });
            rows.forEach(r => tbody.appendChild(r));
            th.dataset.asc = !asc;
        });
    });
}
function openTab(marketId) {
    document.querySelectorAll('.market-tab-content').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + marketId).style.display = 'block';
    document.getElementById('btn-' + marketId).classList.add('active');
}
window.addEventListener('DOMContentLoaded', init);
"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TraderBruh Global</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>{CSS}</style>
    <script>{JS}</script>
</head>
<body>
    <div class="nav-wrapper">
        <div class="nav-tabs">
            <button id="btn-AU" class="tab-btn active" onclick="openTab('AU')">🇦🇺 ASX</button>
            <button id="btn-US" class="tab-btn" onclick="openTab('US')">🇺🇸 USA</button>
            <button id="btn-IN" class="tab-btn" onclick="openTab('IN')">🇮🇳 India</button>
        </div>
    </div>
    <div class="search-container">
        <input type="text" id="globalSearch" class="search-input" placeholder="Search tickers, signals, or patterns...">
    </div>
    <div class="container">
        <div style="margin-bottom:20px">
            <h1 style="font-size:24px; margin:0 0 4px 0">Market Overview</h1>
            <div style="color:var(--text-muted); font-size:13px">Global Technicals & Fundamentals • Updated {datetime.now(SYD).strftime('%I:%M %p %Z')}</div>
        </div>
        {all_market_html}
        <div style="text-align:center; margin-top:40px; color:var(--text-muted); font-size:12px">Generated by TraderBruh • Not Financial Advice</div>
    </div>
</body>
</html>
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_HTML, 'w', encoding='utf-8') as f: f.write(HTML)
print(f'Done: {OUTPUT_HTML}')