# traderbruh_global_final.py
# TraderBruh — Global Web Dashboard (ASX / USA / INDIA)
# Version: Ultimate 4.0 (Exact Feature Parity + Multi-Market Tabs)

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

# ---------------- Global Config ----------------
OUTPUT_DIR          = "docs"
OUTPUT_HTML         = os.path.join(OUTPUT_DIR, "index.html")
ANN_DIR             = 'announcements'
FETCH_DAYS          = 900
MINI_BARS           = 120
SPARK_DAYS          = 90
NEWS_WINDOW_DAYS    = 14
PATTERN_LOOKBACK    = 180
PIVOT_WINDOW        = 4
PRICE_TOL           = 0.03
PATTERNS_CONFIRMED_ONLY = True

# Technical Rules (The Bible)
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

# --- Market Definitions ---
# We define the "Universe" for each market here
MARKETS = {
    'AUS': {
        'name': 'Australia (ASX)',
        'tz': 'Australia/Sydney',
        'currency': 'A$',
        'suffix': '.AX',
        'tickers': {
            'CSL': ('CSL Limited', 'Biotech: vaccines & plasma.'),
            'COH': ('Cochlear', 'Hearing implants leader.'),
            'XRO': ('Xero', 'Cloud accounting SaaS.'),
            'WTC': ('WiseTech', 'Logistics software (CargoWise).'),
            'PLS': ('Pilbara Minerals', 'Lithium producer.'),
            'MIN': ('MinRes', 'Mining services + Li/Fe.'),
            'RMD': ('ResMed', 'Sleep apnea devices.'),
            'DRO': ('DroneShield', 'Counter-UAS defence.'),
            'PNV': ('PolyNovo', 'Wound care (BTM).'),
            'HUB': ('HUB24', 'Wealth platform.'),
            'NXT': ('NEXTDC', 'Data centres.'),
            'TNE': ('TechnologyOne', 'Enterprise ERP SaaS.'),
            'REA': ('REA Group', 'Real estate portals.'),
            'CAR': ('Carsales', 'Auto marketplace.'),
            'WES': ('Wesfarmers', 'Conglomerate retail.'),
        }
    },
    'USA': {
        'name': 'United States (Wall St)',
        'tz': 'America/New_York',
        'currency': 'U$',
        'suffix': '',
        'tickers': {
            'NVDA': ('NVIDIA', 'AI Hardware Leader.'),
            'AAPL': ('Apple', 'Consumer Electronics.'),
            'MSFT': ('Microsoft', 'Cloud & Productivity.'),
            'TSLA': ('Tesla', 'EV & Robotics.'),
            'AMD': ('AMD', 'Semiconductors.'),
            'PLTR': ('Palantir', 'Big Data & Defense AI.'),
            'COIN': ('Coinbase', 'Crypto Exchange.'),
            'MSTR': ('MicroStrategy', 'Bitcoin Treasury.'),
            'AMZN': ('Amazon', 'E-comm & AWS.'),
            'GOOG': ('Alphabet', 'Search & AI.'),
            'META': ('Meta', 'Social & Metaverse.'),
            'NFLX': ('Netflix', 'Streaming.'),
            'CRWD': ('CrowdStrike', 'Cybersecurity.'),
            'NET': ('Cloudflare', 'Edge Cloud.'),
	    'BMNR': ('BioMarin', 'Biotech: Genetic therapies.'),
            'CELH': ('Celsius', 'Fitness energy drinks.'),
            'FUBO': ('FuboTV', 'Sports streaming.'),
            'PGY':  ('Pagaya', 'AI Fintech/Lending.'),
            'RKT':  ('Rocket', 'Fintech & Mortgages.'),
        }
    },
    'IND': {
        'name': 'India (NSE)',
        'tz': 'Asia/Kolkata',
        'currency': '₹',
        'suffix': '.NS',
        'tickers': {
            'RELIANCE': ('Reliance Ind', 'Conglomerate (Oil, Telco, Retail).'),
            'TCS': ('TCS', 'IT Services.'),
            'INFY': ('Infosys', 'IT Services.'),
            'HDFCBANK': ('HDFC Bank', 'Banking Leader.'),
            'TATAMOTORS': ('Tata Motors', 'Auto (Jaguar Land Rover).'),
            'ZOMATO': ('Zomato', 'Food Delivery.'),
            'PAYTM': ('Paytm', 'Fintech.'),
            'ITC': ('ITC Ltd', 'FMCG & Hotels.'),
            'ADANIENT': ('Adani Ent', 'Infra Conglomerate.'),
            'WIPRO': ('Wipro', 'IT Services.'),
            'BAJFINANCE': ('Bajaj Finance', 'NBFC Lender.'),
            'DMART': ('Avenue Supermarts', 'Retail Chain.'),
        }
    }
}

# ---------------- Helper Functions (From Original) ----------------

def fetch_prices(symbol: str, tz_name: str) -> pd.DataFrame:
    """Fetch OHLCV history with Market-Specific Timezone Stitching"""
    try:
        df = yf.download(symbol, period=f'{FETCH_DAYS}d', interval='1d', auto_adjust=False, progress=False, group_by='column', prepost=False)
        if df is None or df.empty: return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(symbol, axis=1, level=-1, drop_level=True)
            except: df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        df['Date'] = pd.to_datetime(df[date_col], utc=True)

        # Intraday stitch (Globalized Logic)
        market_tz = zoneinfo.ZoneInfo(tz_name)
        now_mkt = datetime.now(market_tz)
        last_date_mkt = df['Date'].dt.tz_convert(market_tz).dt.date.max()
        
        # If market is active (after 10am) and we don't have today's daily bar yet
        if (now_mkt.time() >= time(10, 0) and last_date_mkt < now_mkt.date()):
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
    except: return pd.DataFrame()

def fetch_deep_fundamentals(symbol: str):
    """Buffett/Piotroski Quality Score (0-10)"""
    try:
        tick = yf.Ticker(symbol)
        info = tick.info
        try:
            bs = tick.balance_sheet
            is_ = tick.income_stmt
            cf = tick.cashflow
        except: bs, is_, cf = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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

        # 1. Profitability
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

        # 2. Balance Sheet
        curr_ratio = info.get('currentRatio', 0)
        debt_eq = info.get('debtToEquity', 999)
        if debt_eq > 50: debt_eq = debt_eq / 100.0

        cash = get_item(bs, ['Cash And Cash Equivalents', 'Cash Financial'])
        lt_debt = get_item(bs, ['Long Term Debt'])
        
        if cash > lt_debt: score += 1.5
        elif debt_eq < 0.5: score += 1
        
        if curr_ratio > 1.5: score += 1
        elif curr_ratio > 1.1: score += 0.5

        # 3. Capital
        shares_curr = get_item(bs, ['Share Issued', 'Ordinary Shares Number'], 0)
        shares_old = get_item(bs, ['Share Issued', 'Ordinary Shares Number'], 2)
        is_buyback = False
        if shares_old > 0:
            change = (shares_curr - shares_old) / shares_old
            if change < -0.01: score += 1.5; is_buyback = True
            elif change < 0.05: score += 1
        
        # 4. Growth
        rev_cagr = get_cagr(is_, ['Total Revenue', 'Operating Revenue'], 3)
        if rev_cagr > 0.10: score += 1
        
        peg = info.get('pegRatio', 0)
        pe = info.get('trailingPE', 0)
        if (peg and 0 < peg < 2.0) or (pe and 0 < pe < 20): score += 1
        
        score = min(score, 10)
        tier = 'Fortress' if score >= 7 else ('Quality' if score >= 4 else 'Spec')
        
        return {
            'score': round(score, 1), 'tier': tier, 'roe_3y': roe_3y, 'margins': marg_curr,
            'debt_eq': debt_eq, 'rev_cagr': rev_cagr, 'is_buyback': is_buyback,
            'pe': pe, 'cash': cash
        }
    except Exception:
        return {'score': 0, 'tier': 'Error', 'roe_3y':0, 'margins':0, 'debt_eq':0, 'rev_cagr':0, 'is_buyback':False, 'pe':0, 'cash': 0}

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

def auto_dca_gate(ind: pd.DataFrame):
    if len(ind) < 3: return False, {'reason': 'insufficient data'}
    D0, D1, D2 = ind.iloc[-1], ind.iloc[-2], ind.iloc[-3]
    gap_pct = (D1['Open'] / D2['Close'] - 1) * 100.0
    if not np.isfinite(gap_pct) or gap_pct > RULES['autodca']['gap_thresh']:
        return False, {'reason': 'no qualifying gap', 'gap_pct': float(gap_pct)}
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

# ---------------- Commentary & Rendering ----------------

def is_euphoria(r: pd.Series) -> bool:
    return (r['Dist_to_52W_High_%'] > -3.5) and (r['Dist_to_SMA200_%'] > 50.0) and (r['RSI14'] >= 70.0)

def comment_for_row(r: pd.Series) -> str:
    d200, d52, rsi, sig = r['Dist_to_SMA200_%'], r['Dist_to_52W_High_%'], r['RSI14'], str(r.get('Signal', '')).upper()
    f_score = r.get('Fundy_Score', 0)
    
    # Matrix Logic
    if sig == 'BUY':
        if f_score >= 7: return f"<b>CORE BUY (High Conviction):</b> Strong technical uptrend backed by Fortress/Quality fundamentals ({f_score}/10). Good for sizing up."
        elif f_score <= 3: return f"<b>SPECULATIVE BUY (Trash Rally):</b> Uptrend in price, but weak fundamentals ({f_score}/10). Use tight stops; do not marry this trade."
        else: return f"Standard Buy: Uptrend intact (close > 200DMA). RSI {rsi:.0f} is constructive. Fundys OK ({f_score}/10)."
    elif sig == 'DCA':
        if f_score <= 3: return f"<b>AVOID (Falling Knife):</b> Technicals suggest a dip-buy, but fundamentals ({f_score}/10) are Junk. High risk of value trap."
        elif f_score >= 7: return f"<b>QUALITY DIP (Accumulate):</b> Fortress balance sheet ({f_score}/10) on sale near 200DMA. Ideal for measured adds."
        else: return f"DCA Zone: Trading near 200DMA (Δ {d200:.1f}%) with cooling RSI. Decent risk/reward for solid co."
    elif sig == 'WATCH':
        if is_euphoria(r):
            if f_score <= 3: return f"<b>EXIT WARNING:</b> Junk rally in euphoria zone ({abs(d52):.1f}% off high). Take profits."
            return f"Euphoria Zone: Price extended. Quality hold, but trim or trail stops."
        else:
            if f_score >= 7: return f"<b>GOLDEN WATCHLIST:</b> High quality ({f_score}/10) near inflection. Wait for setup."
            return f"Watch: {abs(d52):.1f}% off highs. Momentum mixed."
    elif sig == 'AVOID':
        if f_score >= 7: return f"<b>VALUE WATCH:</b> Great business ({f_score}/10) in a downtrend. Do not catch falling knife; wait for base."
        return f"Avoid: Weak trend (Δ 200DMA {d200:.1f}%) matches weak fundamentals. Sidelines."
    return "Neutral."

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

def mini_spark(ind):
    spark_ind = ind.tail(SPARK_DAYS)
    fig = go.Figure(go.Scatter(x=spark_ind['Date'], y=spark_ind['Close'], mode='lines', line=dict(width=1, color='#94a3b8')))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=50, width=120, xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'staticPlot': True})

def parse_announcements(market_code):
    # Only parse PDFs for ASX (Original logic)
    if market_code != 'AUS': return pd.DataFrame(columns=['Date', 'Ticker', 'Type', 'Tag', 'Headline', 'Details', 'Path', 'Recent'])
    
    NEWS_TYPES_REGEX = [
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

    rows = []
    today_syd = datetime.now(zoneinfo.ZoneInfo('Australia/Sydney')).date()
    
    if not os.path.isdir(ANN_DIR): return pd.DataFrame(columns=['Date', 'Ticker', 'Type', 'Tag', 'Headline', 'Details', 'Path', 'Recent'])
    for fp in sorted(glob.glob(os.path.join(ANN_DIR, '*.pdf'))):
        fname = os.path.basename(fp)
        m = re.match(r'([A-Z]{2,4})[_-]', fname)
        ticker = m.group(1) if m else None
        text = read_pdf_first_text(fp)
        _type, tag = next(( (l, t) for l, p, t in NEWS_TYPES_REGEX if re.search(p, fname + " " + text, re.I) ), ('Announcement', 'gen'))
        d = None
        if (md := re.search(r'(\d{1,2}\s+[A-Za-z]{3,9}\s+20\d{2})', text)):
            try: d = datetime.strptime(md.group(1), '%d %B %Y')
            except: 
                try: d = datetime.strptime(md.group(1), '%d %b %Y')
                except: pass
        if d is None: d = datetime.fromtimestamp(os.path.getmtime(fp)).replace(tzinfo=None)
        details = parse_3y_stats(text) if _type == 'Appendix 3Y' else None
        is_recent = (today_syd - d.date()).days <= NEWS_WINDOW_DAYS
        rows.append({'Date': d.date().isoformat(), 'Ticker': ticker or '', 'Type': _type, 'Tag': tag, 'Headline': _type, 'Details': details or '', 'Path': fp, 'Recent': is_recent})
    return pd.DataFrame(rows)

# ---------------- The Analysis Pipeline ----------------

def process_market(market_code, market_conf):
    print(f"--> Analyzing {market_conf['name']}...")
    snaps = []
    frames = []
    
    # 1. Fetch News (Regional)
    news_df = parse_announcements(market_code)
    
    # 2. Main Loop per Ticker
    for t_key, t_meta in market_conf['tickers'].items():
        full_sym = f"{t_key}{market_conf['suffix']}"
        
        # Fetch Data
        df = fetch_prices(full_sym, market_conf['tz'])
        if df.empty: continue
        df['Ticker'] = t_key
        frames.append(df)
        
        fundy = fetch_deep_fundamentals(full_sym)
        
        ind = indicators(df).dropna(subset=['SMA200', 'SMA50', 'High20', 'RSI14', 'EMA21', 'Vol20', 'ATR14'])
        if ind.empty: continue
        last = ind.iloc[-1]
        sig = label_row(last)
        
        # Pattern Logic
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
        
        # --- START FIX ---
        pname = pats[0]['name'] if pats else ''
        pbias = pattern_bias(pname)
        sig_str = str(sig).lower()

        is_aligned = False
        if pbias == 'neutral' or sig_str == 'watch': 
            is_aligned = True
        elif pbias == 'bullish' and sig_str in ['buy', 'dca']: 
            is_aligned = True
        elif pbias == 'bearish' and sig_str == 'avoid': 
            is_aligned = True

        palign = 'ALIGNED' if is_aligned else 'CONFLICT'
        # --- END FIX ---

        snaps.append({
            'Ticker': t_key, 'Name': t_meta[0], 'Desc': t_meta[1],
            'LastDate': pd.to_datetime(last['Date']).strftime('%Y-%m-%d'), 'LastClose': float(last['Close']),
            'SMA20': float(last['SMA20']), 'SMA50': float(last['SMA50']), 'SMA200': float(last['SMA200']),
            'RSI14': float(last['RSI14']), 'High52W': float(last['High52W']),
            'Dist_to_52W_High_%': float(last['Dist_to_52W_High_%']), 'Dist_to_SMA200_%': float(last['Dist_to_SMA200_%']),
            'Signal': sig, 'SignalAuto': bool(signal_auto), 'Comment': None,
            'Flag': bool(flag_flag), '_flag_info': flag_det, 
            '_pattern_lines': pats[0]['lines'] if pats else None, # <--- Preserved exact logic to avoid KeyError
            '_pattern_name': pname, '_pattern_status': pats[0]['status'] if pats else '', '_pattern_conf': pats[0]['confidence'] if pats else np.nan,
            '_pattern_align': palign, 'AutoDCA_Flag': bool(gate_flag), 'AutoDCA_Gap_%': float(gate_det.get('gap_pct', np.nan)),
            'AutoDCA_ReclaimMid': bool(gate_det.get('reclaim_mid', False)), 'AutoDCA_AboveEMA21': bool(gate_det.get('above_ema21', False)),
            'AutoDCA_Fill_%': float(gate_det.get('gap_fill_%', np.nan)), 'BreakoutReady': bool(breakout_ready),
            'Breakout_Level': float(breakout_info.get('ceiling', np.nan)), 'Breakout_Stop': float(breakout_info.get('stop', np.nan)),
            # Fundamentals
            'Fundy_Score': fundy['score'], 'Fundy_Tier': fundy['tier'], 'Fundy_ROE': fundy['roe_3y'],
            'Fundy_Margin': fundy['margins'], 'Fundy_PE': fundy['pe'], 'Fundy_RevCAGR': fundy['rev_cagr'],
            'Fundy_Growth': fundy['rev_cagr'], 'Fundy_Cash': fundy['cash'], 'Fundy_Debt': 0 if fundy['debt_eq'] == 999 else fundy['debt_eq']
        })

    prices_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    snaps_df = pd.DataFrame(snaps)
    
    # Post-Processing (Commentary & Plots)
    rows = []
    if not snaps_df.empty:
        for _, r in snaps_df.iterrows():
            r = r.copy()
            t = r['Ticker']
            df = prices_all[prices_all['Ticker'] == t].copy()
            ind = indicators(df)
            
            # Generate Comment
            r['Comment'] = comment_for_row(r)
            if r.get('BreakoutReady', False): r['Comment'] += f" • BreakoutReady: cleared {r['Breakout_Level']:.2f}."
            if r.get('SignalAuto', False): r['Comment'] += " • Auto-upgraded (DT invalidation)."
            
            # News Append
            if not news_df.empty:
                nd = news_df[(news_df['Ticker'] == t) & (news_df['Recent'])]
                if not nd.empty:
                    top = nd.sort_values('Date').iloc[-1]
                    badge = 'Director sale' if top['Type'] == 'Appendix 3Y' and 'Disposed' in (top['Details'] or '') else top['Type']
                    r['Comment'] += f" • News: {badge} ({top['Date']})"

            # Charts
            r['_mini_spark'] = mini_spark(ind)
            r['_mini_candle'] = mini_candle(ind, r['_flag_info'] if r['Flag'] else None, r['_pattern_lines'])
            rows.append(r)

    final_df = pd.DataFrame(rows)
    return final_df, news_df

# ---------------- HTML Generation ----------------

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

/* Tab Navigation */
.market-tabs { position: sticky; top: 0; z-index: 200; background: #020617; border-bottom: 1px solid var(--border); display: flex; justify-content: center; gap: 10px; padding: 10px; }
.market-tab { background: transparent; border: 1px solid var(--text-muted); color: var(--text-muted); padding: 8px 20px; border-radius: 99px; cursor: pointer; font-weight: 600; transition: 0.2s; }
.market-tab.active { background: var(--primary); border-color: var(--primary); color: white; }

.nav-wrapper { position: sticky; top: 53px; z-index: 100; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border); padding: 10px 16px; }
.nav-inner { display: flex; align-items: center; gap: 12px; max-width: 1200px; margin: 0 auto; overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: none; }
.nav-inner::-webkit-scrollbar { display: none; }
.nav-link { white-space: nowrap; color: var(--text-muted); text-decoration: none; padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 500; background: rgba(255,255,255,0.03); border: 1px solid transparent; transition: all 0.2s; }
.nav-link:hover, .nav-link.active { background: rgba(255,255,255,0.1); color: white; border-color: rgba(255,255,255,0.1); }

.market-container { display: none; }
.market-container.active { display: block; animation: fadein 0.3s; }
@keyframes fadein { from { opacity: 0; } to { opacity: 1; } }

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

@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
.pulse { animation: pulse 2s infinite; }
.euphoria-glow { box-shadow: 0 0 15px rgba(245, 158, 11, 0.15); border-color: rgba(245, 158, 11, 0.3); }

.kpi-scroll { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; margin-bottom: 24px; scrollbar-width: none; }
.kpi-scroll::-webkit-scrollbar { display: none; }
.kpi-card { min-width: 140px; background: var(--surface-1); border-radius: 12px; padding: 12px; border: 1px solid var(--border); display: flex; flex-direction: column; justify-content: center; }
.kpi-val { font-size: 24px; font-weight: 700; line-height: 1; margin-top: 4px; }
.kpi-lbl { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }

.table-responsive { overflow-x: auto; border-radius: 12px; border: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 12px 16px; color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border); background: rgba(15, 23, 42, 0.5); white-space: nowrap; }
td { padding: 12px 16px; border-bottom: 1px solid var(--border); vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(255,255,255,0.02); }
.chart-container { margin-top: 10px; border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05); }
"""

JS = """
function switchMarket(code) {
    document.querySelectorAll('.market-container').forEach(el => el.classList.remove('active'));
    document.getElementById('cont-'+code).classList.add('active');
    document.querySelectorAll('.market-tab').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-'+code).classList.add('active');
}

function init() {
    document.querySelectorAll('.search-input').forEach(input => {
        input.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            // Only search inside the active market container
            const activeCont = document.querySelector('.market-container.active');
            if (!activeCont) return;
            activeCont.querySelectorAll('.searchable-item').forEach(item => {
                const text = item.innerText.toLowerCase();
                item.classList.toggle('hidden', !text.includes(query));
            });
        });
    });
    
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
window.addEventListener('DOMContentLoaded', init);
"""

def render_card(r, badge_type, curr):
    euphoria_cls = "euphoria-glow" if is_euphoria(r) else ""
    euphoria_tag = '<span class="badge" style="background:rgba(245,158,11,0.2);color:#fbbf24;margin-left:6px">Euphoria</span>' if is_euphoria(r) else ""
    news_tag = '<span class="badge news" style="margin-left:6px">News</span>' if "News:" in (r['Comment'] or "") else ""
    
    # Fundamental Badge
    score = r['Fundy_Score']
    s_badge = "shield-high" if score >= 7 else ("shield-low" if score <= 3 else "buy") 
    s_icon = "💎" if score == 10 else ("🛡️" if score >= 7 else ("⚠️" if score <= 3 else "⚖️"))
    fundy_html = f'<span class="badge {s_badge}" style="margin-left:6px">{s_icon} {score}/10 {r["Fundy_Tier"]}</span>'

    return f"""
    <div class="card searchable-item {euphoria_cls}">
        <div class="card-header">
            <div>
                <a href="#" class="ticker-badge mono">{r['Ticker']}</a>
                <span class="badge {badge_type}" style="margin-left:8px">{r['Signal']}</span>
                {fundy_html} {euphoria_tag} {news_tag}
                <div style="font-size:12px; color:var(--text-muted); margin-top:4px">{r['Name']}</div>
            </div>
            <div class="price-block">
                <div class="price-main mono">{curr}{r['LastClose']:.2f}</div>
                <div class="price-sub">{r['LastDate']}</div>
            </div>
        </div>
        <div class="metrics-row">
            <div class="metric"><label>RSI(14)</label><span class="mono" style="color:{'#ef4444' if r['RSI14']>70 else ('#10b981' if r['RSI14']>45 else '#f59e0b')}">{r['RSI14']:.0f}</span></div>
            <div class="metric"><label>vs 200DMA</label><span class="mono">{r['Dist_to_SMA200_%']:+.1f}%</span></div>
            <div class="metric"><label>vs 52W High</label><span class="mono">{r['Dist_to_52W_High_%']:+.1f}%</span></div>
        </div>
        <div class="comment-box">{r['Comment']}</div>
        <div class="chart-container">{r['_mini_candle']}</div>
    </div>
    """

def render_kpi(label, val, color_cls):
    return f"""<div class="kpi-card"><div class="kpi-lbl">{label}</div><div class="kpi-val {color_cls}">{val}</div></div>"""

def render_market_html(m_code, m_conf, snaps_df, news_df):
    if snaps_df.empty: return f"<div id='cont-{m_code}' class='market-container'><div style='padding:50px;text-align:center'>No Data Found for {m_code}</div></div>"
    
    # Grouping
    BUY = snaps_df[snaps_df.Signal == 'BUY'].sort_values(['Fundy_Score', 'Dist_to_52W_High_%'], ascending=[False, False])
    DCA = snaps_df[snaps_df.Signal == 'DCA'].sort_values(['Fundy_Score', 'Dist_to_SMA200_%'], ascending=[False, True])
    WATCH = snaps_df[snaps_df.Signal == 'WATCH'].sort_values(['Fundy_Score', 'Dist_to_52W_High_%'], ascending=[False, False])
    AVOID = snaps_df[snaps_df.Signal == 'AVOID'].sort_values('Fundy_Score', ascending=True)
    
    GATE  = snaps_df[snaps_df['AutoDCA_Flag'] == True].sort_values('AutoDCA_Fill_%', ascending=False)
    PATS  = snaps_df[snaps_df['_pattern_name'] != ''].sort_values(['_pattern_conf', 'Ticker'], ascending=[False, True])
    
    BRKCOUNT = int(snaps_df['BreakoutReady'].sum())
    NEWSCOUNT = len(news_df)
    curr = m_conf['currency']

    # Cards HTML
    html_cards = []
    for section, df, badge in [('BUY — Actionable', BUY, 'buy'), ('DCA — Accumulate', DCA, 'dca'), ('WATCH — Monitoring', WATCH, 'watch'), ('AVOID — Sidelines', AVOID, 'avoid')]:
        if not df.empty:
            grid_items = "".join([render_card(r, badge, curr) for _, r in df.iterrows()])
            html_cards.append(f"<h2 id='{m_code}-{badge}' style='margin-top:30px; font-size:18px; color:var(--text-muted)'>{section}</h2><div class='grid'>{grid_items}</div>")

    # KPI HTML
    counts = snaps_df['Signal'].value_counts()
    kpi_html = f"""
    <div class="kpi-scroll">
        {render_kpi('Buy Signals', counts.get('BUY', 0), 'text-green')}
        {render_kpi('DCA Zone', counts.get('DCA', 0), 'text-amber')}
        {render_kpi('Watchlist', counts.get('WATCH', 0), 'text-primary')}
        {render_kpi('Avoid', counts.get('AVOID', 0), 'text-red')}
        {render_kpi('Breakouts', BRKCOUNT, 'text-green')}
        {render_kpi('Patterns', len(PATS), 'text-purple')}
        {render_kpi('Recent News', NEWSCOUNT, 'text-main')}
    </div>
    """

    # Tables
    dca_rows = "".join([f"<tr class='searchable-item'><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td class='mono text-red'>{r['AutoDCA_Gap_%']:.1f}%</td><td class='mono'>{'Yes' if r['AutoDCA_ReclaimMid'] else 'No'}</td><td class='mono'>{'Yes' if r['AutoDCA_AboveEMA21'] else 'No'}</td><td class='mono'>{r['AutoDCA_Fill_%']:.1f}%</td><td>{r['_mini_spark']}</td></tr>" for _, r in GATE.iterrows()])
    pat_rows = "".join([f"<tr class='searchable-item'><td>{r['_pattern_name']}</td><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td class='mono'>{r['_pattern_status']}</td><td class='mono'>{r['_pattern_conf']:.2f}</td><td class='mono'>{r['_pattern_align']}</td><td>{r['_mini_candle']}</td></tr>" for _, r in PATS.iterrows()])
    news_rows = "".join([f"<tr class='searchable-item'><td class='mono' style='color:var(--text-muted)'>{r['Date']}</td><td><b>{r['Ticker']}</b></td><td><span class='badge news'>{r['Type']}</span></td><td>{r['Headline']}</td></tr>" for _, r in news_df.sort_values('Date', ascending=False).iterrows()]) if not news_df.empty else "<tr><td colspan='4' style='text-align:center; color:gray'>No news data (PDFs) available for this region.</td></tr>"

    def fmt_pe(x): return f"{x:.1f}" if x and x > 0 else "-"
    def fmt_pct(x): return f"{x*100:.1f}%" if x else "-"
    fundy_rows_list = []
    for _, r in snaps_df.sort_values('Fundy_Score', ascending=False).iterrows():
        score = r['Fundy_Score']
        if score >= 7: b_cls = 'shield-high'
        elif score <= 3: b_cls = 'shield-low'
        else: b_cls = 'watch'
        fundy_rows_list.append(f"<tr class='searchable-item'><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td><span class='badge {b_cls}'>{score}/10 {r['Fundy_Tier']}</span></td><td class='mono'>{fmt_pct(r['Fundy_ROE'])}</td><td class='mono'>{fmt_pct(r['Fundy_Margin'])}</td><td class='mono'>{fmt_pe(r['Fundy_PE'])}</td><td class='mono'>{fmt_pct(r['Fundy_RevCAGR'])}</td><td class='mono'>{r['Fundy_Debt']:.2f}</td></tr>")
    fundy_rows = "".join(fundy_rows_list)

    # Nav Bar inside Market Container
    nav_html = f"""
    <div class="nav-wrapper">
        <div class="nav-inner">
            <a href="#" class="nav-link active" style="font-weight:700; color:white">Overview</a>
            <a href="#{m_code}-buy" class="nav-link">Buy</a>
            <a href="#{m_code}-dca" class="nav-link">DCA</a>
            <a href="#{m_code}-watch" class="nav-link">Watch</a>
            <a href="#{m_code}-fundy" class="nav-link">Fundamentals</a>
            <a href="#{m_code}-gate" class="nav-link">Auto-Gate</a>
            <a href="#{m_code}-patterns" class="nav-link">Patterns</a>
            <a href="#{m_code}-news" class="nav-link">News</a>
        </div>
    </div>
    """

    return f"""
    <div id="cont-{m_code}" class="market-container {'active' if m_code=='AUS' else ''}">
        {nav_html}
        <div class="search-container">
            <input type="text" id="search-{m_code}" class="search-input" placeholder="Search {m_conf['name']}...">
        </div>
        <div class="container">
            <div style="margin-bottom:20px">
                <h1 style="font-size:24px; margin:0 0 4px 0">{m_conf['name']} Overview</h1>
                <div style="color:var(--text-muted); font-size:13px">Updated {datetime.now(zoneinfo.ZoneInfo(m_conf['tz'])).strftime('%I:%M %p %Z')}</div>
            </div>
            {kpi_html}
            {"".join(html_cards)}
            
            <h2 id="{m_code}-fundy" style="margin-top:40px">Fundamental Health Check</h2>
            <div class="card">
                <div class="playbook"><b>The TraderBruh Shield:</b> 💎 7-10 (Fortress), ⚖️ 4-6 (Quality), ⚠️ 0-3 (Junk).</div>
                <div class="table-responsive"><table><thead><tr><th>Ticker</th><th>Score</th><th>ROE</th><th>Margin</th><th>P/E</th><th>Rev Growth</th><th>Debt/Eq</th></tr></thead><tbody>{fundy_rows}</tbody></table></div>
            </div>

            <h2 id="{m_code}-gate" style="margin-top:40px">Auto-DCA Candidates</h2>
            <div class="card">
                <div class="playbook"><b>Playbook:</b> Gap-down < {RULES['autodca']['gap_thresh']}%, Reclaim Mid, > EMA21.</div>
                <div class="table-responsive"><table><thead><tr><th>Ticker</th><th>Gap %</th><th>Reclaim?</th><th>&gt; EMA21?</th><th>Gap-fill %</th><th>Trend</th></tr></thead><tbody>{dca_rows if dca_rows else "<tr><td colspan='6' style='text-align:center'>No setups.</td></tr>"}</tbody></table></div>
            </div>

            <h2 id="{m_code}-patterns" style="margin-top:40px">Patterns &amp; Structures</h2>
            <div class="card">
                <div class="playbook"><b>Playbook:</b> Confirmed Double Tops/Bottoms, Triangles (Last {PATTERN_LOOKBACK} days).</div>
                <div class="table-responsive"><table><thead><tr><th>Pattern</th><th>Ticker</th><th>Status</th><th>Conf</th><th>Align</th><th>Mini</th></tr></thead><tbody>{pat_rows if pat_rows else "<tr><td colspan='6' style='text-align:center'>No patterns.</td></tr>"}</tbody></table></div>
            </div>

            <h2 id="{m_code}-news" style="margin-top:40px">News</h2>
            <div class="card" style="padding:0">
                <div class="table-responsive"><table><thead><tr><th>Date</th><th>Ticker</th><th>Type</th><th>Headline</th></tr></thead><tbody>{news_rows}</tbody></table></div>
            </div>
            <div style="height:50px"></div>
        </div>
    </div>
    """

# ---------------- Main Execution ----------------

if __name__ == "__main__":
    print("Starting Global Analysis...")
    
    # 1. Analyze All Markets
    market_htmls = []
    tab_buttons = []
    
    for m_code, m_conf in MARKETS.items():
        df, news = process_market(m_code, m_conf)
        html_part = render_market_html(m_code, m_conf, df, news)
        market_htmls.append(html_part)
        
        active = 'active' if m_code == 'AUS' else ''
        tab_buttons.append(f"<button id='tab-{m_code}' class='market-tab {active}' onclick=\"switchMarket('{m_code}')\">{m_conf['name']}</button>")

    # 2. Construct Full Page
    full_html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TraderBruh Global Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>{CSS}</style>
        <script>{JS}</script>
    </head>
    <body>
        <div class="market-tabs">
            {''.join(tab_buttons)}
        </div>
        
        {''.join(market_htmls)}
        
        <div style="text-align:center; padding:20px; color:var(--text-muted); font-size:12px">
            Generated by TraderBruh Ultimate 4.0 • Global Edition • Not Financial Advice
        </div>
    </body>
    </html>
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f: f.write(full_html)
    print('Done:', OUTPUT_HTML)