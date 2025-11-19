# traderbruh_web_dashboard_gh.py
# TraderBruh — Web Dashboard for GitHub Pages (ASX TA + FA)
# UI Overhaul: Modern Dark/Glass Theme + Fundamental Analysis Shield

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
SYD = zoneinfo.ZoneInfo('Australia/Sydney')
FETCH_DAYS          = 900       # history window for OHLC
MINI_BARS           = 120       # mini-candle bars
SPARK_DAYS          = 90        # sparkline bars

OUTPUT_DIR          = "docs"    # GitHub Pages artifact root
OUTPUT_HTML         = os.path.join(OUTPUT_DIR, "index.html")

ANN_DIR             = 'announcements'     # drop ASX PDFs here
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

# Fundamental Thresholds (The Buffett/Lynch Filter)
FUNDY_RULES = {
    'roe_min': 0.15,          # 15% Return on Equity
    'margin_min': 0.10,       # 10% Net Margins
    'growth_min': 0.10,       # 10% Revenue Growth
    'peg_max': 2.0,           # PEG Ratio under 2 (Growth at a reasonable price)
    'pe_max': 25.0,           # P/E under 25 (Standard value check)
    'curr_ratio_min': 1.2     # Liquidity check
}

COMPANY_META = {
    'CSL': ('CSL Limited', 'Biotech: vaccines, plasma & specialty therapies.'),
    'COH': ('Cochlear Limited', 'Implantable hearing devices leader.'),
    'PME': ('Pro Medicus', 'Cloud radiology/AI (Visage).'),
    'XRO': ('Xero', 'SMB cloud accounting; strong ecosystem.'),
    'TNE': ('TechnologyOne', 'Enterprise SaaS ERP (gov/edu).'),
    'WTC': ('WiseTech', 'Logistics software (CargoWise).'),
    'NXT': ('NEXTDC', 'Carrier-neutral data centres (AI/Cloud).'),
    'MAQ': ('Macquarie Tech', 'Sovereign cloud & secure DCs.'),
    'PLS': ('Pilbara Minerals', 'Hard-rock lithium producer.'),
    'MIN': ('Mineral Resources', 'Mining services + Li/Fe.'),
    'TLX': ('Telix Pharma', 'Radiopharma diagnostics/Tx.'),
    'RMD': ('ResMed', 'Sleep apnea devices + digital.'),
    'PNV': ('PolyNovo', 'Regenerative wound care (BTM).'),
    'NAN': ('Nanosonics', 'Infection prevention (Trophon).'),
    'REA': ('REA Group', 'Real estate portals & data.'),
    'CAR': ('Carsales', 'Auto marketplace & adjacencies.'),
    'SEK': ('SEEK', 'Jobs marketplace & HR tech.'),
    'HUB': ('HUB24', 'Wealth platform.'),
    'NWL': ('Netwealth', 'Wealth platform peer to HUB.'),
    'CPU': ('Computershare', 'Registry services; rate-sensitive.'),
    'DRO': ('DroneShield', 'Counter-UAS & EW; defence pipeline.'),
    'EOS': ('Electro Optic Systems', 'Defence/space; turnaround.'),
    'ASB': ('Austal', 'Aluminium vessels; US Navy.'),
    'CDA': ('Codan', 'Comms & Minelab metal detection.'),
    'CVL': ('Civmec', 'Engineering/shipbuilding; defence.'),
    'GNP': ('GenusPlus', 'Grid & electrical infrastructure.'),
    'LTR': ('Liontown', 'Lithium developer (Kathleen Valley).'),
    'SYR': ('Syrah', 'Graphite & anode materials.'),
    'BRN': ('BrainChip', 'Neuromorphic AI IP.'),
    'NXL': ('Nuix', 'eDiscovery/forensics; recurring.'),
}
UNIVERSE = [(t, f'{t}.AX') for t in COMPANY_META.keys()]

# ---------------- Data Fetching (TA + FA) ----------------
def fetch_prices(symbol: str) -> pd.DataFrame:
    """Fetch OHLCV history."""
    df = yf.download(symbol, period=f'{FETCH_DAYS}d', interval='1d', auto_adjust=False, progress=False, group_by='column', prepost=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.xs(symbol, axis=1, level=-1, drop_level=True)
        except: df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df['Date'] = pd.to_datetime(df[date_col], utc=True)

    # Intraday stitch
    now_syd = datetime.now(SYD)
    last_date_syd = df['Date'].dt.tz_convert(SYD).dt.date.max()
    if (now_syd.time() >= time(10, 0) and last_date_syd < now_syd.date()):
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

    df['Date'] = df['Date'].dt.tz_convert(SYD).dt.tz_localize(None)
    return df.dropna(subset=['Close'])

def fetch_fundamentals(symbol: str):
    """Fetch key fundamental stats via yfinance info."""
    try:
        tick = yf.Ticker(symbol)
        info = tick.info
        
        # Extract safely with defaults
        data = {
            'roe': info.get('returnOnEquity', 0) or 0,
            'margins': info.get('profitMargins', 0) or 0,
            'curr_ratio': info.get('currentRatio', 0) or 0,
            'debt': info.get('totalDebt', 0) or 0,
            'cash': info.get('totalCash', 0) or 0,
            'rev_growth': info.get('revenueGrowth', 0) or 0,
            'peg': info.get('pegRatio', 0), # Can be None
            'pe': info.get('trailingPE', 0), # Can be None
            'mcap': info.get('marketCap', 0) or 0
        }
        
        # Calculate Score (0 to 5)
        score = 0
        # 1. Quality (ROE)
        if data['roe'] > FUNDY_RULES['roe_min']: score += 1
        # 2. Profitability (Margins)
        if data['margins'] > FUNDY_RULES['margin_min']: score += 1
        # 3. Safety (Liquidity OR Net Cash)
        if data['curr_ratio'] > FUNDY_RULES['curr_ratio_min'] or data['cash'] > data['debt']: score += 1
        # 4. Growth (Revenue)
        if data['rev_growth'] > FUNDY_RULES['growth_min']: score += 1
        # 5. Value (PEG or PE)
        val_ok = False
        if data['peg'] and 0 < data['peg'] < FUNDY_RULES['peg_max']: val_ok = True
        if data['pe'] and 0 < data['pe'] < FUNDY_RULES['pe_max']: val_ok = True
        if val_ok: score += 1
        
        data['score'] = score
        
        # Tier Label
        if score == 5: data['tier'] = 'Fortress'
        elif score >= 3: data['tier'] = 'Solid'
        elif score >= 1: data['tier'] = 'Weak'
        else: data['tier'] = 'Distress'
            
        return data
    except Exception as e:
        # Fallback if YF fails
        return {'roe':0, 'margins':0, 'curr_ratio':0, 'debt':0, 'cash':0, 'rev_growth':0, 'peg':0, 'pe':0, 'mcap':0, 'score':0, 'tier':'Unknown'}

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
        ticker = m.group(1) if (m := re.match(r'([A-Z]{2,4})[_-]', fname)) else None
        text = read_pdf_first_text(fp)
        _type, tag = next(( (l, t) for l, p, t in NEWS_TYPES if re.search(p, fname + " " + text, re.I) ), ('Announcement', 'gen'))
        d = None
        if (md := re.search(r'(\d{1,2}\s+[A-Za-z]{3,9}\s+20\d{2})', text)):
            try: d = datetime.strptime(md.group(1), '%d %B %Y')
            except: 
                try: d = datetime.strptime(md.group(1), '%d %b %Y')
                except: pass
        if d is None: d = datetime.fromtimestamp(os.path.getmtime(fp), tz=SYD).replace(tzinfo=None)
        details = parse_3y_stats(text) if _type == 'Appendix 3Y' else None
        rows.append({'Date': d.date().isoformat(), 'Ticker': ticker or '', 'Type': _type, 'Tag': tag, 'Headline': _type, 'Details': details or '', 'Path': fp})
    return pd.DataFrame(rows)

# ---------------- Commentary Logic ----------------
def is_euphoria(r: pd.Series) -> bool:
    return (r['Dist_to_52W_High_%'] > -3.5) and (r['Dist_to_SMA200_%'] > 50.0) and (r['RSI14'] >= 70.0)

def comment_for_row(r: pd.Series) -> str:
    d200, d52, rsi, sig = r['Dist_to_SMA200_%'], r['Dist_to_52W_High_%'], r['RSI14'], str(r.get('Signal', '')).upper()
    f_score = r.get('Fundy_Score', 0)
    
    # Fundamentals Modifier
    fundy_warn = ""
    if f_score <= 1 and sig in ['BUY', 'DCA']:
        fundy_warn = " ⚠️ Speculative (Low Fundamental Score). Size down."
    elif f_score == 5 and sig == 'WATCH':
        fundy_warn = " 💎 High Quality. Priority watchlist."
        
    base = ""
    if sig == 'BUY':
        base = f"Uptrend intact (close above 200DMA and prior 20-day breakout). RSI {rsi:.0f} is constructive."
    elif sig == 'DCA':
        base = f"Trading close to the 200DMA (Δ {d200:.1f}%), with RSI {rsi:.0f} cooling. Positive bias."
    elif sig == 'AVOID':
        base = f"Trend is weak (Δ to 200DMA {d200:.1f}%). RSI {rsi:.0f} confirms lack of momentum."
    elif sig == 'WATCH':
        if is_euphoria(r):
            base = f"Euphoria zone: {abs(d52):.1f}% off high, {d200:.1f}% above 200DMA. Risk elevated."
        else:
            dist_s = f"within {abs(d52):.1f}% of high" if d52 > -2 else (f"trading {d200:.1f}% above 200DMA" if d200 > 0 else f"{abs(d200):.1f}% below 200DMA")
            base = f"{dist_s}, RSI {rsi:.0f}. Near inflection point."
    else:
        base = "Neutral setup."
        
    return base + fundy_warn

# ---------------- Main Loop ----------------
frames, snaps = [], []
print("Fetching data...")

for t, y in UNIVERSE:
    # 1. Price Data
    df = fetch_prices(y)
    if df.empty: continue
    df['Ticker'] = t
    frames.append(df)
    
    # 2. Fundamental Data
    fundy = fetch_fundamentals(y)
    
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

    snaps.append({
        'Ticker': t, 'Name': COMPANY_META.get(t, ('', ''))[0], 'Desc': COMPANY_META.get(t, ('', ''))[1],
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
        'Fundy_Score': fundy['score'], 'Fundy_Tier': fundy['tier'], 'Fundy_ROE': fundy['roe'],
        'Fundy_Margin': fundy['margins'], 'Fundy_PE': fundy['pe'], 'Fundy_PEG': fundy['peg'],
        'Fundy_Growth': fundy['rev_growth'], 'Fundy_Cash': fundy['cash'], 'Fundy_Debt': fundy['debt']
    })

prices_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
snaps_df = pd.DataFrame(snaps)

news_df = parse_announcements()
today_syd = datetime.now(SYD).date()
news_df['Recent'] = news_df.apply(lambda r: (today_syd - datetime.strptime(r['Date'], '%Y-%m-%d').date()).days <= NEWS_WINDOW_DAYS if r['Date'] else False, axis=1)

rows = []
for _, r in snaps_df.iterrows():
    r = r.copy()
    t = r['Ticker']
    df = prices_all[prices_all['Ticker'] == t].copy()
    ind = indicators(df)
    r['Comment'] = comment_for_row(r)
    if r.get('BreakoutReady', False): r['Comment'] += f" • BreakoutReady: cleared {r['Breakout_Level']:.2f}."
    if r.get('SignalAuto', False): r['Comment'] += " • Auto-upgraded (DT invalidation)."
    
    nd = news_df[(news_df['Ticker'] == t) & (news_df['Recent'])]
    if not nd.empty:
        top = nd.sort_values('Date').iloc[-1]
        badge = 'Director sale' if top['Type'] == 'Appendix 3Y' and 'Disposed' in (top['Details'] or '') else top['Type']
        r['Comment'] += f" • News: {badge} ({top['Date']})"

    spark_ind = ind.tail(SPARK_DAYS)
    r['_mini_spark'] = pio.to_html(go.Figure(go.Scatter(x=spark_ind['Date'], y=spark_ind['Close'], mode='lines', line=dict(width=1, color='#94a3b8'))).update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=50, width=120, xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False
    ), include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'staticPlot': True})
    
    r['_mini_candle'] = mini_candle(ind, r['_flag_info'] if r['Flag'] else None, r['_pattern_lines'])
    rows.append(r)

snaps_df = pd.DataFrame(rows)

def rank(df):
    # Prioritize High Fundamental Score in ranking
    buy = df[df.Signal == 'BUY'].sort_values(['Fundy_Score', 'Dist_to_52W_High_%'], ascending=[False, False])
    dca = df[df.Signal == 'DCA'].sort_values(['Fundy_Score', 'Dist_to_SMA200_%'], ascending=[False, True])
    watch = df[df.Signal == 'WATCH'].sort_values(['Fundy_Score', 'Dist_to_52W_High_%'], ascending=[False, False])
    avoid = df[df.Signal == 'AVOID'].sort_values('Fundy_Score', ascending=True)
    return buy, dca, watch, avoid

BUY, DCA, WATCH, AVOID = rank(snaps_df)
GATE  = snaps_df[snaps_df['AutoDCA_Flag'] == True].sort_values('AutoDCA_Fill_%', ascending=False)
PATS  = snaps_df[snaps_df['_pattern_name'] != ''].sort_values(['_pattern_conf', 'Ticker'], ascending=[False, True])
NEWSCOUNT, BRKCOUNT = len(news_df), int(snaps_df['BreakoutReady'].sum()) if not snaps_df.empty else 0

# ---------------- HTML & CSS ----------------
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
.nav-inner { display: flex; align-items: center; gap: 12px; max-width: 1200px; margin: 0 auto; overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: none; }
.nav-inner::-webkit-scrollbar { display: none; }
.nav-link { white-space: nowrap; color: var(--text-muted); text-decoration: none; padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 500; background: rgba(255,255,255,0.03); border: 1px solid transparent; transition: all 0.2s; }
.nav-link:hover, .nav-link.active { background: rgba(255,255,255,0.1); color: white; border-color: rgba(255,255,255,0.1); }

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
function init() {
    const searchInput = document.getElementById('globalSearch');
    const searchableItems = document.querySelectorAll('.searchable-item');
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        searchableItems.forEach(item => {
            const text = item.innerText.toLowerCase();
            item.classList.toggle('hidden', !text.includes(query));
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

def render_card(r, badge_type):
    euphoria_cls = "euphoria-glow" if is_euphoria(r) else ""
    euphoria_tag = '<span class="badge" style="background:rgba(245,158,11,0.2);color:#fbbf24;margin-left:6px">Euphoria</span>' if is_euphoria(r) else ""
    news_tag = '<span class="badge news" style="margin-left:6px">News</span>' if "News:" in (r['Comment'] or "") else ""
    
    # Fundamental Badge
    score = r['Fundy_Score']
    s_badge = "shield-high" if score >= 3 else "shield-low"
    s_icon = "💎" if score == 5 else ("🛡️" if score >= 3 else "⚠️")
    fundy_html = f'<span class="badge {s_badge}" style="margin-left:6px">{s_icon} {score}/5 {r["Fundy_Tier"]}</span>'

    return f"""
    <div class="card searchable-item {euphoria_cls}">
        <div class="card-header">
            <div>
                <a href="https://au.finance.yahoo.com/quote/{r['Ticker']}.AX" target="_blank" class="ticker-badge mono">{r['Ticker']}</a>
                <span class="badge {badge_type}" style="margin-left:8px">{r['Signal']}</span>
                {fundy_html} {euphoria_tag} {news_tag}
                <div style="font-size:12px; color:var(--text-muted); margin-top:4px">{r['Name']}</div>
            </div>
            <div class="price-block">
                <div class="price-main mono">${r['LastClose']:.2f}</div>
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

html_cards = []
for section, df, badge in [('BUY — Actionable', BUY, 'buy'), ('DCA — Accumulate', DCA, 'dca'), ('WATCH — Monitoring', WATCH, 'watch'), ('AVOID — Sidelines', AVOID, 'avoid')]:
    if not df.empty:
        grid_items = "".join([render_card(r, badge) for _, r in df.iterrows()])
        html_cards.append(f"<h2 id='{badge}' style='margin-top:30px; font-size:18px; color:var(--text-muted)'>{section}</h2><div class='grid'>{grid_items}</div>")

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

dca_rows = "".join([f"<tr class='searchable-item'><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td class='mono text-red'>{r['AutoDCA_Gap_%']:.1f}%</td><td class='mono'>{'Yes' if r['AutoDCA_ReclaimMid'] else 'No'}</td><td class='mono'>{'Yes' if r['AutoDCA_AboveEMA21'] else 'No'}</td><td class='mono'>{r['AutoDCA_Fill_%']:.1f}%</td><td>{r['_mini_spark']}</td></tr>" for _, r in GATE.iterrows()])
pat_rows = "".join([f"<tr class='searchable-item'><td>{r['_pattern_name']}</td><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td class='mono'>{r['_pattern_status']}</td><td class='mono'>{r['_pattern_conf']:.2f}</td><td class='mono'>{r['_pattern_align']}</td><td>{r['_mini_candle']}</td></tr>" for _, r in PATS.iterrows()])
news_rows = "".join([f"<tr class='searchable-item'><td class='mono' style='color:var(--text-muted)'>{r['Date']}</td><td><b>{r['Ticker']}</b></td><td><span class='badge news'>{r['Type']}</span></td><td>{r['Headline']}</td></tr>" for _, r in news_df.sort_values('Date', ascending=False).iterrows()])

# Fundy Rows
def fmt_pe(x): return f"{x:.1f}" if x and x > 0 else "-"
def fmt_pct(x): return f"{x*100:.1f}%" if x else "-"
fundy_rows = "".join([
    f"<tr class='searchable-item'><td><span class='ticker-badge mono'>{r['Ticker']}</span></td>"
    f"<td><span class='badge {'shield-high' if r['Fundy_Score']>=3 else 'shield-low'}'>{r['Fundy_Score']}/5 {r['Fundy_Tier']}</span></td>"
    f"<td class='mono'>{fmt_pct(r['Fundy_ROE'])}</td>"
    f"<td class='mono'>{fmt_pct(r['Fundy_Margin'])}</td>"
    f"<td class='mono'>{fmt_pe(r['Fundy_PE'])}</td>"
    f"<td class='mono'>{fmt_pct(r['Fundy_Growth'])}</td>"
    f"<td class='mono'>{r['Fundy_Cash']/1e6:.0f}M / {r['Fundy_Debt']/1e6:.0f}M</td>"
    f"</tr>"
    for _, r in snaps_df.sort_values('Fundy_Score', ascending=False).iterrows()
])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TraderBruh Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>{CSS}</style>
    <script>{JS}</script>
</head>
<body>
    <div class="nav-wrapper">
        <div class="nav-inner">
            <a href="#" class="nav-link active" style="font-weight:700; color:white">TraderBruh</a>
            <a href="#buy" class="nav-link">Buy</a>
            <a href="#dca" class="nav-link">DCA</a>
            <a href="#watch" class="nav-link">Watch</a>
            <a href="#fundy" class="nav-link">Fundamentals</a>
            <a href="#gate" class="nav-link">Auto-Gate</a>
            <a href="#patterns" class="nav-link">Patterns</a>
            <a href="#news" class="nav-link">News</a>
        </div>
    </div>

    <div class="search-container">
        <input type="text" id="globalSearch" class="search-input" placeholder="Search tickers, signals, or patterns...">
    </div>

    <div class="container">
        <div style="margin-bottom:20px">
            <h1 style="font-size:24px; margin:0 0 4px 0">Market Overview</h1>
            <div style="color:var(--text-muted); font-size:13px">ASX Technical Analysis & Fundamentals • Updated {datetime.now(SYD).strftime('%I:%M %p %Z')}</div>
        </div>
        
        {kpi_html}
        {"".join(html_cards)}

        <h2 id="fundy" style="margin-top:40px">Fundamental Health Check</h2>
        <div class="card">
            <div class="playbook">
                <b>The "Buffett Filter" (0-5 Score):</b><br>
                We check 5 pillars for every stock to separate "Quality" from "Trash".<br>
                1. <b>Quality:</b> ROE > 15% (Efficient management).<br>
                2. <b>Profitability:</b> Net Margins > 10% (Pricing power).<br>
                3. <b>Safety:</b> Current Ratio > 1.2 or Cash > Debt (Can they pay bills?).<br>
                4. <b>Growth:</b> Revenue Growth > 10% (Is the business expanding?).<br>
                5. <b>Value:</b> PEG < 2.0 or P/E < 25 (Are you overpaying?).<br><br>
                <b>Use:</b> If a stock is a "BUY" technically but has a <b>1/5 Score</b>, it is a speculative trade. Size down and use tight stops.
            </div>
            <div class="table-responsive">
                <table>
                    <thead><tr><th>Ticker</th><th>Score</th><th>ROE</th><th>Margin</th><th>P/E</th><th>Rev Growth</th><th>Cash / Debt</th></tr></thead>
                    <tbody>{fundy_rows}</tbody>
                </table>
            </div>
        </div>

        <h2 id="gate" style="margin-top:40px">Auto-DCA Candidates</h2>
        <div class="card">
            <div class="playbook">
                <b>Playbook:</b> Gap-down reaction setups. Focus on tickers where the gap is small (≤ {RULES['autodca']['gap_thresh']}%),
                close reclaimed the prior midpoint, and price is above EMA21. Scale in small.
            </div>
            <div class="table-responsive">
                <table>
                    <thead><tr><th>Ticker</th><th>Gap %</th><th>Reclaim Mid?</th><th>&gt; EMA21?</th><th>Gap-fill %</th><th>Trend</th></tr></thead>
                    <tbody>{dca_rows if dca_rows else "<tr><td colspan='6' style='text-align:center; color:var(--text-muted)'>No active setups today</td></tr>"}</tbody>
                </table>
            </div>
        </div>

        <h2 id="patterns" style="margin-top:40px">Patterns &amp; Structures</h2>
        <div class="card">
            <div class="playbook">
                <b>Playbook:</b> High-conviction patterns (last {PATTERN_LOOKBACK} days). Use for context.<br>
                • <b>Bullish:</b> Ascending Triangles, Double Bottoms.<br>
                • <b>Bearish:</b> Double Tops, Head & Shoulders.<br>
                Check <b>Alignment</b> with the trend bucket (BUY/DCA vs AVOID).
            </div>
            <div class="table-responsive">
                <table>
                    <thead><tr><th>Pattern</th><th>Ticker</th><th>Status</th><th>Confidence</th><th>Alignment</th><th>Mini</th></tr></thead>
                    <tbody>{pat_rows if pat_rows else "<tr><td colspan='6' style='text-align:center; color:var(--text-muted)'>No patterns right now.</td></tr>"}</tbody>
                </table>
            </div>
        </div>

        <h2 id="news" style="margin-top:40px">Recent Announcements</h2>
        <div class="card" style="padding:0">
            <div class="table-responsive">
                <table>
                    <thead><tr><th>Date</th><th>Ticker</th><th>Type</th><th>Detail</th></tr></thead>
                    <tbody>{news_rows}</tbody>
                </table>
            </div>
        </div>
        
        <div style="text-align:center; margin-top:40px; color:var(--text-muted); font-size:12px">Generated by TraderBruh • Not Financial Advice</div>
    </div>
</body>
</html>
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_HTML, 'w', encoding='utf-8') as f: f.write(html)
print('Done:', OUTPUT_HTML)