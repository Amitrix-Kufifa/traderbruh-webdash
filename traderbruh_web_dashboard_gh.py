# traderbruh_web_dashboard_gh.py
# TraderBruh — Web Dashboard for GitHub Pages (ASX TA)

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
PATTERNS_CONFIRMED_ONLY = True  # Patterns tab shows only confirmed patterns

RULES = {
    'buy':     {'rsi_min': 45, 'rsi_max': 70},
    'dca':     {'rsi_max': 45, 'sma200_proximity': 0.05},
    'avoid':   {'death_cross': True},
    'autodca': {'gap_thresh': -2.0, 'fill_req': 50.0}  # gap <=-2% and ≥50% fill & above EMA21 & reclaim mid
}

# Breakout (DT invalidation) rules
BREAKOUT_RULES = {
    'atr_mult':   0.50,   # must clear DT ceiling by ≥ 0.5×ATR(14)
    'vol_mult':   1.30,   # volume ≥ 1.3×Vol20
    'buffer_pct': 0.003,  # and ≥ 0.30% above ceiling
}
# Auto-upgrade Signal when DT breakout is ready
AUTO_UPGRADE_BREAKOUT = True

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

# ---------------- Data & TA ----------------
def fetch(symbol: str) -> pd.DataFrame:
    # Download daily data
    df = yf.download(
        symbol,
        period=f'{FETCH_DAYS}d',
        interval='1d',
        auto_adjust=False,
        progress=False,
        group_by='column',
        prepost=False
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # Handle multi-index columns from yfinance (e.g. ('Open', 'DRO.AX'))
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, axis=1, level=-1, drop_level=True)
        except Exception:
            df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV and bring index out as a Date column
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()

    # --- STEP 1: Parse ALL dates as UTC tz-aware (handles any mix cleanly) ---
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df['Date'] = pd.to_datetime(df[date_col], utc=True)

    # --- STEP 2: If today's daily candle isn't printed yet, stitch last 60m bar ---
    now_syd = datetime.now(SYD)
    # Compare last date in *Sydney* time to today's Sydney date
    last_date_syd = df['Date'].dt.tz_convert(SYD).dt.date.max()

    if (
        now_syd.time() >= time(16, 20) and
        last_date_syd < now_syd.date()
    ):
        intr = yf.download(
            symbol,
            period='5d',
            interval='60m',
            auto_adjust=False,
            progress=False,
            prepost=False,
            group_by='column'
        )
        if intr is not None and not intr.empty:
            if isinstance(intr.columns, pd.MultiIndex):
                try:
                    intr = intr.xs(symbol, axis=1, level=-1, drop_level=True)
                except Exception:
                    intr.columns = intr.columns.get_level_values(0)

            intr = intr.reset_index()
            # Also parse intraday times as UTC tz-aware
            intr['Date'] = pd.to_datetime(intr[intr.columns[0]], utc=True)

            last = intr.tail(1).iloc[0]
            top = pd.DataFrame([{
                'Date': last['Date'],     # still tz-aware UTC here
                'Open': float(last['Open']),
                'High': float(last['High']),
                'Low': float(last['Low']),
                'Close': float(last['Close']),
                'Volume': float(last['Volume']),
            }])

            df = pd.concat([df, top], ignore_index=True)

    # --- STEP 3: Convert everything to Sydney, then drop timezone (tz-naive) ---
    df['Date'] = df['Date'].dt.tz_convert(SYD).dt.tz_localize(None)

    # Now Date is a clean, tz-naive datetime64[ns] series – no mixing possible
    return df.dropna(subset=['Close'])


    # ✅ No extra pd.to_datetime(df['Date']) here – everything is already consistent
    return df.dropna(subset=['Close'])

    # Force datetime again just in case
    df['Date'] = pd.to_datetime(df['Date'])

    return df.dropna(subset=['Close'])


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values('Date').reset_index(drop=True)
    x['SMA20']  = x['Close'].rolling(20).mean()
    x['SMA50']  = x['Close'].rolling(50).mean()
    x['SMA200'] = x['Close'].rolling(200).mean()
    x['EMA21']  = x['Close'].ewm(span=21, adjust=False).mean()
    x['High20'] = x['High'].rolling(20).max()
    x['High52W']= x['High'].rolling(252).max()
    x['Vol20']  = x['Volume'].rolling(20).mean()

    # RSI14
    chg = x['Close'].diff()
    gains = chg.clip(lower=0).rolling(14).mean()
    losses = (-chg).clip(lower=0).rolling(14).mean()
    RS = gains / losses
    x['RSI14'] = 100 - (100 / (1 + RS))

    x['Dist_to_52W_High_%'] = (x['Close'] / x['High52W'] - 1) * 100.0
    x['Dist_to_SMA200_%']   = (x['Close'] / x['SMA200']  - 1) * 100.0

    # ATR(14)
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
    if buy_ok:
        return 'BUY'
    if dca_ok:
        return 'DCA'
    if avoid:
        return 'AVOID'
    return 'WATCH'

def auto_dca_gate(ind: pd.DataFrame):
    if len(ind) < 3:
        return False, {'reason': 'insufficient data'}
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
    return flag, {
        'gap_pct': float(gap_pct),
        'reclaim_mid': reclaim_mid,
        'above_ema21': above_ema21,
        'gap_fill_%': fill_pct
    }

# -------- Patterns --------
def _pivots(ind, window=PIVOT_WINDOW):
    v = ind.tail(PATTERN_LOOKBACK).reset_index(drop=True).copy()
    ph = (v['High'] == v['High'].rolling(window * 2 + 1, center=True).max())
    pl = (v['Low']  == v['Low'].rolling(window * 2 + 1, center=True).min())
    v['PH'] = ph.fillna(False)
    v['PL'] = pl.fillna(False)
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
            if lj - li < 10:
                continue
            p1, p2 = float(v.loc[li, 'Low']), float(v.loc[lj, 'Low'])
            if not _similar(p1, p2):
                continue
            neck = float(v.loc[li:lj, 'High'].max())
            confirmed = bool(v['Close'].iloc[-1] > neck)
            conf = 0.6 + (0.2 if confirmed else 0.0)
            if np.isfinite(v['Vol20'].iloc[-1]) and confirmed and v['Volume'].iloc[-1] > 1.2 * v['Vol20'].iloc[-1]:
                conf += 0.2
            lines = [
                ('h', v.loc[li, 'Date'], v.loc[lj, 'Date'], (p1 + p2) / 2.0),
                ('h', v.loc[li, 'Date'], v['Date'].iloc[-1], neck)
            ]
            out.append({
                'name': 'Double Bottom',
                'status': 'confirmed' if confirmed else 'forming',
                'confidence': round(min(conf, 1.0), 2),
                'levels': {
                    'base': round((p1 + p2) / 2.0, 4),
                    'neckline': round(neck, 4)
                },
                'lines': lines
            })
            return out
    return out

def detect_double_top(ind):
    v = _pivots(ind)
    highs = v.index[v['PH']].tolist()
    out = []
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            hi, hj = highs[i], highs[j]
            if hj - hi < 10:
                continue
            p1, p2 = float(v.loc[hi, 'High']), float(v.loc[hj, 'High'])
            if not _similar(p1, p2):
                continue
            neck = float(v.loc[hi:hj, 'Low'].min())
            confirmed = bool(v['Close'].iloc[-1] < neck)
            conf = 0.6 + (0.2 if confirmed else 0.0)
            if np.isfinite(v['Vol20'].iloc[-1]) and confirmed and v['Volume'].iloc[-1] > 1.2 * v['Vol20'].iloc[-1]:
                conf += 0.2
            lines = [
                ('h', v.loc[hi, 'Date'], v.loc[hj, 'Date'], (p1 + p2) / 2.0),
                ('h', v.loc[hi, 'Date'], v['Date'].iloc[-1], neck)
            ]
            out.append({
                'name': 'Double Top',
                'status': 'confirmed' if confirmed else 'forming',
                'confidence': round(min(conf, 1.0), 2),
                'levels': {
                    'ceiling': round((p1 + p2) / 2.0, 4),
                    'neckline': round(neck, 4)
                },
                'lines': lines
            })
            return out
    return out

def detect_inverse_hs(ind):
    v = _pivots(ind)
    lows = v.index[v['PL']].tolist()
    out = []
    for i in range(len(lows) - 2):
        l1, h, l2 = lows[i], lows[i + 1], lows[i + 2]
        pL1, pH, pL2 = float(v.loc[l1, 'Low']), float(v.loc[h, 'Low']), float(v.loc[l2, 'Low'])
        if not (pH < pL1 * (1 - 0.04) and pH < pL2 * (1 - 0.04)):
            continue
        if not _similar(pL1, pL2):
            continue
        left_high  = float(v.loc[l1:h, 'High'].max())
        right_high = float(v.loc[h:l2, 'High'].max())
        confirmed = bool(v['Close'].iloc[-1] > min(left_high, right_high))
        conf = 0.6 + (0.2 if confirmed else 0.0)
        if np.isfinite(v['Vol20'].iloc[-1]) and confirmed and v['Volume'].iloc[-1] > 1.2 * v['Vol20'].iloc[-1]:
            conf += 0.2
        lines = [('seg', v.loc[l1, 'Date'], left_high, v.loc[l2, 'Date'], right_high)]
        out.append({
            'name': 'Inverse H&S',
            'status': 'confirmed' if confirmed else 'forming',
            'confidence': round(min(conf, 1.0), 2),
            'levels': {
                'neck_left': round(left_high, 4),
                'neck_right': round(right_high, 4)
            },
            'lines': lines
        })
        return out
    return out

def detect_hs(ind):
    v = _pivots(ind)
    highs = v.index[v['PH']].tolist()
    out = []
    for i in range(len(highs) - 2):
        l1, h, l2 = highs[i], highs[i + 1], highs[i + 2]
        pL1, pH, pL2 = float(v.loc[l1, 'High']), float(v.loc[h, 'High']), float(v.loc[l2, 'High'])
        if not (pH > pL1 * (1 + 0.04) and pH > pL2 * (1 + 0.04)):
            continue
        if not _similar(pL1, pL2):
            continue
        left_low  = float(v.loc[l1:h, 'Low'].min())
        right_low = float(v.loc[h:l2, 'Low'].min())
        confirmed = bool(v['Close'].iloc[-1] < max(left_low, right_low))
        conf = 0.6 + (0.2 if confirmed else 0.0)
        if np.isfinite(v['Vol20'].iloc[-1]) and confirmed and v['Volume'].iloc[-1] > 1.2 * v['Vol20'].iloc[-1]:
            conf += 0.2
        lines = [('seg', v.loc[l1, 'Date'], left_low, v.loc[l2, 'Date'], right_low)]
        out.append({
            'name': 'Head & Shoulders',
            'status': 'confirmed' if confirmed else 'forming',
            'confidence': round(min(conf, 1.0), 2),
            'levels': {
                'neck_left': round(left_low, 4),
                'neck_right': round(right_low, 4)
            },
            'lines': lines
        })
        return out
    return out

def detect_triangles(ind):
    v = _pivots(ind)
    tail = v.tail(120).copy()
    phs = tail[tail['PH']]
    pls = tail[tail['PL']]
    out = []
    if len(phs) >= 2 and len(pls) >= 2:
        # Ascending: flat-ish highs, rising lows
        ph_vals = phs['High'].values
        for i in range(len(ph_vals) - 1):
            if _similar(ph_vals[i], ph_vals[i + 1]):
                res = (ph_vals[i] + ph_vals[i + 1]) / 2.0
                x = np.arange(len(pls))
                slope = np.polyfit(x, pls['Low'].values, 1)[0]
                if slope > 0:
                    confirmed = bool(tail['Close'].iloc[-1] > res)
                    conf = 0.55 + (0.25 if confirmed else 0.0)
                    lines = [
                        ('h', pls['Date'].iloc[0], tail['Date'].iloc[-1], res),
                        ('seg', pls['Date'].iloc[0], pls['Low'].iloc[0], pls['Date'].iloc[-1], pls['Low'].iloc[-1])
                    ]
                    out.append({
                        'name': 'Ascending Triangle',
                        'status': 'confirmed' if confirmed else 'forming',
                        'confidence': round(min(conf, 1.0), 2),
                        'levels': {'resistance': round(res, 4)},
                        'lines': lines
                    })
                    break
        # Descending: flat-ish lows, falling highs
        pl_vals = pls['Low'].values
        for i in range(len(pl_vals) - 1):
            if _similar(pl_vals[i], pl_vals[i + 1]):
                sup = (pl_vals[i] + pl_vals[i + 1]) / 2.0
                x = np.arange(len(phs))
                slope = np.polyfit(x, phs['High'].values, 1)[0]
                if slope < 0:
                    confirmed = bool(tail['Close'].iloc[-1] < sup)
                    conf = 0.55 + (0.25 if confirmed else 0.0)
                    lines = [
                        ('h', phs['Date'].iloc[0], tail['Date'].iloc[-1], sup),
                        ('seg', phs['Date'].iloc[0], phs['High'].iloc[0], phs['Date'].iloc[-1], phs['High'].iloc[-1])
                    ]
                    out.append({
                        'name': 'Descending Triangle',
                        'status': 'confirmed' if confirmed else 'forming',
                        'confidence': round(min(conf, 1.0), 2),
                        'levels': {'support': round(sup, 4)},
                        'lines': lines
                    })
                    break
    return out

def detect_flag(ind):
    if len(ind) < 60:
        return False, {}
    look = ind.tail(40)
    impulse = (look['Close'].max() / look['Close'].min() - 1) * 100
    if not np.isfinite(impulse) or impulse < 12:
        return False, {}
    win = 14
    tail = ind.tail(max(win, 8)).copy()
    x = np.arange(len(tail))
    hi = np.polyfit(x, tail['High'].values, 1)
    lo = np.polyfit(x, tail['Low'].values, 1)
    slope_pct = (hi[0] / tail['Close'].iloc[-1]) * 100
    ch = (np.polyval(hi, x[-1]) - np.polyval(lo, x[-1]))
    tight = ch <= max(0.4 * (look['Close'].max() - look['Close'].min()), 0.02 * tail['Close'].iloc[-1])
    gentle = (-0.006 <= slope_pct <= 0.002)
    return (tight and gentle), {'hi': hi.tolist(), 'lo': lo.tolist(), 'win': win}

def pattern_bias(name: str) -> str:
    if name in ("Double Bottom", "Inverse H&S", "Ascending Triangle", "Bull Flag"):
        return "bullish"
    if name in ("Double Top", "Head & Shoulders", "Descending Triangle"):
        return "bearish"
    return "neutral"

def signal_bias(sig: str) -> str:
    if sig in ("BUY", "DCA"):
        return "bullish"
    if sig == "AVOID":
        return "bearish"
    return "neutral"

# ---- DT Breakout Ready (invalidation)
def breakout_ready_dt(ind: pd.DataFrame, pat: dict, rules: dict):
    """Return (flag, info) if last close invalidates a Double Top by breaking the ceiling with volume."""
    if not pat or pat.get('name') != 'Double Top':
        return False, {}
    last = ind.iloc[-1]
    atr   = float(last.get('ATR14', np.nan))
    vol   = float(last.get('Volume', np.nan))
    vol20 = float(last.get('Vol20',  np.nan))
    ceiling = float(pat.get('levels', {}).get('ceiling', np.nan))
    if not (np.isfinite(atr) and np.isfinite(vol) and np.isfinite(vol20) and np.isfinite(ceiling)):
        return False, {}
    close = float(last['Close'])
    ok_price = (close >= ceiling * (1.0 + rules['buffer_pct'])) and (close >= ceiling + rules['atr_mult'] * atr)
    ok_vol   = (vol20 > 0) and (vol >= rules['vol_mult'] * vol20)
    ready = bool(ok_price and ok_vol)
    return ready, {
        'ceiling': round(ceiling, 4),
        'atr': round(atr, 4),
        'stop': round(close - atr, 4)
    }

# ---------------- Mini-charts ----------------
def _expand_lines(lines):
    out = []
    if not lines:
        return out
    for ln in lines:
        if ln[0] == 'h':
            _, d_left, d_right, y = ln
            out.append(('h', d_left, y, d_right, y))
        else:
            _, d1, y1, d2, y2 = ln
            out.append(('seg', d1, y1, d2, y2))
    return out

def mini_candle(ind, flag_info=None, pattern_lines=None):
    v = ind.tail(MINI_BARS).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=v['Date'],
        open=v['Open'],
        high=v['High'],
        low=v['Low'],
        close=v['Close'],
        hoverinfo='skip',
        showlegend=False
    ))
    if 'SMA20' in v.columns:
        fig.add_trace(go.Scatter(
            x=v['Date'],
            y=v['SMA20'],
            mode='lines',
            line=dict(width=1.4, color='rgba(56,189,248,0.9)'),
            hoverinfo='skip',
            showlegend=False
        ))
    if flag_info:
        win = flag_info.get('win', 14)
        t2 = ind.tail(max(win, 8)).copy()
        x = np.arange(len(t2))
        hi = np.poly1d(flag_info['hi'])
        lo = np.poly1d(flag_info['lo'])
        fig.add_trace(go.Scatter(
            x=t2['Date'],
            y=hi(x),
            mode='lines',
            line=dict(width=2, dash='dash', color='rgba(167,139,250,0.95)'),
            hoverinfo='skip',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=t2['Date'],
            y=lo(x),
            mode='lines',
            line=dict(width=2, dash='dash', color='rgba(167,139,250,0.95)'),
            hoverinfo='skip',
            showlegend=False
        ))
    if pattern_lines:
        for (kind, x1, y1, x2, y2) in _expand_lines(pattern_lines):
            if kind == 'h':
                fig.add_trace(go.Scatter(
                    x=[x1, x2],
                    y=[y1, y1],
                    mode='lines',
                    line=dict(width=2, color='rgba(34,197,94,0.95)', dash='dot'),
                    hoverinfo='skip',
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode='lines',
                    line=dict(width=2, color='rgba(234,179,8,0.95)'),
                    hoverinfo='skip',
                    showlegend=False
                ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=140,
        width=260,  # slightly narrower for mobile
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'displayModeBar': False})

# ---------------- News/Announcements (local PDFs) ----------------
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
    act = 'Disposed' if re.search(r'\bDisposed\b', text, re.I) else (
          'Acquired' if re.search(r'\bAcquired\b', text, re.I) else None)
    shares = None
    value = None
    m = re.search(r'(\d{1,3}(?:,\d{3}){1,3})\s+(?:ordinary|fully\s+paid|shares)', text, re.I)
    if m:
        shares = m.group(1)
    v = re.search(r'\$ ?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)', text)
    if v:
        value = v.group(1)
    parts = []
    if act:
        parts.append(act)
    if shares:
        parts.append(f"{shares} shares")
    if value:
        parts.append(f"A${value}")
    return ' • '.join(parts) if parts else None

def classify_text(text: str):
    for label, pat, tag in NEWS_TYPES:
        if re.search(pat, text, re.I):
            return label, tag
    return 'Announcement', 'gen'

def read_pdf_first_text(path: str):
    if not HAVE_PYPDF:
        return ''
    try:
        reader = PdfReader(path)
        if not reader.pages:
            return ''
        page = reader.pages[0]
        t = page.extract_text() or ''
        t = re.sub(r'[ \t]+', ' ', t)
        return t
    except Exception:
        return ''

def parse_announcements():
    rows = []
    if not os.path.isdir(ANN_DIR):
        return pd.DataFrame(columns=['Date', 'Ticker', 'Type', 'Tag', 'Headline', 'Details', 'Path'])
    for fp in sorted(glob.glob(os.path.join(ANN_DIR, '*.pdf'))):
        fname = os.path.basename(fp)
        ticker = None
        m = re.match(r'([A-Z]{2,4})[_-]', fname)
        if m:
            ticker = m.group(1)
        text = read_pdf_first_text(fp)
        _type, tag = classify_text(fname + " " + text)

        d = None
        md = re.search(r'(\d{1,2}\s+[A-Za-z]{3,9}\s+20\d{2})', text)
        if md:
            for fmt in ('%d %B %Y', '%d %b %Y'):
                try:
                    d = datetime.strptime(md.group(1), fmt)
                    break
                except Exception:
                    pass
        if d is None:
            d = datetime.fromtimestamp(os.path.getmtime(fp), tz=SYD).replace(tzinfo=None)

        details = parse_3y_stats(text) if _type == 'Appendix 3Y' else None
        rows.append({
            'Date': d.date().isoformat(),
            'Ticker': ticker or '',
            'Type': _type,
            'Tag': tag,
            'Headline': _type,
            'Details': details or '',
            'Path': fp
        })
    return pd.DataFrame(rows)

# ---------------- Commentary ----------------
def comment_for_row(r: pd.Series) -> str:
    d200 = r['Dist_to_SMA200_%']
    d52 = r['Dist_to_52W_High_%']
    rsi = r['RSI14']
    if r['Signal'] == 'BUY':
        return f'Uptrend intact (Close>SMA200), recent 20D breakout, RSI {rsi:.0f} constructive.'
    if r['Signal'] == 'DCA':
        return f'Near 200DMA (Δ {d200:.1f}%), RSI {rsi:.0f} soft — add on controlled pullbacks.'
    if r['Signal'] == 'AVOID':
        return f'Weak/declining trend (Δ to 200DMA {d200:.1f}%), RSI {rsi:.0f} — wait for reclaim.'
    if d52 > -2:
        return f'Within {abs(d52):.1f}% of 52W high — watch for breakout.'
    if d200 > 0 and not (45 <= rsi <= 70):
        return f'Above 200DMA but RSI {rsi:.0f} — wait for strength/setup.'
    return 'Neutral — build evidence (base, reclaim, or breakouts).'

# ---------------- Build dataset ----------------
frames, snaps = [], []
for t, y in UNIVERSE:
    df = fetch(y)
    if df.empty:
        continue
    df['Ticker'] = t
    frames.append(df)

    ind = indicators(df).dropna(subset=['SMA200', 'SMA50', 'High20', 'RSI14', 'EMA21', 'Vol20', 'ATR14'])
    if ind.empty:
        continue
    last = ind.iloc[-1]
    sig = label_row(last)

    # Flag & Patterns
    flag_flag, flag_det = detect_flag(ind)
    pats = []
    pats += detect_double_bottom(ind)
    pats += detect_double_top(ind)
    pats += detect_inverse_hs(ind)
    pats += detect_hs(ind)
    pats += detect_triangles(ind)
    if PATTERNS_CONFIRMED_ONLY:
        pats = [p for p in pats if p.get('status') == 'confirmed']

    # Breakout-ready (DT invalidation)
    breakout_ready, breakout_info = (False, {})
    dt_pats = [p for p in pats if p.get('name') == 'Double Top']
    if dt_pats:
        breakout_ready, breakout_info = breakout_ready_dt(ind, dt_pats[0], BREAKOUT_RULES)

    signal_auto = False
    if AUTO_UPGRADE_BREAKOUT and breakout_ready:
        sig = 'BUY'
        signal_auto = True

    # Auto-DCA gate
    gate_flag, gate_det = auto_dca_gate(ind)

    pname = pats[0]['name'] if pats else ''
    palign = 'ALIGNED' if (
        pattern_bias(pname) == signal_bias(sig)
        or pattern_bias(pname) == 'neutral'
        or signal_bias(sig) == 'neutral'
    ) else 'CONFLICT'

    snaps.append({
        'Ticker': t,
        'Name': COMPANY_META.get(t, ('', ''))[0],
        'Desc': COMPANY_META.get(t, ('', ''))[1],
        'LastDate': pd.to_datetime(last['Date']).strftime('%Y-%m-%d'),
        'LastClose': float(last['Close']),
        'SMA20': float(last['SMA20']),
        'SMA50': float(last['SMA50']),
        'SMA200': float(last['SMA200']),
        'RSI14': float(last['RSI14']),
        'High52W': float(last['High52W']),
        'Dist_to_52W_High_%': float(last['Dist_to_52W_High_%']),
        'Dist_to_SMA200_%': float(last['Dist_to_SMA200_%']),
        'Signal': sig,
        'SignalAuto': bool(signal_auto),
        'Comment': None,
        'Flag': bool(flag_flag),
        '_flag_info': flag_det,
        '_pattern_lines': pats[0]['lines'] if pats else None,
        '_pattern_name': pname,
        '_pattern_status': pats[0]['status'] if pats else '',
        '_pattern_conf': pats[0]['confidence'] if pats else np.nan,
        '_pattern_align': palign,
        '_mini_candle': None,
        '_mini_spark': None,
        'AutoDCA_Flag': bool(gate_flag),
        'AutoDCA_Gap_%': float(gate_det.get('gap_pct', np.nan)),
        'AutoDCA_ReclaimMid': bool(gate_det.get('reclaim_mid', False)),
        'AutoDCA_AboveEMA21': bool(gate_det.get('above_ema21', False)),
        'AutoDCA_Fill_%': float(gate_det.get('gap_fill_%', np.nan)),
        'BreakoutReady': bool(breakout_ready),
        'Breakout_Level': float(breakout_info.get('ceiling', np.nan)),
        'Breakout_Stop':  float(breakout_info.get('stop',    np.nan)),
    })

prices_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
snaps_df = pd.DataFrame(snaps)

# News ingest
news_df = parse_announcements()
today_syd = datetime.now(SYD).date()

def recent_marker(row):
    try:
        d = datetime.strptime(row['Date'], '%Y-%m-%d').date()
        return (today_syd - d).days <= NEWS_WINDOW_DAYS
    except Exception:
        return False

news_df['Recent'] = news_df.apply(recent_marker, axis=1)

# Attach comments & charts and fold in news markers
rows = []
for _, r in snaps_df.iterrows():
    r = r.copy()
    t = r['Ticker']
    df = prices_all[prices_all['Ticker'] == t].copy()
    ind = indicators(df)
    r['Comment'] = comment_for_row(r)

    if r.get('BreakoutReady', False):
        r['Comment'] += (
            f" • BreakoutReady: cleared DT ceiling {r['Breakout_Level']:.2f} "
            f"with volume; stop≈{r['Breakout_Stop']:.2f} (ATR14)."
        )
    if r.get('SignalAuto', False):
        r['Comment'] += " • Signal auto-upgraded to BUY (DT invalidation)."

    # Short news note (last 14 days)
    nd = news_df[news_df['Ticker'] == t]
    if not nd.empty:
        nd_recent = nd[nd['Recent']]
        if not nd_recent.empty:
            top = nd_recent.sort_values('Date').iloc[-1]
            badge = 'Director sale' if top['Type'] == 'Appendix 3Y' and 'Disposed' in (top['Details'] or '') else top['Type']
            r['Comment'] += f" • News: {badge} ({top['Date']})"

    # charts
    spark_ind = ind.tail(SPARK_DAYS)
    r['_mini_spark'] = pio.to_html(
        go.Figure(go.Scatter(
            x=spark_ind['Date'],
            y=spark_ind['Close'],
            mode='lines',
            line=dict(width=1)
        )).update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=80,
            width=220,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False
        ),
        include_plotlyjs=False,
        full_html=False,
        config={'displayModeBar': False}
    )
    r['_mini_candle'] = mini_candle(ind, r['_flag_info'] if r['Flag'] else None, r['_pattern_lines'])
    rows.append(r)

snaps_df = pd.DataFrame(rows)

# Rankings
def rank(df):
    buy = df[df.Signal == 'BUY'].copy()
    buy = buy.sort_values('Dist_to_52W_High_%', ascending=False) if not buy.empty else buy
    dca = df[df.Signal == 'DCA'].copy().sort_values(
        ['Dist_to_SMA200_%', 'RSI14'],
        ascending=[True, True]
    )
    watch = df[df.Signal == 'WATCH'].copy()
    watch = watch.sort_values('Dist_to_52W_High_%', ascending=False) if not watch.empty else watch
    avoid = df[df.Signal == 'AVOID'].copy().sort_values(
        ['Dist_to_SMA200_%', 'RSI14'],
        ascending=[True, True]
    )
    return buy, dca, watch, avoid

BUY, DCA, WATCH, AVOID = rank(snaps_df)
GATE  = snaps_df[snaps_df['AutoDCA_Flag'] == True].copy().sort_values('AutoDCA_Fill_%', ascending=False)
FLAGS = snaps_df[snaps_df['Flag'] == True].copy()
PATS  = snaps_df[snaps_df['_pattern_name'] != ''].copy().sort_values(
    ['_pattern_conf', 'Ticker'],
    ascending=[False, True]
)
NEWSCOUNT = len(news_df)
BRKCOUNT  = int(snaps_df['BreakoutReady'].sum()) if not snaps_df.empty else 0

# ---------------- HTML helpers ----------------
CSS = """
:root{
  --bg:#020617;
  --card:#0b1120;
  --ink:#e5f0ff;
  --muted:#94a3b8;
  --accent:#38bdf8;
  --green:#16a34a;
  --amber:#f59e0b;
  --red:#ef4444;
  --purple:#a855f7;
}
*{
  box-sizing:border-box;
  font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
}
html,body{
  margin:0;
  padding:0;
}
body{
  background:var(--bg);
  color:var(--ink);
  font-size:16px;
  line-height:1.5;
  overflow-x:hidden;
  -webkit-font-smoothing:antialiased;
}

/* -------- Shell -------- */
.container{
  width:100%;
  max-width:1100px;
  margin:0 auto;
  padding:12px 10px 32px;
}
.section{
  scroll-margin-top:76px;
  margin-top:18px;
}
.card{
  background:var(--card);
  border-radius:14px;
  padding:12px 12px 14px;
  box-shadow:0 0 0 1px rgba(15,23,42,.9);
}

/* -------- Typography -------- */
h1{
  margin:0 0 4px 0;
  font-size:clamp(22px,5.4vw,30px);
}
h2{
  margin:0 0 10px 0;
  font-size:clamp(18px,4.6vw,24px);
}
h3{
  margin:0 0 8px 0;
  font-size:16px;
  color:#cfe6ff;
}
small,
.smallmuted{
  color:var(--muted);
  font-size:12px;
}
.desc{
  color:var(--muted);
  font-size:13px;
}

/* -------- Top nav -------- */
.nav{
  position:sticky;
  top:0;
  z-index:50;
  background:rgba(2,6,23,.96);
  backdrop-filter:blur(12px);
  border-bottom:1px solid rgba(15,23,42,1);
}
.navinner{
  max-width:1100px;
  margin:0 auto;
  padding:8px 10px 10px;
  display:flex;
  flex-wrap:wrap;
  gap:6px;
  align-items:center;
}
.nav a{
  flex:0 0 auto;
  color:#cfe6ff;
  text-decoration:none;
  font-size:13px;
  padding:6px 10px;
  border-radius:999px;
  border:1px solid rgba(148,163,184,.4);
  background:rgba(15,23,42,.95);
}
.nav a:hover{
  background:rgba(148,163,184,.32);
}

/* -------- KPI row -------- */
.kpis{
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:10px;
  margin-top:10px;
}
.kpi{
  border-radius:12px;
  padding:10px 12px;
  font-weight:600;
  display:flex;
  flex-direction:column;
  gap:4px;
}
.kpi .label{
  font-size:11px;
  color:var(--muted);
}
.kpi .num{
  font-size:24px;
}
.kpi.buy{background:rgba(22,163,74,.18);color:#86efac;}
.kpi.dca{background:rgba(245,158,11,.18);color:#fde68a;}
.kpi.watch{background:rgba(56,189,248,.18);color:#7dd3fc;}
.kpi.avoid{background:rgba(239,68,68,.22);color:#fecaca;}
.kpi.flag{background:rgba(167,139,250,.22);color:#e9d5ff;}
.kpi.break{background:rgba(56,189,248,.18);color:#7dd3fc;}
.kpi.gate{background:rgba(245,158,11,.18);color:#fde68a;}
.kpi.pattern{background:rgba(94,234,212,.18);color:#a5f3fc;}
.kpi.news{background:rgba(59,130,246,.18);color:#bfdbfe;}

/* -------- Layout for panels -------- */
.grid{
  display:flex;
  flex-direction:column;
  gap:14px;
  margin-top:12px;
}
.panel{
  width:100%;
}

/* -------- Overview stock cards -------- */
.stock-list{
  display:flex;
  flex-direction:column;
  gap:10px;
  margin-top:6px;
}
.stock-card{
  border-radius:10px;
  background:#020617;
  padding:9px 10px 10px;
  border:1px solid rgba(148,163,184,.35);
}
.stock-card-header{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:8px;
}
.stock-card-last{
  font-size:18px;
  font-weight:600;
}
.stock-card-name{
  margin-top:2px;
}
.stock-card-metrics{
  display:flex;
  flex-wrap:wrap;
  gap:6px 14px;
  margin-top:6px;
  font-size:11px;
}
.stock-card-comment{
  margin-top:6px;
  font-size:13px;
}
.stock-card-metrics .label{
  color:var(--muted);
  margin-right:4px;
}
.stock-card-metrics .value{
  font-variant-numeric:tabular-nums;
}
.stock-card-chart{
  margin-top:6px;
}
.stock-card-chart .mini,
.stock-card-chart .spark,
.stock-card-chart iframe,
.stock-card-chart div{
  width:100% !important;
  max-width:100%;
}

/* -------- Tables -------- */
.table-wrap{
  width:100%;
  overflow-x:auto;
}
.table{
  width:100%;
  border-collapse:collapse;
}
.table th,
.table td{
  padding:8px 6px;
  border-bottom:1px solid rgba(255,255,255,.06);
  font-size:13px;
  vertical-align:middle;
}
.table th{
  color:#9fb3d9;
  text-align:left;
  background:rgba(255,255,255,.02);
}

/* -------- Badges & labels -------- */
.badge{
  display:inline-block;
  margin-left:6px;
  padding:.15rem .45rem;
  border-radius:999px;
  font-size:11px;
}
.badge.buy{background:rgba(22,163,74,.18);color:#86efac;}
.badge.dca{background:rgba(245,158,11,.18);color:#fde68a;}
.badge.watch{background:rgba(56,189,248,.18);color:#7dd3fc;}
.badge.avoid{background:rgba(239,68,68,.18);color:#fca5a5;}
.badge.flag{background:rgba(167,139,250,.24);color:#e9d5ff;}
.badge.newsneg{background:rgba(248,113,113,.2);color:#fecaca;}
.badge.newspos{background:rgba(22,163,74,.2);color:#bbf7d0;}

.patternbadge{
  display:inline-block;
  margin-left:4px;
  padding:2px 6px;
  border-radius:999px;
  font-size:11px;
  background:rgba(94,234,212,.18);
  color:#a5f3fc;
}

/* -------- Text helpers -------- */
.code{
  font-variant-numeric:tabular-nums;
}
.ticker{
  font-weight:700;
  color:#93c5fd;
  text-decoration:none;
}
.ticker:hover{
  text-decoration:underline;
  color:#bfdbfe;
}

/* -------- Mini / spark charts (fallback width) -------- */
.mini,
.spark{
  display:block;
  width:100% !important;
  max-width:100%;
  height:auto;
}

/* -------- Signals card layout -------- */
.signals-list{
  margin-top:10px;
  display:flex;
  flex-direction:column;
  gap:10px;
}
.signal-card{
  border-radius:12px;
  background:#020617;
  padding:10px 11px 11px;
  border:1px solid rgba(148,163,184,.4);
}
.signal-card-top{
  display:flex;
  justify-content:space-between;
  gap:8px;
  align-items:flex-start;
}
.signal-label{
  display:inline-block;
  font-size:11px;
  padding:2px 7px;
  border-radius:999px;
  margin-bottom:4px;
  background:rgba(148,163,184,.25);
}
.signal-label.buy{background:rgba(22,163,74,.2);color:#bbf7d0;}
.signal-label.dca{background:rgba(245,158,11,.2);color:#fde68a;}
.signal-label.watch{background:rgba(56,189,248,.2);color:#7dd3fc;}
.signal-label.avoid{background:rgba(239,68,68,.2);color:#fecaca;}
.signal-label.flag{background:rgba(167,139,250,.2);color:#e9d5ff;}

.signal-name{
  display:block;
  font-size:12px;
  color:var(--muted);
}
.signal-price{
  text-align:right;
  font-size:12px;
}
.signal-price .code{
  font-size:14px;
  font-weight:600;
}

.signal-metrics{
  margin-top:6px;
  display:flex;
  flex-wrap:wrap;
  gap:6px 14px;
  font-size:11px;
}
.signal-metrics .label{
  color:var(--muted);
  margin-right:4px;
}
.signal-metrics .value{
  font-variant-numeric:tabular-nums;
}

.signal-comment{
  margin-top:6px;
  font-size:13px;
}
.signal-chart{
  margin-top:6px;
}
.signal-chart .mini,
.signal-chart .spark,
.signal-chart iframe,
.signal-chart div{
  width:100% !important;
  max-width:100%;
}

/* -------- Misc helpers -------- */
.flex{
  display:flex;
  gap:8px;
  align-items:center;
  flex-wrap:wrap;
}
.footer{
  margin:24px 0 40px 0;
  color:var(--muted);
  font-size:12px;
}

/* -------- Mobile tweaks -------- */
@media (max-width: 480px){
  .container{
    padding:10px 8px 28px;
  }
  .kpis{
    grid-template-columns:1fr;
  }
}

/* -------- Desktop / tablet -------- */
@media (min-width: 900px){
  .container{
    padding:18px 20px 40px;
  }
  .kpis{
    grid-template-columns:repeat(4,minmax(0,1fr));
  }
  .grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:18px;
  }
}
"""


SORT_JS = """
function makeSortable(tableId){
  const table = document.getElementById(tableId); if(!table) return;
  const headers = table.querySelectorAll('th');
  headers.forEach((th, idx)=>{
    th.addEventListener('click', ()=>{
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const asc = th.dataset.asc === '1' ? false : true;
      rows.sort((a,b)=>{
        const ta = a.children[idx].innerText.trim();
        const tb = b.children[idx].innerText.trim();
        const na = parseFloat(ta.replace(/[^0-9.-]/g,'')); const nb = parseFloat(tb.replace(/[^0-9.-]/g,''));
        if(!isNaN(na) && !isNaN(nb)) return asc ? na-nb : nb-na;
        return asc ? ta.localeCompare(tb) : tb.localeCompare(ta);
      });
      rows.forEach(r=>tbody.appendChild(r));
      headers.forEach(h=>h.dataset.asc='');
      th.dataset.asc = asc ? '1' : '0';
    });
  });
}
function makeFilter(inputId, tableId){
  const input = document.getElementById(inputId);
  const table = document.getElementById(tableId);
  if(!input || !table) return;
  input.addEventListener('input', ()=>{
    const q = input.value.toLowerCase();
    table.querySelectorAll('tbody tr').forEach(tr=>{
      tr.style.display = tr.innerText.toLowerCase().includes(q) ? '' : 'none';
    });
  });
}
"""

def panel_overview(title, df, badge_class):
    """Render BUY / DCA / WATCH / AVOID buckets as mobile-friendly cards."""
    if df is None or len(df) == 0:
        return f"""
<div class="panel">
  <div class="card">
    <h3>{title}</h3>
    <p class="smallmuted">No tickers currently in this bucket.</p>
  </div>
</div>
"""

    cards = []
    for _, r in df.iterrows():
        # Mini chart HTML
        mini_html = r.get('_mini_candle', '') or ''

        # Re-use the existing per-row commentary you already build via comment_for_row(...)
        comment = r.get('Comment') or ''

        # Only show this prominently for WATCH bucket
        comment_html = ""
        if badge_class == 'watch' and comment:
            comment_html = f"""
  <div class="stock-card-comment desc">
    {comment}
  </div>"""

        cards.append(f"""
<div class="stock-card">
  <div class="stock-card-header">
    <div>
      <a class="ticker"
         href="https://au.finance.yahoo.com/quote/{r['Ticker']}.AX"
         target="_blank" rel="noopener">
        {r['Ticker']}
      </a>
      <span class="badge {badge_class}">{badge_class.upper()}</span>
    </div>
    <div class="stock-card-last code">{r['LastClose']:.4f}</div>
  </div>

  <div class="stock-card-name desc">{r['Name']}</div>

  <div class="stock-card-metrics">
    <div><span class="label">RSI14</span>
         <span class="value code">{r['RSI14']:.2f}</span></div>
    <div><span class="label">→52W%</span>
         <span class="value code">{r['Dist_to_52W_High_%']:.2f}%</span></div>
    <div><span class="label">→200DMA%</span>
         <span class="value code">{r['Dist_to_SMA200_%']:.2f}%</span></div>
  </div>{comment_html}

  <div class="stock-card-chart">
    {mini_html}
  </div>
</div>
""")

    return f"""
<div class="panel">
  <div class="card">
    <h3>{title}</h3>
    <div class="stock-list">
      {''.join(cards)}
    </div>
  </div>
</div>
"""




def make_table(table_id, cols, rows_html, with_search=True):
    search = (
        f'<div class="card search-card"><input id="{table_id}_q" '
        f'placeholder="Type to filter..." '
        f'style="width:100%;padding:10px;border-radius:10px;'
        f'border:1px solid rgba(255,255,255,.08);'
        f'background:#0d152b;color:#e7f1ff"/></div>'
        if with_search else ''
    )
    return f"""
{search}
<div class="card">
  <div class="table-wrap">
    <table class="table" id="{table_id}">
      <thead><tr>{''.join(f'<th>{c}</th>' for c in cols)}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>
<script>
  makeSortable("{table_id}");
  {"makeFilter('" + table_id + "_q', '" + table_id + "');" if with_search else ""}
</script>
"""


# ---------------- Overview blocks ----------------
counts = snaps_df['Signal'].value_counts().reindex(['BUY', 'DCA', 'WATCH', 'AVOID'], fill_value=0).to_dict()
kpi_html = f"""
<div class="kpis">
  <div class="kpi buy">BUY<div class="num">{counts.get('BUY',0)}</div></div>
  <div class="kpi dca">DCA<div class="num">{counts.get('DCA',0)}</div></div>
  <div class="kpi watch">WATCH<div class="num">{counts.get('WATCH',0)}</div></div>
  <div class="kpi avoid">AVOID<div class="num">{counts.get('AVOID',0)}</div></div>
  <div class="kpi break">Breakout Ready<div class="num">{BRKCOUNT}</div></div>
  <div class="kpi gate">Auto-DCA Gate<div class="num">{len(GATE)}</div></div>
  <div class="kpi flag">Flags<div class="num">{len(FLAGS)}</div></div>
  <div class="kpi pattern">Patterns<div class="num">{len(PATS)}</div></div>
  <div class="kpi news">News/Announcements<div class="num">{NEWSCOUNT}</div></div>
</div>"""

overview_html = f"""
<div class="grid">
  {panel_overview('BUY — Act now (ranked)',  BUY.head(12), 'buy')}
  {panel_overview('DCA — Controlled pullbacks', DCA.head(12), 'dca')}
  {panel_overview('WATCH — Close to triggers', WATCH.head(12), 'watch')}
  {panel_overview('AVOID — Downtrends', AVOID.head(12), 'avoid')}
</div>"""


# Auto-DCA Gate
gate_rows = []
for _, r in GATE.iterrows():
    gate_rows.append(f"""
<tr>
  <td><a class="ticker" href="https://au.finance.yahoo.com/quote/{r['Ticker']}.AX" target="_blank">{r['Ticker']}</a><div class="desc">{r['Name']}</div></td>
  <td>{r['AutoDCA_Gap_%']:.2f}%</td>
  <td>{'Yes' if r['AutoDCA_ReclaimMid'] else 'No'}</td>
  <td>{'Yes' if r['AutoDCA_AboveEMA21'] else 'No'}</td>
  <td>{r['AutoDCA_Fill_%']:.1f}%</td>
  <td>{r['_mini_candle']}</td>
</tr>""")

gate_html = f"""
<div class="card">
  <h3 style="margin:0 0 6px 0">Auto-DCA Gate — Reaction setups</h3>
  <div class="smallmuted">Gap ≤ {RULES['autodca']['gap_thresh']}%, close &gt; D-1 midpoint & EMA21, gap-fill ≥ {RULES['autodca']['fill_req']}%.</div>
  <table class="table" id="tbl_gate">
    <thead><tr><th>Ticker</th><th>Gap % (D-1)</th><th>Reclaim Mid?</th><th>&gt; EMA21?</th><th>Gap-fill %</th><th>Mini</th></tr></thead>
    <tbody>{''.join(gate_rows) if gate_rows else '<tr><td colspan="6" class="desc">None today</td></tr>'}</tbody>
  </table>
</div>
<script>makeSortable("tbl_gate");</script>
"""

# Tickers table
tick_rows = []
for _, r in snaps_df.sort_values('Ticker').iterrows():
    flag_badge = '<span class="badge">FLAG</span>' if r.get('Flag') else ''
    pat_badge  = f"<span class='patternbadge'>{r.get('_pattern_name','')}</span>" if r.get('_pattern_name','') else ''
    tick_rows.append(
        f"<tr><td><a class='ticker' target='_blank' href='https://au.finance.yahoo.com/quote/{r['Ticker']}.AX'>{r['Ticker']}</a>"
        f"{flag_badge}{pat_badge}</td><td>{r['Name']}</td><td>{r['Desc']}</td></tr>"
    )
tickers_html = make_table('tbl_tickers', ['Ticker', 'Name', 'Description'], ''.join(tick_rows))

# Signals (with commentary)
# Signals (with commentary) — mobile-friendly card layout
signal_cards = []
for _, r in snaps_df.sort_values(['Signal', 'Ticker']).iterrows():
    pat = r.get('_pattern_name', '')
    pat_txt = (
        f" • Pattern: {pat} ({r.get('_pattern_status','')}, "
        f"conf {r.get('_pattern_conf', np.nan)})"
        if pat else ''
    )

    sig_label = r.get('Signal', '') or 'N/A'
    sig_auto = ' (auto)' if r.get('SignalAuto') else ''

    # Decide badge colour bucket
    bucket = (sig_label or '').split()[0].lower()
    if bucket.startswith('buy'):
        badge_cls = 'buy'
    elif bucket.startswith('dca'):
        badge_cls = 'dca'
    elif bucket.startswith('watch'):
        badge_cls = 'watch'
    elif bucket.startswith('avoid') or bucket.startswith('sell'):
        badge_cls = 'avoid'
    else:
        badge_cls = 'flag'

    mini_html = r.get('_mini_candle', '') or ''

    signal_cards.append(f"""
<div class="signal-card">
  <div class="signal-card-top">
    <div>
      <div class="signal-label {badge_cls}">{sig_label}{sig_auto}</div>
      <a class="ticker" href="#overview">{r['Ticker']}</a>
      <span class="signal-name">{r['Name']}</span>
    </div>
    <div class="signal-price">
      <div class="code">{r['LastClose']:.4f}</div>
      <div class="smallmuted">{r['LastDate']}</div>
    </div>
  </div>

  <div class="signal-metrics">
    <div><span class="label">RSI14</span>
         <span class="value code">{r['RSI14']:.2f}</span></div>
    <div><span class="label">→52W%</span>
         <span class="value code">{r['Dist_to_52W_High_%']:.2f}%</span></div>
    <div><span class="label">→200DMA%</span>
         <span class="value code">{r['Dist_to_SMA200_%']:.2f}%</span></div>
  </div>

  <div class="signal-comment desc">
    {r['Comment']}{pat_txt}
  </div>

  <div class="signal-chart">
    {mini_html}
  </div>
</div>
""")

signals_html = f"""
<div class="card">
  <div class="smallmuted">
    <b>Glossary:</b> SMAs are 20/50/200-day simple moving averages.
    →52W% and →200DMA% are % distances from those anchors.
    Commentary explains WHY each label was assigned.
    ETF / News tags show exposure & recent news; 📰 appears if news is within the last
    {NEWS_WINDOW_DAYS} days.
  </div>
</div>
<div class="signals-list">
  {''.join(signal_cards)}
</div>
"""


# Patterns
pat_rows = []
for _, r in PATS.iterrows():
    pat_rows.append(f"""
<tr>
  <td>{r.get('_pattern_name','')}</td>
  <td><a class="ticker" href="#overview">{r['Ticker']}</a></td>
  <td>{r.get('_pattern_status','')}</td>
  <td>{r.get('_pattern_conf',np.nan)}</td>
  <td>{r.get('_pattern_align','')}</td>
  <td>{r['_mini_candle']}</td>
</tr>""")
patterns_html = f"""
<div class="card">
  <h3 style="margin:0 0 6px 0">Patterns — key structures (last {PATTERN_LOOKBACK} bars)</h3>
  <div class="smallmuted">Only confirmed patterns shown here (toggleable in code). Green = horizontal neckline/resistance, Amber = slanted trendlines.</div>
  <table class="table" id="tbl_patterns">
    <thead><tr><th>Pattern</th><th>Ticker</th><th>Status</th><th>Confidence</th><th>Alignment</th><th>Mini</th></tr></thead>
    <tbody>{''.join(pat_rows) if pat_rows else '<tr><td colspan="6" class="desc">No high-confidence patterns right now.</td></tr>'}</tbody>
  </table>
</div>
<script>makeSortable("tbl_patterns");</script>
"""

# News/Announcements table
news_rows = []
for _, n in news_df.sort_values('Date', ascending=False).iterrows():
    badge_class = 'newsneg' if n['Tag'] in ('reg', 'director') else ('newspos' if n['Tag'] in ('ops', 'issue') else '')
    b = f"<span class='badge {badge_class}'>{n['Type']}</span>"
    link = f"<a class='ticker' href='{n['Path']}' target='_blank'>open</a>" if n['Path'] else ''
    news_rows.append(
        f"<tr><td>{n['Date']}</td><td>{n['Ticker']}</td><td>{b}</td><td>{n['Headline']}</td><td>{n['Details']}</td><td>{link}</td></tr>"
    )
news_html = f"""
<div class="card">
  <div class="smallmuted">Drop ASX PDFs into <code>{ANN_DIR}</code>. We parse type (3Y/2A/Price Query/etc.), extract director sale stats when present, and flag items from the last {NEWS_WINDOW_DAYS} days in Signals/Overview comments.</div>
  {make_table('tbl_news', ['Date','Ticker','Type','Headline','Details','File'], ''.join(news_rows))}
</div>
"""

# Prices preview (no CSV export now)
if not prices_all.empty:
    prev_rows = []
    for t in sorted(prices_all['Ticker'].unique()):
        sub = prices_all[prices_all['Ticker'] == t].tail(1).iloc[0]
        prev_rows.append(
            f"<tr><td>{t}</td><td>{pd.to_datetime(sub['Date']).strftime('%Y-%m-%d')}</td>"
            f"<td>{float(sub['Open']):.4f}</td><td>{float(sub['High']):.4f}</td>"
            f"<td>{float(sub['Low']):.4f}</td><td>{float(sub['Close']):.4f}</td>"
            f"<td>{int(sub['Volume'])}</td></tr>"
        )
    prices_preview = make_table(
        'tbl_prices',
        ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
        ''.join(prev_rows)
    )
else:
    prices_preview = '<div class="card"><div class="smallmuted">No price data.</div></div>'

prices_html = prices_preview

# MiniData (sparklines)
spark_rows = []
for _, r in snaps_df.sort_values('Ticker').iterrows():
    spark_rows.append(
        f"<tr><td><a class='ticker' target='_blank' href='https://au.finance.yahoo.com/quote/{r['Ticker']}.AX'>{r['Ticker']}</a></td><td>{r['Name']}</td><td>{r['_mini_spark']}</td></tr>"
    )
minidata_html = make_table('tbl_minidata', ['Ticker', 'Name', '90-day Sparkline'], ''.join(spark_rows))

rules_html = f"<div class='card'><pre style='margin:0;white-space:pre-wrap'>{json.dumps(RULES | {'BREAKOUT_RULES': BREAKOUT_RULES, 'AUTO_UPGRADE_BREAKOUT': AUTO_UPGRADE_BREAKOUT}, indent=2)}</pre></div>"

# ---------------- Final HTML ----------------
html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
<title>TraderBruh — ASX TA (Web)</title>
<link rel="preconnect" href="https://cdn.plot.ly"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style><script>{SORT_JS}</script></head>
<body>
  <div class="nav"><div class="navinner">
    <a href="#overview">Overview</a><a href="#gate">Auto-DCA Gate</a><a href="#tickers">Tickers</a>
    <a href="#signals">Signals</a><a href="#patterns">Patterns</a><a href="#news">News/Announcements</a>
    <a href="#prices">Prices</a><a href="#minidata">MiniData</a><a href="#rules">Rules</a>
  </div></div>

  <div class="container">
    <div class="section" id="overview">
      <div class="card" style="margin-bottom:12px">
        <h1>TraderBruh — ASX TA</h1>
        <small>Updated {datetime.now(SYD).strftime('%d %b %Y, %I:%M %p %Z')} • Mini-charts: Candles (last {MINI_BARS} sessions) • Flags, Auto-DCA, Patterns & News active • DT Breakout auto-BUY.</small>
      </div>
      {kpi_html}
      {overview_html}
    </div>

    <div class="section" id="gate">{gate_html}</div>
    <div class="section" id="tickers"><h2>Tickers</h2>{tickers_html}</div>
    <div class="section" id="signals"><h2>Signals</h2>{signals_html}</div>
    <div class="section" id="patterns"><h2>Patterns</h2>{patterns_html}</div>
    <div class="section" id="news"><h2>News/Announcements</h2>{news_html}</div>
    <div class="section" id="prices"><h2>Prices</h2>{prices_html}</div>
    <div class="section" id="minidata"><h2>MiniData</h2>{minidata_html}</div>
    <div class="section" id="rules"><h2>Rules</h2>{rules_html}</div>

    <div class="card" style="margin-top:16px"><div class="smallmuted">© {datetime.now().year} TraderBruh — patterns & news parsing are heuristic; verify with primary filings.</div></div>
  </div>
</body></html>
"""

# Ensure output directory exists and write HTML
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print('Done:', OUTPUT_HTML)
