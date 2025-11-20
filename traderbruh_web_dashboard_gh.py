# traderbruh_global_ultimate.py
# TraderBruh — Global Web Dashboard (ASX / USA / INDIA)
# Version: Ultimate 3.2 (Bugfix: Pattern Rendering)

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
ANN_DIR             = 'announcements'  # For ASX PDFs
FETCH_DAYS          = 900
MINI_BARS           = 120
SPARK_DAYS          = 90
NEWS_WINDOW_DAYS    = 14
PATTERN_LOOKBACK    = 180
PIVOT_WINDOW        = 4
PRICE_TOL           = 0.03
PATTERNS_CONFIRMED_ONLY = True

# --- Market Definitions ---
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
        }
    }
}

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

# ---------------- Data Fetching ----------------
def fetch_prices(symbol: str, tz_name: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=f'{FETCH_DAYS}d', interval='1d', auto_adjust=False, progress=False, prepost=False)
        if df is None or df.empty: return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(symbol, axis=1, level=-1, drop_level=True)
            except: df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        df['Date'] = pd.to_datetime(df[date_col], utc=True)

        # Intraday Stitch
        market_tz = zoneinfo.ZoneInfo(tz_name)
        now_mkt = datetime.now(market_tz)
        last_date_mkt = df['Date'].dt.tz_convert(market_tz).dt.date.max()
        
        if last_date_mkt < now_mkt.date():
            try:
                intr = yf.download(symbol, period='5d', interval='60m', auto_adjust=False, progress=False)
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
    """Replicated Buffett/Piotroski Logic"""
    try:
        tick = yf.Ticker(symbol)
        info = tick.info
        try:
            bs = tick.balance_sheet
            is_ = tick.income_stmt
            cf = tick.cashflow
        except: bs, is_, cf = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        def get_item(df, names, idx=0):
            if df.empty: return 0
            for n in names:
                if n in df.index:
                    try: return float(df.loc[n].iloc[idx])
                    except: return 0
            return 0

        def get_cagr(df, names, years=3):
            if df.empty or df.shape[1] < years: return 0
            curr = get_item(df, names, 0)
            past = get_item(df, names, years-1)
            if past <= 0 or curr <= 0: return 0
            return (curr / past)**(1/(years-1)) - 1

        # 1. Profitability
        roe_3y = 0
        if not is_.empty and not bs.empty:
            try:
                roes = []
                for i in range(min(3, len(is_.columns))):
                    ni = get_item(is_, ['Net Income'], i)
                    eq = get_item(bs, ['Stockholders Equity', 'Total Equity Gross Minority Interest'], i)
                    if eq > 0: roes.append(ni/eq)
                if roes: roe_3y = sum(roes)/len(roes)
            except: pass
        
        ocf = get_item(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
        net_inc = get_item(is_, ['Net Income'])
        marg_curr = info.get('profitMargins', 0)
        
        score = 0
        if roe_3y > 0.15: score += 2
        elif roe_3y > 0.10: score += 1
        if marg_curr > 0.10: score += 1
        if ocf > net_inc: score += 0.5

        # 2. Balance Sheet
        curr_ratio = info.get('currentRatio', 0)
        debt_eq = info.get('debtToEquity', 999)
        if debt_eq > 50: debt_eq /= 100.0
        
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
            change = (shares_curr - shares_old)/shares_old
            if change < -0.01: score += 1.5; is_buyback = True
            elif change < 0.05: score += 1

        # 4. Growth
        rev_cagr = get_cagr(is_, ['Total Revenue', 'Operating Revenue'], 3)
        if rev_cagr > 0.10: score += 1
        
        peg = info.get('pegRatio', 0)
        pe = info.get('trailingPE', 0)
        if (peg and 0 < peg < 2.0) or (pe and 0 < pe < 20): score += 1
        
        score = min(score, 10)
        tier = 'Fortress' if score >= 7 else ('Quality' if score >= 4 else ('Spec' if score >= 2 else 'Junk'))
        
        return {
            'score': round(score, 1), 'tier': tier, 'roe_3y': roe_3y, 'margins': marg_curr,
            'debt_eq': debt_eq, 'rev_cagr': rev_cagr, 'is_buyback': is_buyback,
            'pe': pe, 'cash': cash
        }
    except:
        return {'score': 0, 'tier': 'Error', 'roe_3y':0, 'margins':0, 'debt_eq':0, 'rev_cagr':0, 'pe':0, 'cash': 0}

# ---------------- TA Indicators & Patterns ----------------
def indicators(df):
    x = df.copy().sort_values('Date').reset_index(drop=True)
    x['SMA20'] = x['Close'].rolling(20).mean()
    x['SMA50'] = x['Close'].rolling(50).mean()
    x['SMA200'] = x['Close'].rolling(200).mean()
    x['EMA21'] = x['Close'].ewm(span=21, adjust=False).mean()
    x['High20'] = x['High'].rolling(20).max()
    x['High52W'] = x['High'].rolling(252).max()
    x['Vol20'] = x['Volume'].rolling(20).mean()
    
    chg = x['Close'].diff()
    gains = chg.clip(lower=0).rolling(14).mean()
    losses = (-chg).clip(lower=0).rolling(14).mean()
    rs = gains / losses
    x['RSI14'] = 100 - (100 / (1 + rs))
    
    x['Dist_to_52W_High_%'] = (x['Close']/x['High52W'] - 1) * 100
    x['Dist_to_SMA200_%'] = (x['Close']/x['SMA200'] - 1) * 100
    
    x['TR'] = x[['High', 'Low', 'Close']].apply(lambda r: max(r['High']-r['Low'], abs(r['High']-r['Close']), abs(r['Low']-r['Close'])), axis=1)
    x['ATR14'] = x['TR'].rolling(14).mean()
    return x

def label_row(r):
    buy = (r['Close'] > r['SMA200']) and (r['Close'] > r['High20']) and (r['SMA50'] > r['SMA200']) and (RULES['buy']['rsi_min'] <= r['RSI14'] <= RULES['buy']['rsi_max'])
    dca = (r['Close'] >= r['SMA200']) and (r['RSI14'] < RULES['dca']['rsi_max']) and (r['Close'] <= r['SMA200']*(1+RULES['dca']['sma200_proximity']))
    avoid = (r['SMA50'] < r['SMA200']) if RULES['avoid']['death_cross'] else False
    if buy: return 'BUY'
    if dca: return 'DCA'
    if avoid: return 'AVOID'
    return 'WATCH'

def auto_dca_gate(ind):
    if len(ind) < 3: return False, {}
    D0, D1, D2 = ind.iloc[-1], ind.iloc[-2], ind.iloc[-3]
    gap_pct = (D1['Open'] / D2['Close'] - 1) * 100.0
    if gap_pct > RULES['autodca']['gap_thresh']: return False, {'gap_pct': gap_pct}
    gap_mid = (D1['High'] + D1['Low']) / 2.0
    reclaim = D0['Close'] > gap_mid
    above_ema = D0['Close'] > D0['EMA21']
    gap_sz = max(D2['Close'] - D1['Open'], 0.0)
    fill_pct = 0.0 if gap_sz == 0 else (D0['Close'] - D1['Open']) / gap_sz * 100.0
    flag = reclaim and above_ema and (fill_pct >= RULES['autodca']['fill_req'])
    return flag, {'gap_pct': gap_pct, 'reclaim_mid': reclaim, 'above_ema21': above_ema, 'gap_fill_%': fill_pct}

def _pivots(ind, win=PIVOT_WINDOW):
    v = ind.tail(PATTERN_LOOKBACK).reset_index(drop=True).copy()
    v['PH'] = (v['High'] == v['High'].rolling(win*2+1, center=True).max()).fillna(False)
    v['PL'] = (v['Low'] == v['Low'].rolling(win*2+1, center=True).min()).fillna(False)
    return v

def _similar(a, b): return (abs(a-b)/((a+b)/2)) <= PRICE_TOL

def detect_patterns(ind):
    v = _pivots(ind)
    out = []
    # Double Bottom
    lows = v.index[v['PL']].tolist()
    for i in range(len(lows)):
        for j in range(i+1, len(lows)):
            if lows[j]-lows[i]<10: continue
            p1, p2 = float(v.loc[lows[i],'Low']), float(v.loc[lows[j],'Low'])
            if _similar(p1,p2):
                neck = float(v.loc[lows[i]:lows[j],'High'].max())
                conf = v['Close'].iloc[-1] > neck
                lines = [('h', v.loc[lows[i],'Date'], v.loc[lows[j],'Date'], (p1+p2)/2), ('h', v.loc[lows[i],'Date'], v['Date'].iloc[-1], neck)]
                out.append({'name':'Double Bottom','status':'confirmed' if conf else 'forming','confidence':0.8 if conf else 0.6,'lines':lines})
                break
    # Double Top
    highs = v.index[v['PH']].tolist()
    for i in range(len(highs)):
        for j in range(i+1, len(highs)):
            if highs[j]-highs[i]<10: continue
            p1, p2 = float(v.loc[highs[i],'High']), float(v.loc[highs[j],'High'])
            if _similar(p1,p2):
                neck = float(v.loc[highs[i]:highs[j],'Low'].min())
                conf = v['Close'].iloc[-1] < neck
                lines = [('h', v.loc[highs[i],'Date'], v.loc[highs[j],'Date'], (p1+p2)/2), ('h', v.loc[highs[i],'Date'], v['Date'].iloc[-1], neck)]
                out.append({'name':'Double Top','status':'confirmed' if conf else 'forming','confidence':0.8 if conf else 0.6,'levels':{'ceiling':(p1+p2)/2},'lines':lines})
                break
    # Triangles (Simplified)
    tail = v.tail(120)
    phs, pls = tail[tail['PH']], tail[tail['PL']]
    if len(phs)>=2 and len(pls)>=2:
        # Asc
        if _similar(phs['High'].iloc[0], phs['High'].iloc[-1]):
            slope = np.polyfit(np.arange(len(pls)), pls['Low'].values, 1)[0]
            if slope > 0:
                res = phs['High'].mean()
                conf = tail['Close'].iloc[-1] > res
                lines = [('h', pls['Date'].iloc[0], tail['Date'].iloc[-1], res), ('seg', pls['Date'].iloc[0], pls['Low'].iloc[0], pls['Date'].iloc[-1], pls['Low'].iloc[-1])]
                out.append({'name':'Ascending Triangle','status':'confirmed' if conf else 'forming','confidence':0.8 if conf else 0.55,'lines':lines})
    return out

def detect_flag(ind):
    if len(ind) < 60: return False, {}
    look = ind.tail(40)
    impulse = (look['Close'].max()/look['Close'].min()-1)*100
    if impulse < 12: return False, {}
    tail = ind.tail(14).copy()
    x = np.arange(len(tail))
    hi, lo = np.polyfit(x, tail['High'].values, 1), np.polyfit(x, tail['Low'].values, 1)
    slope = (hi[0]/tail['Close'].iloc[-1])*100
    tight = (np.polyval(hi, x[-1]) - np.polyval(lo, x[-1])) <= max(0.4*(look['Close'].max()-look['Close'].min()), 0.02*tail['Close'].iloc[-1])
    return (tight and -0.6 <= slope <= 0.2), {'hi': hi.tolist(), 'lo': lo.tolist(), 'win': 14}

# ---------------- Visualization ----------------
def mini_candle(ind, flag_info=None, pattern_data=None):
    v = ind.tail(MINI_BARS).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=v['Date'], open=v['Open'], high=v['High'], low=v['Low'], close=v['Close'], increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
    fig.add_trace(go.Scatter(x=v['Date'], y=v['SMA20'], mode='lines', line=dict(width=1, color='rgba(56,189,248,0.8)')))
    
    if flag_info:
        t2 = ind.tail(flag_info['win'])
        x = np.arange(len(t2))
        hi, lo = np.poly1d(flag_info['hi']), np.poly1d(flag_info['lo'])
        fig.add_trace(go.Scatter(x=t2['Date'], y=hi(x), mode='lines', line=dict(width=2, dash='dash', color='#a855f7')))
        fig.add_trace(go.Scatter(x=t2['Date'], y=lo(x), mode='lines', line=dict(width=2, dash='dash', color='#a855f7')))
    
    # BUGFIX: Check if pattern_data is a list of dicts (Full Objects) or list of tuples (Lines)
    all_lines = []
    if pattern_data:
        if isinstance(pattern_data, list) and len(pattern_data) > 0 and isinstance(pattern_data[0], dict):
            for p in pattern_data:
                all_lines.extend(p.get('lines', []))
        elif isinstance(pattern_data, list):
            all_lines = pattern_data

    for ln in all_lines:
        if ln[0] == 'h':
            fig.add_trace(go.Scatter(x=[ln[1], ln[2]], y=[ln[3], ln[3]], mode='lines', line=dict(width=2, color='#facc15', dash='dot')))
        elif ln[0] == 'seg':
            fig.add_trace(go.Scatter(x=[ln[1], ln[3]], y=[ln[2], ln[4]], mode='lines', line=dict(width=2, color='#facc15')))

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=130, width=280, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'staticPlot': True})

def mini_spark(ind):
    v = ind.tail(SPARK_DAYS)
    fig = go.Figure(go.Scatter(x=v['Date'], y=v['Close'], mode='lines', line=dict(width=1, color='#94a3b8')))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=50, width=120, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'staticPlot': True})

# ---------------- News (Regional) ----------------
def parse_news(market_code):
    # Only AUS supported for PDF parsing currently
    rows = []
    if market_code == 'AUS' and os.path.isdir(ANN_DIR) and HAVE_PYPDF:
        for fp in sorted(glob.glob(os.path.join(ANN_DIR, '*.pdf'))):
            try:
                text = re.sub(r'[ \t]+', ' ', PdfReader(fp).pages[0].extract_text() or '')
                fname = os.path.basename(fp)
                m = re.match(r'([A-Z]{3})[_-]', fname)
                tick = m.group(1) if m else '???'
                typ = 'Appendix 3Y' if 'Change of Director' in text else ('Report' if 'Quarterly' in text else 'Announcement')
                d = datetime.fromtimestamp(os.path.getmtime(fp)).date().isoformat()
                rows.append({'Date': d, 'Ticker': tick, 'Type': typ, 'Headline': typ, 'Path': fp})
            except: pass
    return pd.DataFrame(rows, columns=['Date', 'Ticker', 'Type', 'Headline', 'Path'])

# ---------------- Logic & Loop ----------------
def analyze_market(m_code, m_conf):
    print(f"Processing {m_conf['name']}...")
    snaps = []
    
    # News
    news_df = parse_news(m_code)
    
    for t_key, t_meta in m_conf['tickers'].items():
        full_sym = f"{t_key}{m_conf['suffix']}"
        df = fetch_prices(full_sym, m_conf['tz'])
        if df.empty: continue
        
        fundy = fetch_deep_fundamentals(full_sym)
        ind = indicators(df).dropna(subset=['SMA200'])
        if ind.empty: continue
        
        last = ind.iloc[-1]
        sig = label_row(last)
        
        pats = detect_patterns(ind)
        if PATTERNS_CONFIRMED_ONLY: pats = [p for p in pats if p.get('status')=='confirmed']
        
        flag_flag, flag_det = detect_flag(ind)
        gate_flag, gate_det = auto_dca_gate(ind)
        
        brk_ready, brk_lvl = False, 0
        if (dts := [p for p in pats if p['name']=='Double Top']):
            ceil = dts[0].get('levels', {}).get('ceiling', 0)
            if last['Close'] >= ceil*(1+BREAKOUT_RULES['buffer_pct']):
                brk_ready, brk_lvl = True, ceil
        
        if AUTO_UPGRADE_BREAKOUT and brk_ready: sig = 'BUY'
        
        # Comment Generation
        d52 = last['Dist_to_52W_High_%']
        d200 = last['Dist_to_SMA200_%']
        score = fundy['score']
        
        com = "Neutral."
        if sig == 'BUY':
            com = f"<b>CORE BUY:</b> High Conviction ({score}/10)." if score >= 7 else f"<b>SPEC BUY:</b> Weak fundamentals ({score}/10). Tight stops."
        elif sig == 'DCA':
            com = f"<b>QUALITY DIP:</b> Fortress ({score}/10) on sale." if score >= 7 else f"<b>AVOID:</b> Value Trap ({score}/10)."
        elif sig == 'WATCH':
            com = f"<b>EUPHORIA:</b> Extended ({d52:.1f}%). Trim." if (d52 > -3.5 and d200 > 50) else "Watchlist."
        
        if brk_ready: com += f" • BREAKOUT > {brk_lvl:.2f}"
        
        snaps.append({
            'Ticker': t_key, 'Name': t_meta[0], 'Desc': t_meta[1],
            'LastDate': last['Date'].strftime('%Y-%m-%d'), 'LastClose': last['Close'],
            'RSI14': last['RSI14'], 'Dist_200': d200, 'Dist_52': d52,
            'Signal': sig, 'Comment': com,
            'Fundy': fundy,
            'Flag': flag_flag, 'Gate': gate_flag, 'Gate_Info': gate_det,
            '_ind': ind, '_pats': pats, '_flag_info': flag_det,
            'Breakout': brk_ready
        })

    return pd.DataFrame(snaps), news_df

# ---------------- HTML Generation ----------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --bg: #0f172a; --surface: #1e293b; --primary: #3b82f6; --text: #f1f5f9; --muted: #94a3b8; --border: rgba(148,163,184,0.1); }
body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; margin: 0; padding-bottom: 80px; }
.mono { font-family: 'JetBrains Mono', monospace; }
.hidden { display: none !important; }

/* Top Bar */
.market-bar { position: fixed; top: 0; left: 0; right: 0; height: 50px; background: #020617; border-bottom: 1px solid var(--border); display: flex; justify-content: center; align-items: center; gap: 10px; z-index: 200; }
.market-btn { background: transparent; border: 1px solid var(--muted); color: var(--muted); padding: 6px 16px; border-radius: 99px; cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.2s; }
.market-btn.active { background: var(--primary); border-color: var(--primary); color: white; }

/* Sub Nav */
.nav-wrapper { position: sticky; top: 50px; z-index: 100; background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(10px); border-bottom: 1px solid var(--border); padding: 10px 0; }
.nav-inner { display: flex; gap: 8px; overflow-x: auto; max-width: 1200px; margin: 0 auto; padding: 0 16px; scrollbar-width: none; }
.nav-link { color: var(--muted); text-decoration: none; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 500; white-space: nowrap; }
.nav-link:hover, .nav-link.active { background: rgba(255,255,255,0.1); color: white; }

.container { max-width: 1200px; margin: 60px auto 0; padding: 20px 16px; }
.market-container { display: none; }
.market-container.active { display: block; animation: fade 0.3s; }
@keyframes fade { from { opacity:0; } to { opacity:1; } }

/* Cards */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; margin-bottom: 40px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 16px; }
.badge { padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase; display: inline-block; margin-left: 6px; }
.badge.buy { background: rgba(16,185,129,0.15); color: #34d399; }
.badge.dca { background: rgba(245,158,11,0.15); color: #fbbf24; }
.badge.watch { background: rgba(59,130,246,0.15); color: #60a5fa; }
.badge.avoid { background: rgba(239,68,68,0.15); color: #f87171; }
.shield { margin-left: 6px; font-size: 12px; }

/* Tables */
.table-res { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 12px; background: rgba(0,0,0,0.2); color: var(--muted); }
td { padding: 12px; border-bottom: 1px solid var(--border); }
.ticker-badge { background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 4px; font-weight: 700; }

/* KPI */
.kpi-row { display: flex; gap: 12px; overflow-x: auto; margin-bottom: 24px; padding-bottom: 5px; }
.kpi-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px; min-width: 130px; }
.kpi-val { font-size: 24px; font-weight: 700; }
.kpi-lbl { font-size: 11px; color: var(--muted); text-transform: uppercase; }
.text-green { color: #34d399; } .text-amber { color: #fbbf24; } .text-blue { color: #60a5fa; }

.metrics-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; background: rgba(0,0,0,0.2); padding: 8px; margin-bottom: 10px; border-radius: 8px; }
.metric label { font-size: 10px; color: var(--muted); display: block; }
.metric span { font-size: 13px; font-weight: 600; }
"""

JS = """
function switchMarket(code) {
    document.querySelectorAll('.market-container').forEach(el => el.classList.remove('active'));
    document.getElementById('cont-'+code).classList.add('active');
    document.querySelectorAll('.market-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('btn-'+code).classList.add('active');
}
function filter(id) {
    let q = document.getElementById(id).value.toLowerCase();
    document.querySelectorAll('.active .search-item').forEach(r => {
        r.style.display = r.innerText.toLowerCase().includes(q) ? '' : 'none';
    });
}
"""

def render_market_html(m_code, m_conf, df, news_df):
    if df.empty: return f"<div id='cont-{m_code}' class='market-container'><div style='padding:40px;text-align:center'>No data for {m_code}</div></div>"
    
    curr = m_conf['currency']
    # Segments
    buy = df[df['Signal']=='BUY'].sort_values('Fundy', key=lambda x: x.apply(lambda y: y['score']), ascending=False)
    dca = df[df['Signal']=='DCA'].sort_values('Fundy', key=lambda x: x.apply(lambda y: y['score']), ascending=False)
    watch = df[df['Signal']=='WATCH']
    avoid = df[df['Signal']=='AVOID']
    
    # KPIs
    kpi_html = f"""
    <div class="kpi-row">
        <div class="kpi-card"><div class="kpi-lbl">Buy Signal</div><div class="kpi-val text-green">{len(buy)}</div></div>
        <div class="kpi-card"><div class="kpi-lbl">DCA Zone</div><div class="kpi-val text-amber">{len(dca)}</div></div>
        <div class="kpi-card"><div class="kpi-lbl">Watchlist</div><div class="kpi-val text-blue">{len(watch)}</div></div>
        <div class="kpi-card"><div class="kpi-lbl">Breakouts</div><div class="kpi-val">{df['Breakout'].sum()}</div></div>
    </div>
    """

    # Helper for Cards
    def make_cards(sub_df, badge_cls, section_id):
        if sub_df.empty: return ""
        html = f"<h2 id='{m_code}-{section_id}' style='color:var(--muted); font-size:18px; margin-top:30px'>{section_id.upper()} ({len(sub_df)})</h2><div class='grid'>"
        for _, r in sub_df.iterrows():
            f = r['Fundy']
            shield = "🛡️" if f['score']>=7 else ("⚠️" if f['score']<=3 else "")
            chart = mini_candle(r['_ind'], r['_flag_info'] if r['Flag'] else None, r['_pats'])
            
            html += f"""
            <div class="card search-item">
                <div style="display:flex; justify-content:space-between; margin-bottom:10px">
                    <div>
                        <span class="ticker-badge mono">{r['Ticker']}</span>
                        <span class="badge {badge_cls}">{r['Signal']}</span>
                        <span class="shield">{shield} {f['score']}/10 {f['tier']}</span>
                        <div style="font-size:12px; color:var(--muted); margin-top:4px">{r['Name']}</div>
                    </div>
                    <div style="text-align:right">
                        <div class="mono" style="font-size:18px; font-weight:600">{curr}{r['LastClose']:.2f}</div>
                        <div style="font-size:11px; color:var(--muted)">{r['LastDate']}</div>
                    </div>
                </div>
                <div class="metrics-row">
                    <div class="metric"><label>RSI</label><span class="mono">{r['RSI14']:.0f}</span></div>
                    <div class="metric"><label>vs 200DMA</label><span class="mono">{r['Dist_200']:+.1f}%</span></div>
                    <div class="metric"><label>vs 52W Hi</label><span class="mono">{r['Dist_52']:+.1f}%</span></div>
                </div>
                <div style="font-size:13px; color:#cbd5e1; margin-bottom:10px; min-height:40px">{r['Comment']}</div>
                <div>{chart}</div>
            </div>
            """
        html += "</div>"
        return html

    cards_html = ""
    cards_html += make_cards(buy, 'buy', 'buy')
    cards_html += make_cards(dca, 'dca', 'dca')
    cards_html += make_cards(watch, 'watch', 'watch')
    cards_html += make_cards(avoid, 'avoid', 'avoid')

    # Fundy Table
    f_rows = ""
    for _, r in df.sort_values('Fundy', key=lambda x: x.apply(lambda y: y['score']), ascending=False).iterrows():
        f = r['Fundy']
        f_rows += f"<tr class='search-item'><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td>{f['score']}/10 {f['tier']}</td><td class='mono'>{f['roe_3y']*100:.1f}%</td><td class='mono'>{f['margins']*100:.1f}%</td><td class='mono'>{f['rev_cagr']*100:.1f}%</td><td class='mono'>{f['debt_eq']:.2f}</td><td class='mono'>{f['pe']:.1f}</td></tr>"

    # Gate Table
    gate_rows = ""
    gates = df[df['Gate']==True]
    if not gates.empty:
        for _, r in gates.iterrows():
            g = r['Gate_Info']
            gate_rows += f"<tr class='search-item'><td><b>{r['Ticker']}</b></td><td class='mono'>{g['gap_pct']:.1f}%</td><td class='mono'>Yes</td><td class='mono'>{g['gap_fill_%']:.1f}%</td><td>{mini_spark(r['_ind'])}</td></tr>"
    else: gate_rows = "<tr><td colspan='5' style='text-align:center; color:var(--muted)'>No setups today</td></tr>"

    # Pattern Table
    p_rows = ""
    pat_df = []
    for _, r in df.iterrows():
        for p in r['_pats']:
            pat_df.append({'Ticker':r['Ticker'], 'Pattern':p['name'], 'Status':p['status'], 'Conf':p['confidence'], 'Ind':r['_ind']})
    if pat_df:
        for p in sorted(pat_df, key=lambda x: x['Conf'], reverse=True):
            p_rows += f"<tr class='search-item'><td>{p['Pattern']}</td><td><b>{p['Ticker']}</b></td><td>{p['Status']}</td><td>{p['Conf']:.2f}</td><td>{mini_candle(p['Ind'])}</td></tr>"
    else: p_rows = "<tr><td colspan='5' style='text-align:center; color:var(--muted)'>No patterns found</td></tr>"

    # News Table
    n_rows = ""
    if not news_df.empty:
        for _, r in news_df.sort_values('Date', ascending=False).iterrows():
            n_rows += f"<tr class='search-item'><td class='mono' style='color:var(--muted)'>{r['Date']}</td><td><b>{r['Ticker']}</b></td><td>{r['Type']}</td><td>{r['Headline']}</td></tr>"
    else: n_rows = "<tr><td colspan='4' style='text-align:center; color:var(--muted)'>No news data available</td></tr>"

    # Navigation HTML (Sub-nav)
    nav_html = f"""
    <div class="nav-wrapper">
        <div class="nav-inner">
            <a href="#{m_code}-buy" class="nav-link">Buy</a>
            <a href="#{m_code}-dca" class="nav-link">DCA</a>
            <a href="#{m_code}-watch" class="nav-link">Watch</a>
            <a href="#{m_code}-fundy" class="nav-link">Fundamentals</a>
            <a href="#{m_code}-gate" class="nav-link">Auto-Gate</a>
            <a href="#{m_code}-patterns" class="nav-link">Patterns</a>
            <a href="#{m_code}-news" class="nav-link">News</a>
            <input type="text" id="search-{m_code}" onkeyup="filter('search-{m_code}')" placeholder="Filter..." style="background:rgba(0,0,0,0.3); border:1px solid var(--border); color:white; padding:4px 10px; border-radius:8px; font-size:13px; margin-left:auto">
        </div>
    </div>
    """

    return f"""
    <div id="cont-{m_code}" class="market-container {'active' if m_code=='AUS' else ''}">
        {nav_html}
        <div class="container" style="margin-top:10px">
            <div style="margin-bottom:20px">
                <h1 style="font-size:24px; margin:0">{m_conf['name']}</h1>
                <div style="color:var(--muted); font-size:13px">Updated: {datetime.now(zoneinfo.ZoneInfo(m_conf['tz'])).strftime('%I:%M %p')} local time</div>
            </div>
            
            {kpi_html}
            {cards_html}
            
            <h2 id="{m_code}-fundy" style="margin-top:40px; border-top:1px solid var(--border); padding-top:20px">Fundamental Shield</h2>
            <div class="table-res"><table><thead><tr><th>Ticker</th><th>Score</th><th>ROE</th><th>Margin</th><th>Rev Growth</th><th>Debt/Eq</th><th>P/E</th></tr></thead><tbody>{f_rows}</tbody></table></div>

            <h2 id="{m_code}-gate" style="margin-top:40px; border-top:1px solid var(--border); padding-top:20px">Auto-DCA Gate</h2>
            <div class="table-res"><table><thead><tr><th>Ticker</th><th>Gap %</th><th>Reclaim?</th><th>Fill %</th><th>Spark</th></tr></thead><tbody>{gate_rows}</tbody></table></div>

            <h2 id="{m_code}-patterns" style="margin-top:40px; border-top:1px solid var(--border); padding-top:20px">Technical Patterns</h2>
            <div class="table-res"><table><thead><tr><th>Pattern</th><th>Ticker</th><th>Status</th><th>Conf</th><th>Chart</th></tr></thead><tbody>{p_rows}</tbody></table></div>

            <h2 id="{m_code}-news" style="margin-top:40px; border-top:1px solid var(--border); padding-top:20px">News & Announcements</h2>
            <div class="table-res"><table><thead><tr><th>Date</th><th>Ticker</th><th>Type</th><th>Headline</th></tr></thead><tbody>{n_rows}</tbody></table></div>
            
            <div style="height:100px"></div>
        </div>
    </div>
    """

if __name__ == "__main__":
    print("Starting Ultimate Global Analysis...")
    
    # Build Top Nav
    top_nav = ""
    for c, conf in MARKETS.items():
        act = "active" if c == 'AUS' else ""
        top_nav += f"<button id='btn-{c}' class='market-btn {act}' onclick=\"switchMarket('{c}')\">{conf['name']}</button>"
    
    # Analyze & Render
    body_html = ""
    for c, conf in MARKETS.items():
        df, news = analyze_market(c, conf)
        body_html += render_market_html(c, conf, df, news)
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TraderBruh Global Ultimate</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>{CSS}</style>
        <script>{JS}</script>
    </head>
    <body>
        <div class="market-bar">{top_nav}</div>
        {body_html}
    </body>
    </html>
    """
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"Done. Dashboard: {OUTPUT_HTML}")