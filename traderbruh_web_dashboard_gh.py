# traderbruh_global_gh.py
# TraderBruh — Global Web Dashboard (ASX / USA / INDIA)
# Version: Global 3.0 (Multi-Market Tabs + Deep Fundamentals)

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
ANN_DIR             = 'announcements'  # Primarily for ASX PDFs
FETCH_DAYS          = 900
MINI_BARS           = 120
SPARK_DAYS          = 90

# Market Configurations
# Add your tickers here. 
# Suffix: .AX for Australia, .NS for India (NSE), empty for US.
MARKETS = {
    'AUS': {
        'name': 'Australia (ASX)',
        'tz': 'Australia/Sydney',
        'currency': 'A$',
        'suffix': '.AX',
        'tickers': {
            'CSL': 'CSL Limited', 'COH': 'Cochlear', 'XRO': 'Xero', 
            'WTC': 'WiseTech', 'PLS': 'Pilbara Minerals', 'MIN': 'MinRes',
            'RMD': 'ResMed', 'DRO': 'DroneShield', 'NEA': 'Nearmap',
            'PNV': 'PolyNovo', 'WEB': 'WebTravel', 'HUB': 'HUB24'
        }
    },
    'USA': {
        'name': 'United States (NYSE/NAS)',
        'tz': 'America/New_York',
        'currency': 'U$',
        'suffix': '',
        'tickers': {
            'NVDA': 'NVIDIA', 'AAPL': 'Apple', 'MSFT': 'Microsoft',
            'TSLA': 'Tesla', 'AMD': 'AMD', 'PLTR': 'Palantir',
            'COIN': 'Coinbase', 'MSTR': 'MicroStrategy', 'AMZN': 'Amazon',
            'GOOG': 'Google', 'META': 'Meta', 'NFLX': 'Netflix'
        }
    },
    'IND': {
        'name': 'India (NSE)',
        'tz': 'Asia/Kolkata',
        'currency': '₹',
        'suffix': '.NS',
        'tickers': {
            'RELIANCE': 'Reliance Ind', 'TCS': 'TCS', 'INFY': 'Infosys',
            'HDFCBANK': 'HDFC Bank', 'TATAMOTORS': 'Tata Motors',
            'ZOMATO': 'Zomato', 'PAYTM': 'Paytm', 'ITC': 'ITC Ltd',
            'ADANIENT': 'Adani Ent', 'WIPRO': 'Wipro'
        }
    }
}

# Technical & Fundamental Rules (Global)
RULES = {
    'buy':     {'rsi_min': 45, 'rsi_max': 70},
    'dca':     {'rsi_max': 45, 'sma200_proximity': 0.05},
    'avoid':   {'death_cross': True},
    'autodca': {'gap_thresh': -2.0, 'fill_req': 50.0}
}

BREAKOUT_RULES = {'atr_mult': 0.50, 'vol_mult': 1.30, 'buffer_pct': 0.003}
AUTO_UPGRADE_BREAKOUT = True
PATTERNS_CONFIRMED_ONLY = True

# ---------------- Data Fetching ----------------
def fetch_prices(symbol: str, tz_name: str) -> pd.DataFrame:
    """Fetch OHLCV history adjusted for specific market timezone."""
    try:
        df = yf.download(symbol, period=f'{FETCH_DAYS}d', interval='1d', auto_adjust=False, progress=False, prepost=False)
        if df is None or df.empty: return pd.DataFrame()
        
        # Handle MultiIndex if YF returns it
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(symbol, axis=1, level=-1, drop_level=True)
            except: df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        df['Date'] = pd.to_datetime(df[date_col], utc=True)
        
        market_tz = zoneinfo.ZoneInfo(tz_name)
        
        # Intraday Stitching Logic
        now_market = datetime.now(market_tz)
        last_date_market = df['Date'].dt.tz_convert(market_tz).dt.date.max()
        
        # If market is open/active and we don't have today's bar yet
        if last_date_market < now_market.date():
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
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def fetch_deep_fundamentals(symbol: str):
    """Calculates Buffett/Piotroski Quality Score (0-10)"""
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
        marg_curr = info.get('profitMargins', 0)
        
        score = 0
        if roe_3y > 0.15: score += 2
        elif roe_3y > 0.10: score += 1
        if marg_curr > 0.10: score += 1
        if ocf > net_inc: score += 0.5

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
        tier = 'Fortress' if score >= 7 else ('Quality' if score >= 4 else ('Spec' if score >= 2 else 'Junk'))
        
        return {
            'score': round(score, 1), 'tier': tier, 'roe_3y': roe_3y, 'margins': marg_curr,
            'debt_eq': debt_eq, 'rev_cagr': rev_cagr, 'pe': pe, 'cash': cash
        }
    except:
        return {'score': 0, 'tier': 'Error', 'roe_3y':0, 'margins':0, 'debt_eq':0, 'rev_cagr':0, 'pe':0, 'cash': 0}

# ---------------- TA Logic (Shared) ----------------
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
    x['ATR14'] = x[['High', 'Low', 'Close']].apply(lambda r: max(r['High']-r['Low'], abs(r['High']-r['Close']), abs(r['Low']-r['Close'])), axis=1).rolling(14).mean()
    return x

def label_row(r: pd.Series) -> str:
    buy_ok = (r['Close'] > r['SMA200']) and (r['Close'] > r['High20']) and (r['SMA50'] > r['SMA200']) and (RULES['buy']['rsi_min'] <= r['RSI14'] <= RULES['buy']['rsi_max'])
    dca_ok = (r['Close'] >= r['SMA200']) and (r['RSI14'] < RULES['dca']['rsi_max']) and (r['Close'] <= r['SMA200'] * (1 + RULES['dca']['sma200_proximity']))
    avoid = (r['SMA50'] < r['SMA200']) if RULES['avoid']['death_cross'] else False
    if buy_ok: return 'BUY'
    if dca_ok: return 'DCA'
    if avoid: return 'AVOID'
    return 'WATCH'

# --- Pattern Recognition (Condensed for space, logic identical to original) ---
def _pivots(ind, window=4):
    v = ind.tail(180).reset_index(drop=True).copy()
    v['PH'] = (v['High'] == v['High'].rolling(window*2+1, center=True).max()).fillna(False)
    v['PL'] = (v['Low'] == v['Low'].rolling(window*2+1, center=True).min()).fillna(False)
    return v

def _similar(a, b, tol=0.03): return (abs(a-b)/((a+b)/2.0)) <= tol

def detect_patterns(ind):
    v = _pivots(ind)
    pats = []
    # Double Bottom
    lows = v.index[v['PL']].tolist()
    for i in range(len(lows)):
        for j in range(i+1, len(lows)):
            if lows[j]-lows[i] < 10: continue
            p1, p2 = float(v.loc[lows[i], 'Low']), float(v.loc[lows[j], 'Low'])
            if _similar(p1, p2):
                neck = float(v.loc[lows[i]:lows[j], 'High'].max())
                conf = v['Close'].iloc[-1] > neck
                pats.append({'name': 'Double Bottom', 'status': 'confirmed' if conf else 'forming', 'confidence': 0.8 if conf else 0.6, 'lines': [('h', v.loc[lows[i], 'Date'], v['Date'].iloc[-1], neck)]})
                break # One per type
    # Double Top
    highs = v.index[v['PH']].tolist()
    for i in range(len(highs)):
        for j in range(i+1, len(highs)):
            if highs[j]-highs[i] < 10: continue
            p1, p2 = float(v.loc[highs[i], 'High']), float(v.loc[highs[j], 'High'])
            if _similar(p1, p2):
                neck = float(v.loc[highs[i]:highs[j], 'Low'].min())
                conf = v['Close'].iloc[-1] < neck
                pats.append({'name': 'Double Top', 'status': 'confirmed' if conf else 'forming', 'confidence': 0.8 if conf else 0.6, 'levels': {'ceiling': (p1+p2)/2}, 'lines': [('h', v.loc[highs[i], 'Date'], v['Date'].iloc[-1], neck)]})
                break
    return pats

def detect_flag(ind):
    if len(ind) < 60: return False, {}
    look = ind.tail(40)
    impulse = (look['Close'].max() / look['Close'].min() - 1) * 100
    if impulse < 12: return False, {}
    tail = ind.tail(14).copy()
    x = np.arange(len(tail))
    hi, lo = np.polyfit(x, tail['High'].values, 1), np.polyfit(x, tail['Low'].values, 1)
    slope = (hi[0] / tail['Close'].iloc[-1]) * 100
    tight = (np.polyval(hi, x[-1]) - np.polyval(lo, x[-1])) <= max(0.4*(look['Close'].max()-look['Close'].min()), 0.02*tail['Close'].iloc[-1])
    return (tight and -0.6 <= slope <= 0.2), {'hi': hi.tolist(), 'lo': lo.tolist(), 'win': 14}

def auto_dca_gate(ind):
    if len(ind) < 3: return False, {}
    D0, D1, D2 = ind.iloc[-1], ind.iloc[-2], ind.iloc[-3]
    gap_pct = (D1['Open']/D2['Close']-1)*100
    if gap_pct > RULES['autodca']['gap_thresh']: return False, {}
    mid = (D1['High']+D1['Low'])/2
    gap_sz = max(D2['Close']-D1['Open'], 0)
    fill = 0 if gap_sz==0 else (D0['Close']-D1['Open'])/gap_sz*100
    flag = (D0['Close'] > mid) and (D0['Close'] > D0['EMA21']) and (fill >= 50)
    return flag, {'gap_pct': gap_pct, 'fill': fill}

# ---------------- Market Analysis Engine ----------------
def analyze_market(market_code, config):
    print(f"--> Analyzing {config['name']}...")
    frames, snaps = [], []
    
    # Parse ASX News if applicable, otherwise generic empty
    news_rows = []
    if market_code == 'AUS' and os.path.isdir(ANN_DIR):
        # (Keep original ASX specific PDF parsing here if needed, simplified for brevity)
        pass 
    news_df = pd.DataFrame(news_rows, columns=['Date', 'Ticker', 'Type', 'Headline']) # Placeholder

    for ticker_key, ticker_name in config['tickers'].items():
        full_sym = f"{ticker_key}{config['suffix']}"
        df = fetch_prices(full_sym, config['tz'])
        if df.empty: continue
        df['Ticker'] = ticker_key # Use clean name for display

        # Deep Fundamentals
        fundy = fetch_deep_fundamentals(full_sym)

        # Indicators
        ind = indicators(df).dropna(subset=['SMA200', 'SMA50', 'ATR14'])
        if ind.empty: continue
        last = ind.iloc[-1]
        sig = label_row(last)

        # Patterns
        pats = detect_patterns(ind)
        if PATTERNS_CONFIRMED_ONLY: pats = [p for p in pats if p.get('status') == 'confirmed']
        
        # Breakout check
        brk_ready, brk_info = False, {}
        if (dt := [p for p in pats if p['name']=='Double Top']):
            cl, atr, vol = float(last['Close']), float(last['ATR14']), float(last['Volume'])
            ceil = dt[0].get('levels', {}).get('ceiling', 0)
            if ceil > 0 and cl >= ceil*(1+BREAKOUT_RULES['buffer_pct']) and vol > 1.3*last['Vol20']:
                brk_ready, brk_info = True, {'level': ceil}
        
        if AUTO_UPGRADE_BREAKOUT and brk_ready: sig = 'BUY'

        # Auto DCA Gate
        gate_flag, gate_det = auto_dca_gate(ind)
        flag_flag, flag_det = detect_flag(ind)

        # Commentary Logic
        d52, d200, rsi = last['Dist_to_52W_High_%'], last['Dist_to_SMA200_%'], last['RSI14']
        f_sc = fundy['score']
        
        com = "Neutral."
        if sig == 'BUY':
            com = f"<b>CORE BUY:</b> Strong Trend + Fundy {f_sc}/10." if f_sc >= 7 else f"<b>SPEC BUY:</b> Trend UP but Fundy Weak ({f_sc}/10)."
        elif sig == 'DCA':
            com = f"<b>QUALITY DIP:</b> Great Co ({f_sc}/10) near 200DMA." if f_sc >= 7 else f"<b>AVOID:</b> Falling Knife (Junk Fundy)."
        elif sig == 'WATCH':
            com = "Euphoria Warning." if (d52 > -3.5 and d200 > 50) else "Watchlist."
        elif sig == 'AVOID':
            com = "Sidelines."

        snaps.append({
            'Ticker': ticker_key, 'Name': ticker_name, 'Market': market_code,
            'LastDate': last['Date'].strftime('%Y-%m-%d'), 'LastClose': float(last['Close']),
            'RSI14': float(last['RSI14']), 'Dist_to_SMA200_%': float(last['Dist_to_SMA200_%']),
            'Dist_to_52W_High_%': float(last['Dist_to_52W_High_%']), 'Signal': sig,
            'Comment': com, 'Flag': flag_flag, 'Breakout': brk_ready,
            'Gate': gate_flag, 'Gate_Info': gate_det,
            'Fundy': fundy, '_ind': ind, '_pats': pats, '_flag_info': flag_det
        })

    return pd.DataFrame(snaps)

# ---------------- Rendering (HTML/CSS/JS) ----------------

def generate_sparkline(ind):
    if ind is None or ind.empty: return ""
    v = ind.tail(SPARK_DAYS)
    fig = go.Figure(go.Scatter(x=v['Date'], y=v['Close'], mode='lines', line=dict(width=1, color='#94a3b8')))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=40, width=100, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'staticPlot': True})

def generate_candle(ind, flag_info, pats):
    if ind is None: return ""
    v = ind.tail(MINI_BARS)
    fig = go.Figure(data=[go.Candlestick(x=v['Date'], open=v['Open'], high=v['High'], low=v['Low'], close=v['Close'], increasing_line_color='#4ade80', decreasing_line_color='#f87171')])
    fig.add_trace(go.Scatter(x=v['Date'], y=v['SMA20'], line=dict(color='rgba(56,189,248,0.8)', width=1)))
    
    if flag_info:
        t2 = ind.tail(flag_info['win'])
        x = np.arange(len(t2))
        fig.add_trace(go.Scatter(x=t2['Date'], y=np.polyval(flag_info['hi'], x), line=dict(dash='dash', color='#a855f7')))
        fig.add_trace(go.Scatter(x=t2['Date'], y=np.polyval(flag_info['lo'], x), line=dict(dash='dash', color='#a855f7')))
    
    if pats:
        for p in pats:
            for ln in p.get('lines', []):
                if ln[0] == 'h':
                    fig.add_trace(go.Scatter(x=[ln[1], ln[2]], y=[ln[3], ln[3]], mode='lines', line=dict(color='#facc15', width=2, dash='dot')))

    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=120, width=260, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'staticPlot': True})

def render_dashboard(market_data):
    # market_data is dict: {'AUS': df, 'USA': df, 'IND': df}
    
    CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    :root { --bg: #0f172a; --surface: #1e293b; --primary: #3b82f6; --text: #f1f5f9; --muted: #94a3b8; --border: rgba(148,163,184,0.1); }
    body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; margin: 0; padding-bottom: 50px; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    
    /* Tabs */
    .market-nav { display: flex; justify-content: center; gap: 10px; padding: 20px 0; background: rgba(15,23,42,0.95); position: sticky; top: 0; z-index: 50; border-bottom: 1px solid var(--border); }
    .market-btn { background: transparent; border: 1px solid var(--muted); color: var(--muted); padding: 8px 24px; border-radius: 99px; cursor: pointer; font-weight: 600; transition: all 0.2s; }
    .market-btn:hover { border-color: var(--text); color: var(--text); }
    .market-btn.active { background: var(--primary); border-color: var(--primary); color: white; box-shadow: 0 0 15px rgba(59,130,246,0.4); }
    
    .container { max-width: 1300px; margin: 0 auto; padding: 0 15px; }
    .market-view { display: none; animation: fadeIn 0.3s ease; }
    .market-view.active { display: block; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 15px; margin-top: 15px; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 15px; position: relative; overflow: hidden; }
    .card-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
    .badge { padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
    .b-buy { background: rgba(16,185,129,0.2); color: #34d399; }
    .b-dca { background: rgba(245,158,11,0.2); color: #fbbf24; }
    .b-watch { background: rgba(59,130,246,0.2); color: #60a5fa; }
    .b-avoid { background: rgba(239,68,68,0.2); color: #f87171; }
    
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px; margin-bottom: 10px; }
    .m-lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; }
    .m-val { font-size: 13px; font-weight: 600; }
    
    .table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); margin-top: 20px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 10px 15px; text-align: left; border-bottom: 1px solid var(--border); }
    th { background: rgba(0,0,0,0.2); color: var(--muted); }
    
    .sub-header { font-size: 18px; font-weight: 600; color: var(--muted); margin-top: 30px; border-left: 4px solid var(--primary); padding-left: 10px; }
    .shield { font-size: 14px; margin-left: 5px; }
    """

    JS = """
    function switchMarket(code) {
        document.querySelectorAll('.market-view').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.market-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('view-'+code).classList.add('active');
        document.getElementById('btn-'+code).classList.add('active');
    }
    """

    html_parts = []
    for code, df in market_data.items():
        conf = MARKETS[code]
        curr = conf['currency']
        
        # Sections
        buy = df[df['Signal']=='BUY'].sort_values('Fundy', key=lambda x: x.apply(lambda y: y['score']), ascending=False)
        dca = df[df['Signal']=='DCA']
        watch = df[df['Signal']=='WATCH']
        avoid = df[df['Signal']=='AVOID']
        
        card_html = ""
        for sec_name, sub_df, b_cls in [("Actionable Buys", buy, "b-buy"), ("DCA / Accumulate", dca, "b-dca"), ("Watchlist", watch, "b-watch"), ("Avoid", avoid, "b-avoid")]:
            if sub_df.empty: continue
            card_html += f"<div class='sub-header'>{sec_name}</div><div class='grid'>"
            for _, r in sub_df.iterrows():
                f = r['Fundy']
                shield = "🛡️" if f['score']>=7 else ("⚠️" if f['score']<=3 else "")
                plot = generate_candle(r['_ind'], r['_flag_info'] if r['Flag'] else None, r['_pats'])
                
                card_html += f"""
                <div class="card">
                    <div class="card-head">
                        <div>
                            <span class="mono" style="font-weight:700; font-size:16px">{r['Ticker']}</span>
                            <span class="badge {b_cls}">{r['Signal']}</span>
                            <div style="font-size:12px; color:var(--muted)">{r['Name']}</div>
                        </div>
                        <div style="text-align:right">
                            <div class="mono" style="font-size:18px">{curr}{r['LastClose']:.2f}</div>
                            <div style="font-size:11px; color:var(--muted)">{f['tier']} {shield}</div>
                        </div>
                    </div>
                    <div class="metrics">
                        <div><div class="m-lbl">RSI</div><div class="m-val mono">{r['RSI14']:.0f}</div></div>
                        <div><div class="m-lbl">vs 200DMA</div><div class="m-val mono">{r['Dist_to_SMA200_%']:+.1f}%</div></div>
                        <div><div class="m-lbl">Fundy Score</div><div class="m-val mono">{f['score']}/10</div></div>
                    </div>
                    <div style="font-size:12px; margin-bottom:10px; color:#cbd5e1">{r['Comment']}</div>
                    {plot}
                </div>"""
            card_html += "</div>"
            
        # Fundy Table
        fundy_rows = ""
        for _, r in df.sort_values('Fundy', key=lambda x: x.apply(lambda y: y['score']), ascending=False).iterrows():
            f = r['Fundy']
            fundy_rows += f"<tr><td class='mono'><b>{r['Ticker']}</b></td><td>{f['score']}/10 {f['tier']}</td><td class='mono'>{f['roe_3y']*100:.1f}%</td><td class='mono'>{f['margins']*100:.1f}%</td><td class='mono'>{f['rev_cagr']*100:.1f}%</td><td class='mono'>{f['debt_eq']:.2f}</td></tr>"
            
        view_cls = "active" if code == 'AUS' else "" # Default to AUS
        
        html_parts.append(f"""
        <div id="view-{code}" class="market-view {view_cls}">
            <div style="text-align:center; margin:20px 0; color:var(--muted)">
                <h1 style="color:white; margin:0">{conf['name']}</h1>
                <div>Data updated: {datetime.now(zoneinfo.ZoneInfo(conf['tz'])).strftime('%I:%M %p')} local time</div>
            </div>
            {card_html}
            
            <div class="sub-header" style="margin-top:40px">Deep Fundamentals</div>
            <div class="table-wrap">
                <table>
                    <thead><tr><th>Ticker</th><th>Score</th><th>ROE (3y)</th><th>Margins</th><th>Rev Growth</th><th>Debt/Eq</th></tr></thead>
                    <tbody>{fundy_rows}</tbody>
                </table>
            </div>
        </div>
        """)

    # Navigation Buttons
    nav_btns = ""
    for code in MARKETS.keys():
        act = "active" if code == 'AUS' else ""
        nav_btns += f"<button id='btn-{code}' class='market-btn {act}' onclick=\"switchMarket('{code}')\">{MARKETS[code]['name']}</button>"

    full_html = f"""<!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>TraderBruh Global</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>{CSS}</style><script>{JS}</script>
    </head>
    <body>
        <div class="market-nav">{nav_btns}</div>
        <div class="container">
            {"".join(html_parts)}
        </div>
    </body>
    </html>
    """
    
    return full_html

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    print("Starting Global Analysis...")
    
    # Process all markets
    all_market_data = {}
    for m_code, m_conf in MARKETS.items():
        df = analyze_market(m_code, m_conf)
        all_market_data[m_code] = df
        print(f"Finished {m_code}: {len(df)} tickers processed.")

    # Generate HTML
    print("Generating Dashboard...")
    html_content = render_dashboard(all_market_data)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Done! Dashboard saved to: {OUTPUT_HTML}")