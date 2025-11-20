# traderbruh_web_dashboard_gh.py
# TraderBruh — Global Edition (ASX, US, INDIA)
# Version: Ultimate 2.2 (Stable - Fixed Fundamental Keys)

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
# Timezone definitions
SYD = zoneinfo.ZoneInfo('Australia/Sydney')

OUTPUT_DIR = "docs"
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "index.html")
ANN_DIR = 'announcements'

# Tuning
FETCH_DAYS = 900
MINI_BARS = 120
PATTERN_LOOKBACK = 180
PIVOT_WINDOW = 4
PRICE_TOL = 0.03
PATTERNS_CONFIRMED_ONLY = True
NEWS_WINDOW_DAYS = 14

RULES = {
    'buy': {'rsi_min': 45, 'rsi_max': 70},
    'dca': {'rsi_max': 45, 'sma200_proximity': 0.05},
    'avoid': {'death_cross': True},
    'autodca': {'gap_thresh': -2.0, 'fill_req': 50.0}
}

BREAKOUT_RULES = {'atr_mult': 0.50, 'vol_mult': 1.30, 'buffer_pct': 0.003}
AUTO_UPGRADE_BREAKOUT = True

FUNDY_THRESHOLDS = {
    'roe_high': 0.15, 'roe_mid': 0.10,
    'margin_high': 0.15, 'margin_mid': 0.08,
    'de_safe': 0.50, 'curr_ratio': 1.5,
    'peg_fair': 2.0, 'fcf_yield': 0.03
}

# ---------------- The Global Universe ----------------
MARKETS = {
    'AU': {
        'name': '🇦🇺 ASX', 'tz': 'Australia/Sydney', 'suffix': '.AX',
        'stocks': {
            'CSL': ('CSL Limited', 'Biotech.'), 'BHP': ('BHP Group', 'Mining.'),
            'CBA': ('CommBank', 'Banking.'), 'WTC': ('WiseTech', 'Logistics Tech.'),
            'XRO': ('Xero', 'Accounting SaaS.'), 'PME': ('Pro Medicus', 'Health Imaging.'),
            'PLS': ('Pilbara', 'Lithium.'), 'FMG': ('Fortescue', 'Iron Ore.'),
            'TLS': ('Telstra', 'Telecoms.'), 'WOW': ('Woolworths', 'Retail.'),
            'STO': ('Santos', 'Energy.'), 'NXT': ('NextDC', 'Data Centres.')
        }
    },
    'US': {
        'name': '🇺🇸 Wall St', 'tz': 'America/New_York', 'suffix': '',
        'stocks': {
            'NVDA': ('NVIDIA', 'AI Hardware.'), 'MSFT': ('Microsoft', 'Cloud/AI.'),
            'AAPL': ('Apple', 'Consumer Tech.'), 'AMZN': ('Amazon', 'E-comm/AWS.'),
            'GOOGL': ('Alphabet', 'Search.'), 'META': ('Meta', 'Social.'),
            'TSLA': ('Tesla', 'EV/Robotics.'), 'PLTR': ('Palantir', 'Data Ops.'),
            'COIN': ('Coinbase', 'Crypto.'), 'BRK-B': ('Berkshire', 'Conglomerate.')
        }
    },
    'IN': {
        'name': '🇮🇳 India', 'tz': 'Asia/Kolkata', 'suffix': '.NS',
        'stocks': {
            'RELIANCE': ('Reliance', 'Conglomerate.'), 'TCS': ('TCS', 'IT Services.'),
            'HDFCBANK': ('HDFC Bank', 'Banking.'), 'INFY': ('Infosys', 'IT Services.'),
            'TATAMOTORS': ('Tata Motors', 'Auto.'), 'ICICIBANK': ('ICICI', 'Banking.'),
            'ITC': ('ITC', 'FMCG.'), 'ZOMATO': ('Zomato', 'Food Tech.')
        }
    }
}

# ---------------- Data Fetching ----------------
def fetch_prices(symbol: str, timezone: str) -> pd.DataFrame:
    df = yf.download(symbol, period=f'{FETCH_DAYS}d', interval='1d', auto_adjust=False, progress=False, group_by='column', prepost=False)
    if df is None or df.empty: return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.xs(symbol, axis=1, level=-1, drop_level=True)
        except: df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df['Date'] = pd.to_datetime(df[date_col], utc=True)

    # Intraday Stitch
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
    try:
        tick = yf.Ticker(symbol)
        info = tick.info
        try: bs = tick.balance_sheet; is_ = tick.income_stmt; cf = tick.cashflow
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

        curr_ratio = info.get('currentRatio', 0)
        debt_eq = info.get('debtToEquity', 999)
        if debt_eq > 50: debt_eq = debt_eq / 100.0
        
        cash = get_item(bs, ['Cash And Cash Equivalents', 'Cash Financial'])
        lt_debt = get_item(bs, ['Long Term Debt'])
        
        if cash > lt_debt: score += 1.5
        elif debt_eq < 0.5: score += 1
        if curr_ratio > 1.5: score += 1
        elif curr_ratio > 1.1: score += 0.5

        shares_curr = get_item(bs, ['Share Issued', 'Ordinary Shares Number'], 0)
        shares_old = get_item(bs, ['Share Issued', 'Ordinary Shares Number'], 2)
        is_buyback = False
        if shares_old > 0:
            change = (shares_curr - shares_old) / shares_old
            if change < -0.01: score += 1.5; is_buyback = True
            elif change < 0.05: score += 1
        
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
            'pe': pe, 'cash': cash, 'peg': peg  # <--- FIXED: Added peg to return
        }
    except:
        # FIXED: Added peg to exception return
        return {'score': 0, 'tier': 'Error', 'roe_3y':0, 'margins':0, 'debt_eq':0, 'rev_cagr':0, 'is_buyback':False, 'fcf_yield':0, 'pe':0, 'cash': 0, 'peg': 0}

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

def read_pdf_first_text(path: str):
    if not HAVE_PYPDF: return ''
    try: return re.sub(r'[ \t]+', ' ', PdfReader(path).pages[0].extract_text() or '')
    except: return ''

def parse_announcements():
    rows = []
    if not os.path.isdir(ANN_DIR): return pd.DataFrame(columns=['Date', 'Ticker', 'Type', 'Tag', 'Headline', 'Path'])
    for fp in sorted(glob.glob(os.path.join(ANN_DIR, '*.pdf'))):
        fname = os.path.basename(fp)
        m = re.match(r'([A-Z]{2,6})[_-]', fname)
        ticker = m.group(1) if m else 'UNKNOWN'
        text = read_pdf_first_text(fp)
        
        _type, tag = 'Announcement', 'gen'
        for label, pat, t in NEWS_TYPES:
            if re.search(pat, fname + " " + text, re.I):
                _type, tag = label, t
                break
                
        d = datetime.fromtimestamp(os.path.getmtime(fp), tz=SYD).replace(tzinfo=None)
        if (md := re.search(r'(\d{1,2}\s+[A-Za-z]{3,9}\s+20\d{2})', text)):
            try: d = datetime.strptime(md.group(1), '%d %B %Y')
            except: pass
            
        rows.append({'Date': d.date().isoformat(), 'Ticker': ticker, 'Type': _type, 'Tag': tag, 'Headline': _type, 'Path': fp})
    return pd.DataFrame(rows)

# ---------------- Indicators & Logic ----------------
def indicators(df):
    x = df.copy().sort_values('Date').reset_index(drop=True)
    x['SMA20'] = x['Close'].rolling(20).mean()
    x['SMA50'] = x['Close'].rolling(50).mean()
    x['SMA200'] = x['Close'].rolling(200).mean()
    x['EMA21'] = x['Close'].ewm(span=21, adjust=False).mean()
    x['High20'] = x['High'].rolling(20).max()
    x['High52W'] = x['High'].rolling(252).max()
    
    chg = x['Close'].diff()
    gains = chg.clip(lower=0).rolling(14).mean()
    losses = (-chg).clip(lower=0).rolling(14).mean()
    RS = gains / losses
    x['RSI14'] = 100 - (100 / (1 + RS))
    x['Dist_to_52W_High_%'] = (x['Close'] / x['High52W'] - 1) * 100.0
    x['Dist_to_SMA200_%'] = (x['Close'] / x['SMA200'] - 1) * 100.0
    
    x['H-L'] = x['High'] - x['Low']
    x['H-C'] = (x['High'] - x['Close'].shift(1)).abs()
    x['L-C'] = (x['Low'] - x['Close'].shift(1)).abs()
    x['TR'] = x[['H-L', 'H-C', 'L-C']].max(axis=1)
    x['ATR14'] = x['TR'].rolling(14).mean()
    return x

def label_row(r):
    buy_ok = (r['Close'] > r['SMA200'] and r['Close'] > r['High20'] and r['SMA50'] > r['SMA200'] and RULES['buy']['rsi_min'] <= r['RSI14'] <= RULES['buy']['rsi_max'])
    dca_ok = (r['Close'] >= r['SMA200'] and r['RSI14'] < RULES['dca']['rsi_max'] and r['Close'] <= r['SMA200'] * (1 + RULES['dca']['sma200_proximity']))
    avoid = (r['SMA50'] < r['SMA200']) if RULES['avoid']['death_cross'] else False
    if buy_ok: return 'BUY'
    if dca_ok: return 'DCA'
    if avoid: return 'AVOID'
    return 'WATCH'

# ---------------- Pattern Engine ----------------
def _pivots(ind, window=PIVOT_WINDOW):
    v = ind.tail(PATTERN_LOOKBACK).reset_index(drop=True).copy()
    ph = (v['High'] == v['High'].rolling(window * 2 + 1, center=True).max())
    pl = (v['Low']  == v['Low'].rolling(window * 2 + 1, center=True).min())
    v['PH'] = ph.fillna(False); v['PL'] = pl.fillna(False)
    return v

def _similar(a, b, tol=PRICE_TOL): return (abs(a - b) / ((a+b)/2)) <= tol

def detect_patterns(ind):
    v = _pivots(ind)
    pats = []
    last_cls = v['Close'].iloc[-1]
    
    # 1. Double Bottom
    lows = v.index[v['PL']].tolist()
    for i in range(len(lows)):
        for j in range(i+1, len(lows)):
            li, lj = lows[i], lows[j]
            if lj - li < 10: continue
            p1, p2 = float(v.loc[li, 'Low']), float(v.loc[lj, 'Low'])
            if _similar(p1, p2):
                neck = float(v.loc[li:lj, 'High'].max())
                if last_cls > neck:
                    pats.append({'name': 'Double Bottom', 'status': 'confirmed', 'conf': 0.8, 'lines': [('h', v.loc[li,'Date'], v.loc[lj,'Date'], (p1+p2)/2), ('h', v.loc[li,'Date'], v['Date'].iloc[-1], neck)]})
                    return pats

    # 2. Double Top
    highs = v.index[v['PH']].tolist()
    for i in range(len(highs)):
        for j in range(i+1, len(highs)):
            hi, hj = highs[i], highs[j]
            if hj - hi < 10: continue
            p1, p2 = float(v.loc[hi, 'High']), float(v.loc[hj, 'High'])
            if _similar(p1, p2):
                neck = float(v.loc[hi:hj, 'Low'].min())
                if last_cls < neck:
                    pats.append({'name': 'Double Top', 'status': 'confirmed', 'conf': 0.8, 'lines': [('h', v.loc[hi,'Date'], v.loc[hj,'Date'], (p1+p2)/2), ('h', v.loc[hi,'Date'], v['Date'].iloc[-1], neck)]})
                    return pats
                    
    # 3. Ascending Triangle
    tail = v.tail(120).copy()
    phs, pls = tail[tail['PH']], tail[tail['PL']]
    if len(phs) >= 2 and len(pls) >= 2:
        ph_v = phs['High'].values
        for i in range(len(ph_v)-1):
            if _similar(ph_v[i], ph_v[i+1]):
                res = (ph_v[i]+ph_v[i+1])/2
                if tail['Close'].iloc[-1] > res:
                     pats.append({'name': 'Ascending Triangle', 'status': 'confirmed', 'conf': 0.75, 'lines': [('h', pls['Date'].iloc[0], tail['Date'].iloc[-1], res)]})
                     return pats
    return pats

def detect_flag(ind):
    if len(ind) < 60: return False, {}
    tail = ind.tail(15).copy()
    x = np.arange(len(tail))
    hi, lo = np.polyfit(x, tail['High'].values, 1), np.polyfit(x, tail['Low'].values, 1)
    slope_pct = (hi[0] / tail['Close'].iloc[-1]) * 100
    return (-0.6 <= slope_pct <= 0.2), {'hi': hi.tolist(), 'lo': lo.tolist()}

def auto_dca_gate(ind):
    if len(ind) < 3: return False, {}
    D0, D1, D2 = ind.iloc[-1], ind.iloc[-2], ind.iloc[-3]
    gap_pct = (D1['Open'] / D2['Close'] - 1) * 100.0
    if gap_pct > RULES['autodca']['gap_thresh']: return False, {}
    gap_mid = (D1['High'] + D1['Low']) / 2.0
    reclaim = D0['Close'] > gap_mid and D0['Close'] > D0['EMA21']
    return reclaim, {'gap_pct': gap_pct, 'reclaim': reclaim}

def pattern_bias(name):
    if name in ["Double Bottom", "Ascending Triangle"]: return "bullish"
    if name in ["Double Top"]: return "bearish"
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

def mini_candle(ind, flag_info=None, pat_lines=None):
    v = ind.tail(MINI_BARS).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=v['Date'], open=v['Open'], high=v['High'], low=v['Low'], close=v['Close'], hoverinfo='skip', showlegend=False, increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
    if 'SMA20' in v.columns:
        fig.add_trace(go.Scatter(x=v['Date'], y=v['SMA20'], mode='lines', line=dict(width=1.4, color='rgba(56,189,248,0.8)'), hoverinfo='skip', showlegend=False))
    if flag_info:
        t2 = ind.tail(max(flag_info.get('win', 14), 8)).copy()
        x = np.arange(len(t2))
        hi, lo = np.poly1d(flag_info['hi']), np.poly1d(flag_info['lo'])
        for line_data in [hi(x), lo(x)]:
            fig.add_trace(go.Scatter(x=t2['Date'], y=line_data, mode='lines', line=dict(width=2, dash='dash', color='rgba(167,139,250,0.95)'), hoverinfo='skip', showlegend=False))
    if pat_lines:
        for ln in pat_lines:
            if ln[0] == 'h': _, d1, d2, y = ln; fig.add_trace(go.Scatter(x=[d1, d2], y=[y, y], mode='lines', line=dict(width=2, color='rgba(234,179,8,0.9)', dash='dot'), hoverinfo='skip', showlegend=False))
    
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=120, width=260, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'staticPlot': True})

def comment_for_row(r):
    d200, d52, sig, f_score = r['Dist_to_SMA200_%'], r['Dist_to_52W_High_%'], r['Signal'], r['Fundy_Score']
    is_euphoria = (d52 > -3.5 and d200 > 50.0 and r['RSI14'] >= 70.0)
    
    if sig == 'BUY':
        if f_score >= 7: return f"<b>CORE BUY:</b> Strong trend + Fortress ({f_score}/10). Size up."
        elif f_score <= 3: return f"<b>SPECULATIVE:</b> Trend OK, Fundys Junk ({f_score}/10). Tight stops."
        return "Standard Buy: Trend + Fundys OK."
    elif sig == 'DCA':
        if f_score <= 3: return f"<b>AVOID (Falling Knife):</b> Dip-buy signal but Junk fundys. Danger."
        elif f_score >= 7: return f"<b>QUALITY DIP:</b> Fortress on sale."
        return "DCA Zone: Good risk/reward."
    elif sig == 'WATCH':
        if is_euphoria: return f"<b>EUPHORIA:</b> Extended. {('Exit (Junk)' if f_score<=3 else 'Trail Stops')}."
        return f"Watchlist: Near inflection. Quality: {f_score}/10."
    return "Avoid/Neutral."

# ---------------- Main Execution ----------------
all_market_html = ""
print("Parsing local news...")
news_df_all = parse_announcements()

for mkt_code, config in MARKETS.items():
    print(f"--- Processing {config['name']} ---")
    rows = []
    
    for ticker, meta in config['stocks'].items():
        symbol = ticker + config['suffix']
        print(f"Fetching {symbol}...")
        
        df = fetch_prices(symbol, config['tz'])
        if df.empty: continue
        fundy = fetch_deep_fundamentals(symbol)
        ind = indicators(df).dropna(subset=['SMA200'])
        if ind.empty: continue
        
        last = ind.iloc[-1]
        sig = label_row(last)
        flag_flag, flag_det = detect_flag(ind)
        pats = detect_patterns(ind)
        gate_flag, gate_det = auto_dca_gate(ind)
        
        pname = pats[0]['name'] if pats else ''
        pat_lines = pats[0]['lines'] if pats else None
        
        r = {
            'Ticker': ticker, 'Name': meta[0], 'Desc': meta[1],
            'LastClose': float(last['Close']), 'LastDate': last['Date'].strftime('%Y-%m-%d'),
            'Signal': sig, 'RSI14': float(last['RSI14']),
            'Dist_to_SMA200_%': float(last['Dist_to_SMA200_%']), 'Dist_to_52W_High_%': float(last['Dist_to_52W_High_%']),
            'Fundy_Score': fundy['score'], 'Fundy_Tier': fundy['tier'],
            'Fundy_ROE': fundy['roe_3y'], 'Fundy_Margin': fundy['margins'], 'Fundy_PE': fundy['pe'], 'Fundy_PEG': fundy['peg'],
            'Fundy_RevCAGR': fundy['rev_cagr'], 'Fundy_Debt': fundy['debt_eq'], 'Fundy_Cash': fundy['cash'],
            'Flag': flag_flag, 'Gate': gate_flag, 'Gate_Gap': gate_det.get('gap_pct', 0),
            'Pattern': pname
        }
        r['Comment'] = comment_for_row(r)
        
        # Attach News (If ASX)
        news_hit = news_df_all[news_df_all['Ticker'] == ticker]
        if not news_hit.empty:
             r['Comment'] += f" • 📰 Recent News: {len(news_hit)} items."

        r['_mini_candle'] = mini_candle(ind, flag_det if flag_flag else None, pat_lines)
        rows.append(r)
    
    df_mkt = pd.DataFrame(rows)
    if df_mkt.empty: continue
    
    # HTML Components
    def render_card(r, badge_type):
        score = r['Fundy_Score']
        s_badge = "shield-high" if score >= 7 else ("shield-low" if score <= 3 else "buy")
        fundy_html = f'<span class="badge {s_badge}" style="margin-left:6px">{score}/10 {r["Fundy_Tier"]}</span>'
        tags = ""
        if r['Flag']: tags += '<span class="badge news">🚩 FLAG</span>'
        if r['Gate']: tags += '<span class="badge dca">GATE</span>'
        if r['Pattern']: tags += f'<span class="badge watch">{r["Pattern"]}</span>'

        return f"""
        <div class="card searchable-item">
            <div class="card-header">
                <div><a class="ticker-badge mono">{r['Ticker']}</a><span class="badge {badge_type}" style="margin-left:8px">{r['Signal']}</span> {fundy_html} {tags}
                <div style="font-size:12px; color:var(--text-muted); margin-top:4px">{r['Name']}</div></div>
                <div class="price-block"><div class="price-main mono">{r['LastClose']:.2f}</div></div>
            </div>
            <div class="metrics-row">
                <div class="metric"><label>RSI</label><span class="mono" style="color:{'#ef4444' if r['RSI14']>70 else '#10b981'}">{r['RSI14']:.0f}</span></div>
                <div class="metric"><label>vs 200DMA</label><span class="mono">{r['Dist_to_SMA200_%']:+.1f}%</span></div>
                <div class="metric"><label>Score</label><span class="mono">{r['Fundy_Score']}</span></div>
            </div>
            <div class="comment-box">{r['Comment']}</div>
            <div class="chart-container">{r['_mini_candle']}</div>
        </div>"""

    buy = df_mkt[df_mkt.Signal == 'BUY'].sort_values('Fundy_Score', ascending=False)
    dca = df_mkt[df_mkt.Signal == 'DCA'].sort_values('Fundy_Score', ascending=False)
    watch = df_mkt[df_mkt.Signal == 'WATCH'].sort_values('Fundy_Score', ascending=False)
    
    grid_html = ""
    for section, sub, b in [('BUY', buy, 'buy'), ('DCA', dca, 'dca'), ('WATCH', watch, 'watch')]:
        if not sub.empty: grid_html += f"<h3 style='margin-top:20px'>{section}</h3><div class='grid'>{''.join([render_card(r, b) for _, r in sub.iterrows()])}</div>"

    # Fundy Table
    def fmt_pct(x): return f"{x*100:.1f}%" if x else "-"
    def fmt_pe(x): return f"{x:.1f}" if x and x > 0 else "-"
    
    f_rows = "".join([f"<tr><td><b>{r['Ticker']}</b></td><td>{r['Fundy_Score']}</td><td class='mono'>{fmt_pct(r['Fundy_ROE'])}</td><td class='mono'>{fmt_pct(r['Fundy_Margin'])}</td><td class='mono'>{fmt_pe(r['Fundy_PE'])}</td><td class='mono'>{r['Fundy_Debt']:.2f}</td></tr>" for _, r in df_mkt.sort_values('Fundy_Score', ascending=False).iterrows()])

    # News Table
    news_html = ""
    if not news_df_all.empty and mkt_code == 'AU':
        n_rows = "".join([f"<tr><td>{n['Date']}</td><td><b>{n['Ticker']}</b></td><td>{n['Type']}</td><td>{n['Headline']}</td></tr>" for _, n in news_df_all.sort_values('Date', ascending=False).iterrows()])
        news_html = f"<h3 style='margin-top:40px'>Local Announcements</h3><div class='card'><div class='table-responsive'><table><thead><tr><th>Date</th><th>Ticker</th><th>Type</th><th>Headline</th></tr></thead><tbody>{n_rows}</tbody></table></div></div>"

    # Playbook Block
    playbook_html = """
    <div class="playbook">
        <b>The "TraderBruh Shield" (0-10 Score):</b><br>
        • <b>Profitability (3 pts):</b> ROE > 15% (Efficiency), Margins > 15% (Pricing Power).<br>
        • <b>Survival (3 pts):</b> Low Debt/Equity (<0.5), High Current Ratio (>1.5).<br>
        • <b>Growth/Val (2 pts):</b> Rev Growth > 10%, PEG < 2.<br><br>
        💎 <b>High Quality (Score 7-10):</b> Fortress balance sheets. OK to DCA dips or size up.<br>
        ⚠️ <b>Speculative/Junk (Score 0-3):</b> Weak fundamentals. If TA says "Buy", use tight stops.
    </div>
    """

    all_market_html += f"""
    <div id="tab-{mkt_code}" class="market-tab-content" style="display:{'block' if mkt_code=='AU' else 'none'}">
        <div style="margin-bottom:20px"><h1 style="font-size:24px; margin:0">{config['name']} Overview</h1></div>
        {playbook_html}
        {grid_html}
        <h3 style="margin-top:40px">Fundamental Health</h3>
        <div class="card"><div class="table-responsive"><table><thead><tr><th>Ticker</th><th>Score</th><th>ROE</th><th>Margin</th><th>P/E</th><th>Debt/Eq</th></tr></thead><tbody>{f_rows}</tbody></table></div></div>
        {news_html}
    </div>"""

# ---------------- CSS/JS & Export ----------------
CSS = """
:root { --bg:#0f172a; --surface:#1e293b; --primary:#3b82f6; --text:#f1f5f9; --muted:#94a3b8; --glass:rgba(30,41,59,0.7); }
body { background:var(--bg); color:var(--text); font-family:sans-serif; margin:0; padding-bottom:60px; }
.nav-wrapper { position:sticky; top:0; z-index:100; background:rgba(15,23,42,0.9); backdrop-filter:blur(10px); padding:10px; border-bottom:1px solid rgba(255,255,255,0.1); }
.nav-tabs { display:flex; gap:10px; max-width:800px; margin:0 auto; }
.tab-btn { background:rgba(255,255,255,0.05); border:none; color:var(--muted); padding:8px 16px; border-radius:20px; cursor:pointer; font-weight:600; transition:all 0.2s; }
.tab-btn.active, .tab-btn:hover { background:var(--primary); color:white; }
.container { max-width:1200px; margin:0 auto; padding:20px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:16px; }
.card { background:var(--glass); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:16px; }
.card-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px; }
.ticker-badge { font-weight:700; color:white; text-decoration:none; font-size:16px; }
.badge { padding:3px 8px; border-radius:4px; font-size:11px; font-weight:600; text-transform:uppercase; }
.badge.buy { background:rgba(16,185,129,0.2); color:#34d399; }
.badge.dca { background:rgba(245,158,11,0.2); color:#fbbf24; }
.badge.watch { background:rgba(59,130,246,0.2); color:#60a5fa; }
.badge.avoid { background:rgba(239,68,68,0.2); color:#f87171; }
.badge.shield-high { background:rgba(16,185,129,0.2); color:#34d399; border:1px solid #34d399; }
.badge.shield-low { background:rgba(239,68,68,0.2); color:#f87171; border:1px solid #f87171; }
.badge.news { background:rgba(168, 85, 247, 0.2); color:#c084fc; }
.metrics-row { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; background:rgba(0,0,0,0.2); padding:8px; border-radius:8px; margin-bottom:10px; font-size:12px; }
.metric label { display:block; color:var(--muted); font-size:10px; }
.comment-box { font-size:13px; line-height:1.4; color:#cbd5e1; margin-bottom:10px; }
.playbook { background:rgba(0,0,0,0.2); padding:12px; border-radius:8px; margin-bottom:16px; font-size:13px; color:#e2e8f0; line-height:1.6; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; padding:8px; color:var(--muted); border-bottom:1px solid rgba(255,255,255,0.1); }
td { padding:8px; border-bottom:1px solid rgba(255,255,255,0.05); }
.mono { font-family:monospace; }
.table-responsive { overflow-x: auto; }
"""

JS = """
function openTab(marketId) {
    document.querySelectorAll('.market-tab-content').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + marketId).style.display = 'block';
    document.getElementById('btn-' + marketId).classList.add('active');
}
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
    <div class="container">
        {all_market_html}
    </div>
</body>
</html>
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_HTML, 'w', encoding='utf-8') as f: f.write(HTML)
print(f'Done: {OUTPUT_HTML}')