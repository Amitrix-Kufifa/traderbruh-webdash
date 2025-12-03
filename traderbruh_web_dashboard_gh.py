# traderbruh_web_dashboard_gh.py
# TraderBruh — Global Web Dashboard (ASX / USA / INDIA)
# Version: Ultimate 5.2 (Sassy Commentary + Robust Data + Fixes)

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
ANN_DIR             = "announcements"
FETCH_DAYS          = 900
MINI_BARS           = 120
SPARK_DAYS          = 90
NEWS_WINDOW_DAYS    = 14
PATTERN_LOOKBACK    = 180
PIVOT_WINDOW        = 4
PRICE_TOL           = 0.03
PATTERNS_CONFIRMED_ONLY = True

RULES = {
    "buy":     {"rsi_min": 45, "rsi_max": 70},
    "dca":     {"rsi_max": 45, "sma200_proximity": 0.05},
    "avoid":   {"death_cross": True},
    "autodca": {"gap_thresh": -2.0, "fill_req": 50.0},
}

BREAKOUT_RULES = {
    "atr_mult":   0.50,
    "vol_mult":   1.30,
    "buffer_pct": 0.003,
}
AUTO_UPGRADE_BREAKOUT = True

# --- Market Definitions ---
MARKETS = {
    "AUS": {
        "name": "Australia (ASX)",
        "tz": "Australia/Sydney",
        "currency": "A$",
        "suffix": ".AX",
        "tickers": {
            # -- ETFs & Indices --
            "A200": ("Betashares A200", "ASX 200 ETF.", "Core"),
            
            # -- Tech & WAAAX --
            "XRO": ("Xero", "Cloud accounting globally.", "Growth"),
            "WTC": ("WiseTech", "Logistics software (CargoWise).", "Growth"),
            "TNE": ("TechnologyOne", "Gov/Edu Enterprise SaaS.", "Core"),
            "NXT": ("NEXTDC", "Data Centers (AI Infrastructure).", "Growth"),
            "ALU": ("Altium", "PCB Design Software.", "Growth"),
            "PME": ("Pro Medicus", "Radiology AI software.", "Growth"),
            "MP1": ("Megaport", "Network-as-a-Service.", "Spec"),
            "CDA": ("Codan", "Comms & Metal Detection.", "Core"),
            "HUB": ("HUB24", "Investment Platform.", "Growth"),
            "NWL": ("Netwealth", "Wealth Platform.", "Growth"),
            
            # -- Engineering & Infra --
            "CVL": ("Civmec", "Defense Shipbuilding/Eng.", "Growth"),
            "GNP": ("GenusPlus", "Power/Grid Infrastructure.", "Growth"),
            
            # -- Defense & Aero --
            "DRO": ("DroneShield", "Counter-UAS/Drone defense.", "Growth"),
            "EOS": ("Electro Optic", "Space & Defense systems.", "Spec"),
            "ASB": ("Austal", "Shipbuilding (US Navy).", "Core"),
            
            # -- Strategic Minerals --
            "PLS": ("Pilbara Minerals", "Hard-rock Lithium.", "Growth"),
            "MIN": ("Mineral Resources", "Mining services + Lithium.", "Growth"),
            "IGO": ("IGO Ltd", "Nickel & Lithium clean energy.", "Growth"),
            "LTR": ("Liontown", "Lithium developer.", "Spec"),
            "LYC": ("Lynas", "Rare Earths (Non-China supply).", "Core"),
            "BOE": ("Boss Energy", "Uranium producer.", "Growth"),
            "PDN": ("Paladin", "Uranium (Namibia).", "Growth"),
            "DYL": ("Deep Yellow", "Uranium exploration.", "Spec"),
            "SYR": ("Syrah", "Graphite Anodes.", "Spec"),
            
            # -- Healthcare & Bio --
            "CSL": ("CSL Limited", "Blood plasma & Vaccines.", "Core"),
            "COH": ("Cochlear", "Hearing implants.", "Core"),
            "RMD": ("ResMed", "Sleep apnea/Digital health.", "Core"),
            "TLX": ("Telix", "Radiopharmaceuticals.", "Growth"),
            "PNV": ("PolyNovo", "Synthetic skin/Wound care.", "Growth"),
            "NAN": ("Nanosonics", "Infection prevention.", "Growth"),
            "NEU": ("Neuren", "Neurodevelopmental drugs.", "Spec"),
            "RAC": ("Race Oncology", "Anti cancer drugs.", "Growth"),
            
            # -- Blue Chip / Cyclical --
            "BHP": ("BHP Group", "Big Australian Miner.", "Core"),
            "FMG": ("Fortescue", "Iron Ore & Green Hydrogen.", "Core"),
            "WDS": ("Woodside", "Oil & Gas energy security.", "Core"),
            "STO": ("Santos", "LNG & Gas.", "Core"),
            "MQG": ("Macquarie", "Global Asset Mgmt.", "Core"),
            "CBA": ("CommBank", "Largest AU Bank.", "Core"),
            "WES": ("Wesfarmers", "Retail conglomerate (Bunnings/Kmart).", "Core"),
            "WOW": ("Woolworths", "Grocery dominance.", "Core"),
            "GMG": ("Goodman Group", "Industrial Real Estate (Warehouses).", "Core"),
            "REA": ("REA Group", "Real Estate advertising.", "Core"),
            "CAR": ("Carsales", "Auto marketplace.", "Core"),
            "JHX": ("James Hardie", "US Housing materials.", "Core"),
            "ALL": ("Aristocrat", "Gaming & Slots.", "Core"),
            "QAN": ("Qantas", "Airline & Loyalty.", "Core"),
            
            # -- Spec/Degen --
            "ZIP": ("Zip Co", "BNPL/Fintech.", "Spec"),
            "BRN": ("BrainChip", "Neuromorphic AI.", "Spec"),
        },
    },
    "USA": {
        "name": "United States (Wall St)",
        "tz": "America/New_York",
        "currency": "U$",
        "suffix": "",
        "tickers": {
            # -- The Magnificent 7 --
            "NVDA": ("NVIDIA", "AI Hardware Leader.", "Core"),
            "MSFT": ("Microsoft", "Cloud & AI (OpenAI).", "Core"),
            "AAPL": ("Apple", "Consumer ecosystem.", "Core"),
            "AMZN": ("Amazon", "AWS & Ecommerce.", "Core"),
            "GOOG": ("Alphabet", "Search & DeepMind.", "Core"),
            "META": ("Meta", "Social & Ads.", "Core"),
            "TSLA": ("Tesla", "EVs, Robotics, FSD.", "Growth"),
            
            # -- Semiconductors & Hardware --
            "AMD":  ("AMD", "Chips (CPU/GPU).", "Growth"),
            "AVGO": ("Broadcom", "AI Networking & Custom chips.", "Core"),
            "TSM":  ("TSMC", "The world's foundry.", "Core"),
            "ARM":  ("ARM", "Chip architecture IP.", "Growth"),
            "MU":   ("Micron", "Memory for AI.", "Growth"),
            "SMCI": ("Super Micro", "AI Servers (High Volatility).", "Spec"),
            
            # -- Cybersecurity --
            "PANW": ("Palo Alto", "Cybersec platform.", "Core"),
            "CRWD": ("CrowdStrike", "Endpoint security.", "Core"),
            "NET":  ("Cloudflare", "Internet infrastructure.", "Growth"),
            
            # -- Software / SaaS / Data --
            "PLTR": ("Palantir", "Gov Intel & Data AI.", "Growth"),
            "NOW":  ("ServiceNow", "Enterprise workflow.", "Core"),
            "CRM":  ("Salesforce", "CRM Giant.", "Core"),
            "SNOW": ("Snowflake", "Cloud Data.", "Growth"),
            "UBER": ("Uber", "Mobility & Delivery.", "Growth"),
            
            # -- Fintech & Crypto --
            "COIN": ("Coinbase", "Crypto Exchange.", "Growth"),
            "MSTR": ("MicroStrategy", "Bitcoin Proxy.", "Spec"),
            "SQ":   ("Block", "Payments & CashApp.", "Growth"),
            "PYPL": ("PayPal", "Digital Payments.", "Core"),
            "HOOD": ("Robinhood", "Retail trading.", "Spec"),
            "AFRM": ("Affirm", "BNPL Lending.", "Spec"),
            
            # -- Healthcare / Weight Loss --
            "LLY":  ("Eli Lilly", "GLP-1 Weight loss leader.", "Core"),
            "NVO":  ("Novo Nordisk", "Ozempic/Wegovy.", "Core"),
            "VRTX": ("Vertex", "Cystic Fibrosis/Gene editing.", "Core"),
            "BMRN": ("BioMarin", "Biotech: Genetic therapies.", "Growth"),
            
            # -- Defense & Industrial --
            "LMT":  ("Lockheed", "Defense contractor.", "Core"),
            "RTX":  ("Raytheon", "Missiles & Aero.", "Core"),
            "GE":   ("GE Aerospace", "Jet Engines.", "Core"),
            
            # -- Consumer / Growth / Degen --
            "COST": ("Costco", "The ultimate retailer.", "Core"),
            "CELH": ("Celsius", "Energy Drinks (Growth).", "Growth"),
            "ONON": ("On Holding", "Running shoes growth.", "Growth"),
            "DKNG": ("DraftKings", "Sports Betting.", "Spec"),
            "RKT":  ("Rocket", "Mortgage Tech.", "Growth"),
            "SOFI": ("SoFi", "Neobank/Student Loans.", "Growth"),
            "FUBO": ("FuboTV", "Sports Streaming.", "Spec"),
            "PGY":  ("Pagaya", "AI Lending.", "Spec"),
            "GME":  ("GameStop", 'Video Game Retailer', "Spec"),
            
            # -- True Degens --
            "BMNR": ("BitMine", "Crypto Mining Hardware.", "Growth"),
        },
    },
    "IND": {
        "name": "India (NSE)",
        "tz": "Asia/Kolkata",
        "currency": "₹",
        "suffix": ".NS",
        "tickers": {
            # -- The Titans --
            "RELIANCE":   ("Reliance", "Oil, Retail, Jio Telecom.", "Core"),
            "HDFCBANK":   ("HDFC Bank", "Largest private bank.", "Core"),
            "ICICIBANK":  ("ICICI Bank", "Banking leader.", "Core"),
            "SBIN":       ("SBI", "State Bank of India (PSU).", "Core"),
            "LICI":       ("LIC", "Insurance giant.", "Core"),
            
            # -- IT Services --
            "TCS":      ("TCS", "IT Services global.", "Core"),
            "INFY":     ("Infosys", "IT Services global.", "Core"),
            "HCLTECH":  ("HCL", "IT & Engineering.", "Core"),
            
            # -- Auto & Mobility --
            "TATAMOTORS": ("Tata Motors", "EVs + Jaguar Land Rover.", "Growth"),
            "M&M":        ("Mahindra", "SUVs & Tractors.", "Core"),
            "MARUTI":     ("Maruti", "Passenger cars.", "Core"),
            "BAJAJ-AUTO": ("Bajaj", "2 & 3 Wheelers.", "Core"),
            
            # -- Defense & PSU --
            "HAL":      ("HAL", "Hindustan Aeronautics (Jets).", "Core"),
            "BEL":      ("Bharat Elec", "Defense electronics.", "Core"),
            "MAZDOCK":  ("Mazagon", "Shipbuilding/Submarines.", "Growth"),
            "COCHINSHIP": ("Cochin Ship", "Shipyards.", "Growth"),
            
            # -- Infrastructure & Power --
            "ADANIENT":   ("Adani Ent", "Infra incubator.", "Growth"),
            "ADANIGREEN": ("Adani Green", "Renewables.", "Growth"),
            "NTPC":       ("NTPC", "Power generation.", "Core"),
            "TATAPOWER":  ("Tata Power", "Power & EV Charging.", "Core"),
            "LT":         ("L&T", "Construction & Engineering.", "Core"),
            
            # -- Railways --
            "IRFC":     ("IRFC", "Railway Finance.", "Growth"),
            "RVNL":     ("RVNL", "Rail Infrastructure.", "Growth"),
            
            # -- Consumer & New Age --
            "ITC":      ("ITC", "FMCG, Hotels, Tobacco.", "Core"),
            "TITAN":    ("Titan", "Jewelry (Wealth proxy).", "Core"),
            "VBL":      ("Varun Bev", "Pepsi bottler (Growth).", "Growth"),
            "ZOMATO":   ("Zomato", "Food Delivery/Blinkit.", "Growth"),
            "PAYTM":    ("Paytm", "Fintech turnaround?", "Spec"),
            "TRENT":    ("Trent", "Fashion Retail (Zudio).", "Growth"),
            "DMART":    ("Avenue Super", "Retail Chain.", "Core"),
            "BAJFINANCE": ("Bajaj Fin", "Consumer Lending.", "Core"),
        },
    },
}

# ---------------- Robust Data Fetcher ----------------

def fetch_prices(symbol: str, tz_name: str) -> pd.DataFrame:
    try:
        # 1. Configured Fetch
        df = yf.download(symbol, period=f"{FETCH_DAYS}d", interval="1d", auto_adjust=False, progress=False, group_by="column", prepost=False)
        
        # 2. Fallback to Max if empty
        if df is None or df.empty:
            df = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False, group_by="column", prepost=False)

        if df is None or df.empty: return pd.DataFrame()

        # 3. Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns.get_level_values(-1):
                df = df.xs(symbol, axis=1, level=-1, drop_level=True)
            else:
                df.columns = df.columns.get_level_values(0)

        # 4. Normalize columns
        col_map = {}
        for col in df.columns:
            c = str(col).lower()
            if c == "open": col_map[col] = "Open"
            elif c == "high": col_map[col] = "High"
            elif c == "low": col_map[col] = "Low"
            elif c in ("close", "price"): col_map[col] = "Close"
            elif "volume" in c: col_map[col] = "Volume"
        
        df = df.rename(columns=col_map)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in needed): return pd.DataFrame()

        df = df[needed].reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df["Date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")

        # 5. Intraday Stitch
        market_tz = zoneinfo.ZoneInfo(tz_name)
        now_mkt = datetime.now(market_tz)
        last_date_mkt = df["Date"].dt.tz_convert(market_tz).dt.date.max()

        if (now_mkt.time() >= time(10, 0)) and (last_date_mkt < now_mkt.date()):
            try:
                intr = yf.download(symbol, period="5d", interval="60m", auto_adjust=False, progress=False, prepost=False, group_by="column")
                if intr is not None and not intr.empty:
                    if isinstance(intr.columns, pd.MultiIndex):
                        if symbol in intr.columns.get_level_values(-1): intr = intr.xs(symbol, axis=1, level=-1, drop_level=True)
                        else: intr.columns = intr.columns.get_level_values(0)
                    intr = intr.reset_index()
                    intr["Date"] = pd.to_datetime(intr[intr.columns[0]], utc=True, errors="coerce")
                    last = intr.tail(1).iloc[0]
                    top = pd.DataFrame([{
                        "Date": last["Date"], "Open": float(last["Open"]), "High": float(last["High"]),
                        "Low": float(last["Low"]), "Close": float(last["Close"]), "Volume": float(last["Volume"]),
                    }])
                    df = pd.concat([df, top], ignore_index=True)
            except Exception: pass

        df["Date"] = df["Date"].dt.tz_convert(market_tz).dt.tz_localize(None)
        return df.dropna(subset=["Close"])

    except Exception as e:
        print(f"[ERROR] fetch_prices failed for {symbol}: {e}")
        return pd.DataFrame()

def fetch_deep_fundamentals(symbol: str):
    try:
        tick = yf.Ticker(symbol)
        info = tick.info
        try: bs, is_, cf = tick.balance_sheet, tick.income_stmt, tick.cashflow
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
            past = get_item(df, item_names, years - 1)
            if past <= 0 or curr <= 0: return 0
            return (curr / past) ** (1 / (years - 1)) - 1

        roe_3y = 0
        try:
            if not is_.empty and not bs.empty:
                roes = []
                for i in range(min(3, len(is_.columns), len(bs.columns))):
                    ni = get_item(is_, ["Net Income"], i)
                    eq = get_item(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest"], i)
                    if eq > 0: roes.append(ni / eq)
                if roes: roe_3y = sum(roes) / len(roes)
        except: pass

        ocf = get_item(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        net_inc = get_item(is_, ["Net Income"])
        high_quality_earnings = ocf > net_inc
        marg_curr = info.get("profitMargins", 0)
        
        score = 0
        if roe_3y > 0.15: score += 2
        elif roe_3y > 0.10: score += 1
        if marg_curr > 0.10: score += 1
        if high_quality_earnings: score += 0.5

        curr_ratio = info.get("currentRatio", 0) or 0
        debt_eq = info.get("debtToEquity", 999)
        if debt_eq and debt_eq > 50: debt_eq = debt_eq / 100.0

        cash = get_item(bs, ["Cash And Cash Equivalents", "Cash Financial"])
        lt_debt = get_item(bs, ["Long Term Debt"])
        
        if cash > lt_debt: score += 1.5
        elif debt_eq < 0.5: score += 1
        if curr_ratio > 1.5: score += 1
        elif curr_ratio > 1.1: score += 0.5

        shares_curr = get_item(bs, ["Share Issued", "Ordinary Shares Number"], 0)
        shares_old = get_item(bs, ["Share Issued", "Ordinary Shares Number"], 2)
        is_buyback = False
        if shares_old > 0:
            change = (shares_curr - shares_old) / shares_old
            if change < -0.01: score += 1.5; is_buyback = True
            elif change < 0.05: score += 1
        
        rev_cagr = get_cagr(is_, ["Total Revenue", "Operating Revenue"], 3)
        if rev_cagr > 0.10: score += 1
        
        peg, pe = info.get("pegRatio", 0) or 0, info.get("trailingPE", 0) or 0
        if (peg and 0 < peg < 2.0) or (pe and 0 < pe < 20): score += 1
        
        score = min(score, 10)
        tier = "Fortress" if score >= 7 else ("Quality" if score >= 4 else "Spec")
        
        return {
            "score": round(score, 1), "tier": tier, "roe_3y": roe_3y, "margins": marg_curr,
            "debt_eq": debt_eq, "rev_cagr": rev_cagr, "is_buyback": is_buyback, "pe": pe, "cash": cash
        }
    except:
        return {"score": 0, "tier": "Error", "roe_3y": 0, "margins": 0, "debt_eq": 0, "rev_cagr": 0, "is_buyback": False, "pe": 0, "cash": 0}

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("Date").reset_index(drop=True)
    x["SMA20"]  = x["Close"].rolling(20).mean()
    x["SMA50"]  = x["Close"].rolling(50).mean()
    x["SMA200"] = x["Close"].rolling(200).mean()
    x["EMA21"]  = x["Close"].ewm(span=21, adjust=False).mean()
    x["High20"] = x["High"].rolling(20).max()
    x["High52W"]= x["High"].rolling(252).max()
    x["Vol20"]  = x["Volume"].rolling(20).mean()

    chg = x["Close"].diff()
    gains = chg.clip(lower=0).rolling(14).mean()
    losses = (-chg).clip(lower=0).rolling(14).mean()
    RS = gains / losses
    x["RSI14"] = 100 - (100 / (1 + RS))

    x["Dist_to_52W_High_%"] = (x["Close"] / x["High52W"] - 1) * 100.0
    x["Dist_to_SMA200_%"]   = (x["Close"] / x["SMA200"]  - 1) * 100.0

    x["H-L"] = x["High"] - x["Low"]
    x["H-C"] = (x["High"] - x["Close"].shift(1)).abs()
    x["L-C"] = (x["Low"]  - x["Close"].shift(1)).abs()
    x["TR"]  = x[["H-L", "H-C", "L-C"]].max(axis=1)
    x["ATR14"] = x["TR"].rolling(14).mean()
    return x

def label_row(r: pd.Series) -> str:
    buy_ok = (r["Close"] > r["SMA200"]) and (r["Close"] > r["High20"]) and (r["SMA50"] > r["SMA200"]) and (RULES["buy"]["rsi_min"] <= r["RSI14"] <= RULES["buy"]["rsi_max"])
    dca_ok = (r["Close"] >= r["SMA200"]) and (r["RSI14"] < RULES["dca"]["rsi_max"]) and (r["Close"] <= r["SMA200"] * (1 + RULES["dca"]["sma200_proximity"]))
    avoid = (r["SMA50"] < r["SMA200"]) if RULES["avoid"]["death_cross"] else False
    if buy_ok: return "BUY"
    if dca_ok: return "DCA"
    if avoid: return "AVOID"
    return "WATCH"

def auto_dca_gate(ind: pd.DataFrame):
    if len(ind) < 3: return False, {"reason": "insufficient data"}
    D0, D1, D2 = ind.iloc[-1], ind.iloc[-2], ind.iloc[-3]
    gap_pct = (D1["Open"] / D2["Close"] - 1) * 100.0
    if not np.isfinite(gap_pct) or gap_pct > RULES["autodca"]["gap_thresh"]: return False, {"gap_pct": float(gap_pct)}
    gap_mid = (D1["High"] + D1["Low"]) / 2.0
    reclaim_mid = bool(D0["Close"] > gap_mid)
    above_ema21 = bool(D0["Close"] > D0["EMA21"])
    gap_size = max(D2["Close"] - D1["Open"], 0.0)
    fill_pct = float(0.0 if gap_size == 0 else (D0["Close"] - D1["Open"]) / gap_size * 100.0)
    flag = reclaim_mid and above_ema21 and (fill_pct >= RULES["autodca"]["fill_req"])
    return flag, {"gap_pct": float(gap_pct), "reclaim_mid": reclaim_mid, "above_ema21": above_ema21, "gap_fill_%": fill_pct}

def _pivots(ind, window=PIVOT_WINDOW):
    v = ind.tail(PATTERN_LOOKBACK).reset_index(drop=True).copy()
    v["PH"] = (v["High"] == v["High"].rolling(window * 2 + 1, center=True).max()).fillna(False)
    v["PL"] = (v["Low"]  == v["Low"].rolling(window * 2 + 1, center=True).min()).fillna(False)
    return v

def _similar(a, b, tol=PRICE_TOL):
    m = (a + b) / 2.0
    return (abs(a - b) / m) <= tol

def detect_double_bottom(ind):
    v = _pivots(ind)
    lows = v.index[v["PL"]].tolist()
    out = []
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            li, lj = lows[i], lows[j]
            if lj - li < 10: continue
            p1, p2 = float(v.loc[li, "Low"]), float(v.loc[lj, "Low"])
            if not _similar(p1, p2): continue
            neck = float(v.loc[li:lj, "High"].max())
            confirmed = bool(v["Close"].iloc[-1] > neck)
            conf = 0.6 + (0.2 if confirmed else 0.0)
            if np.isfinite(v["Vol20"].iloc[-1]) and confirmed and v["Volume"].iloc[-1] > 1.2 * v["Vol20"].iloc[-1]: conf += 0.2
            lines = [("h", v.loc[li, "Date"], v.loc[lj, "Date"], (p1 + p2) / 2.0), ("h", v.loc[li, "Date"], v["Date"].iloc[-1], neck)]
            out.append({"name": "Double Bottom", "status": "confirmed" if confirmed else "forming", "confidence": round(min(conf, 1.0), 2), "levels": {"base": round((p1 + p2) / 2.0, 4), "neckline": round(neck, 4)}, "lines": lines})
            return out
    return out

def detect_double_top(ind):
    v = _pivots(ind)
    highs = v.index[v["PH"]].tolist()
    out = []
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            hi, hj = highs[i], highs[j]
            if hj - hi < 10: continue
            p1, p2 = float(v.loc[hi, "High"]), float(v.loc[hj, "High"])
            if not _similar(p1, p2): continue
            neck = float(v.loc[hi:hj, "Low"].min())
            confirmed = bool(v["Close"].iloc[-1] < neck)
            conf = 0.6 + (0.2 if confirmed else 0.0)
            if np.isfinite(v["Vol20"].iloc[-1]) and confirmed and v["Volume"].iloc[-1] > 1.2 * v["Vol20"].iloc[-1]: conf += 0.2
            lines = [("h", v.loc[hi, "Date"], v.loc[hj, "Date"], (p1 + p2) / 2.0), ("h", v.loc[hi, "Date"], v["Date"].iloc[-1], neck)]
            out.append({"name": "Double Top", "status": "confirmed" if confirmed else "forming", "confidence": round(min(conf, 1.0), 2), "levels": {"ceiling": round((p1 + p2) / 2.0, 4), "neckline": round(neck, 4)}, "lines": lines})
            return out
    return out

def detect_inverse_hs(ind):
    v = _pivots(ind)
    lows = v.index[v["PL"]].tolist()
    out = []
    for i in range(len(lows) - 2):
        l1, h, l2 = lows[i], lows[i + 1], lows[i + 2]
        pL1, pH, pL2 = float(v.loc[l1, "Low"]), float(v.loc[h, "Low"]), float(v.loc[l2, "Low"])
        if not (pH < pL1 * (1 - 0.04) and pH < pL2 * (1 - 0.04)): continue
        if not _similar(pL1, pL2): continue
        left_high = float(v.loc[l1:h, "High"].max())
        right_high = float(v.loc[h:l2, "High"].max())
        confirmed = bool(v["Close"].iloc[-1] > min(left_high, right_high))
        conf = 0.6 + (0.2 if confirmed else 0.0)
        lines = [("seg", v.loc[l1, "Date"], left_high, v.loc[l2, "Date"], right_high)]
        out.append({"name": "Inverse H&S", "status": "confirmed" if confirmed else "forming", "confidence": round(min(conf, 1.0), 2), "levels": {"neck_left": round(left_high, 4), "neck_right": round(right_high, 4)}, "lines": lines})
        return out
    return out

def detect_hs(ind):
    v = _pivots(ind)
    highs = v.index[v["PH"]].tolist()
    out = []
    for i in range(len(highs) - 2):
        l1, h, l2 = highs[i], highs[i + 1], highs[i + 2]
        pL1, pH, pL2 = float(v.loc[l1, "High"]), float(v.loc[h, "High"]), float(v.loc[l2, "High"])
        if not (pH > pL1 * (1 + 0.04) and pH > pL2 * (1 + 0.04)): continue
        if not _similar(pL1, pL2): continue
        left_low = float(v.loc[l1:h, "Low"].min())
        right_low = float(v.loc[h:l2, "Low"].min())
        confirmed = bool(v["Close"].iloc[-1] < max(left_low, right_low))
        conf = 0.6 + (0.2 if confirmed else 0.0)
        lines = [("seg", v.loc[l1, "Date"], left_low, v.loc[l2, "Date"], right_low)]
        out.append({"name": "Head & Shoulders", "status": "confirmed" if confirmed else "forming", "confidence": round(min(conf, 1.0), 2), "levels": {"neck_left": round(left_low, 4), "neck_right": round(right_low, 4)}, "lines": lines})
        return out
    return out

def detect_triangles(ind):
    v = _pivots(ind)
    tail = v.tail(120).copy()
    phs, pls = tail[tail["PH"]], tail[tail["PL"]]
    out = []
    if len(phs) >= 2 and len(pls) >= 2:
        ph_vals = phs["High"].values
        for i in range(len(ph_vals) - 1):
            if _similar(ph_vals[i], ph_vals[i + 1]):
                res = (ph_vals[i] + ph_vals[i + 1]) / 2.0
                slope = np.polyfit(np.arange(len(pls)), pls["Low"].values, 1)[0]
                if slope > 0:
                    confirmed = bool(tail["Close"].iloc[-1] > res)
                    conf = 0.55 + (0.25 if confirmed else 0.0)
                    lines = [("h", pls["Date"].iloc[0], tail["Date"].iloc[-1], res), ("seg", pls["Date"].iloc[0], pls["Low"].iloc[0], pls["Date"].iloc[-1], pls["Low"].iloc[-1])]
                    out.append({"name": "Ascending Triangle", "status": "confirmed" if confirmed else "forming", "confidence": round(min(conf, 1.0), 2), "levels": {"resistance": round(res, 4)}, "lines": lines})
                    break
        pl_vals = pls["Low"].values
        for i in range(len(pl_vals) - 1):
            if _similar(pl_vals[i], pl_vals[i + 1]):
                sup = (pl_vals[i] + pl_vals[i + 1]) / 2.0
                slope = np.polyfit(np.arange(len(phs)), phs["High"].values, 1)[0]
                if slope < 0:
                    confirmed = bool(tail["Close"].iloc[-1] < sup)
                    conf = 0.55 + (0.25 if confirmed else 0.0)
                    lines = [("h", phs["Date"].iloc[0], tail["Date"].iloc[-1], sup), ("seg", phs["Date"].iloc[0], phs["High"].iloc[0], phs["Date"].iloc[-1], phs["High"].iloc[-1])]
                    out.append({"name": "Descending Triangle", "status": "confirmed" if confirmed else "forming", "confidence": round(min(conf, 1.0), 2), "levels": {"support": round(sup, 4)}, "lines": lines})
                    break
    return out

def detect_flag(ind):
    if len(ind) < 60: return False, {}
    look = ind.tail(40)
    impulse = (look["Close"].max() / look["Close"].min() - 1) * 100
    if not np.isfinite(impulse) or impulse < 12: return False, {}
    win = 14
    tail = ind.tail(max(win, 8)).copy()
    x = np.arange(len(tail))
    hi, lo = np.polyfit(x, tail["High"].values, 1), np.polyfit(x, tail["Low"].values, 1)
    slope_pct = (hi[0] / tail["Close"].iloc[-1]) * 100
    ch = np.polyval(hi, x[-1]) - np.polyval(lo, x[-1])
    tight = ch <= max(0.4 * (look["Close"].max() - look["Close"].min()), 0.02 * tail["Close"].iloc[-1])
    gentle = (-0.006 <= slope_pct <= 0.002)
    return (tight and gentle), {"hi": hi.tolist(), "lo": lo.tolist(), "win": win}

def pattern_bias(name: str) -> str:
    if name in ("Double Bottom", "Inverse H&S", "Ascending Triangle", "Bull Flag"): return "bullish"
    if name in ("Double Top", "Head & Shoulders", "Descending Triangle"): return "bearish"
    return "neutral"

def breakout_ready_dt(ind: pd.DataFrame, pat: dict, rules: dict):
    if not pat or pat.get("name") != "Double Top": return False, {}
    last = ind.iloc[-1]
    atr, vol, vol20 = float(last.get("ATR14", np.nan)), float(last.get("Volume", np.nan)), float(last.get("Vol20", np.nan))
    ceiling = float(pat.get("levels", {}).get("ceiling", np.nan))
    if not (np.isfinite(atr) and np.isfinite(vol) and np.isfinite(vol20) and np.isfinite(ceiling)): return False, {}
    close = float(last["Close"])
    ok_price = (close >= ceiling * (1.0 + rules["buffer_pct"])) and (close >= ceiling + rules["atr_mult"] * atr)
    ok_vol = (vol20 > 0) and (vol >= rules["vol_mult"] * vol20)
    return bool(ok_price and ok_vol), {"ceiling": round(ceiling, 4), "atr": round(atr, 4), "stop": round(close - atr, 4)}

# ---------------- Commentary ----------------

def is_euphoria(r):
    return (r["Dist_to_52W_High_%"] > -3.5) and (r["Dist_to_SMA200_%"] > 50.0) and (r["RSI14"] >= 70.0)

def comment_for_row(r: pd.Series) -> str:
    d200 = r["Dist_to_SMA200_%"]
    d52  = r["Dist_to_52W_High_%"]
    dist_high = abs(d52)
    rsi  = r["RSI14"]
    sig  = str(r.get("Signal", "")).upper()
    f_score = r.get("Fundy_Score", 0)
    cat  = str(r.get("Category", "Core"))
    eup  = is_euphoria(r)

    is_degen = (cat.lower() == "spec")
    is_trash = (f_score < 4)
    is_fortress = (f_score >= 7)

    base = ""

    if sig == "BUY":
        if is_fortress: base = f"<b>🚀 ROCKET FUEL (High Conviction):</b> Strong Uptrend + Fortress Shield ({f_score}/10). Scale in."
        elif is_trash: base = f"<b>🗑️ TRASH RALLY (Momentum):</b> Flying, but engine broken ({f_score}/10). Trade chart, tight stops."
        else: base = f"<b>✅ STANDARD BUY:</b> Healthy trend > 200DMA. Fundys OK ({f_score}/10). Starter size."
    elif sig == "DCA":
        if is_trash: base = f"<b>🩸 TOXIC KNIFE:</b> Dip looks cheap, but Shield is {f_score}/10. Don't catch it."
        elif is_fortress: base = f"<b>💎 COMPOUNDER ON SALE:</b> Rare pullback on 7+ Quality. Patient money adds."
        else: base = f"<b>📉 SWING ZONE:</b> Near 200DMA (Δ{d200:.1f}%) with RSI cooling. Decent bounce play."
    elif sig == "WATCH":
        if eup:
            if is_trash: base = f"<b>🚨 EXIT SCAM WARNING:</b> Garbage ({f_score}/10) in euphoria. Take profit."
            else: base = f"<b>🍾 EUPHORIA ZONE:</b> Price stretched. Don't chase."
        elif is_fortress:
            base = f"<b>🎯 SNIPER LIST:</b> Elite quality ({f_score}/10) drifting. Stalk for setup."
        else:
            if dist_high < 5.0: base = f"<b>🚪 KNOCKING ON DOOR:</b> Coiling {dist_high:.1f}% below highs. Watch breakout."
            elif dist_high > 20.0: base = f"<b>🤕 RECOVERY WARD:</b> Down {dist_high:.1f}% from highs. Needs repair."
            elif rsi > 60: base = f"<b>🔥 HEATING UP:</b> Momentum building (RSI {rsi:.0f}), entry not clean yet."
            else: base = f"<b>💤 CHOP CITY:</b> Stuck ({dist_high:.1f}% off highs). Dead money."
    elif sig == "AVOID":
        if is_fortress: base = f"<b>🥶 VALUE TRAP:</b> Great business ({f_score}/10) in downtrend. Wait for base."
        else: base = f"<b>💀 RADIOACTIVE:</b> Broken chart + Weak business. Delete."
    else: base = "Neutral."

    if is_degen: base += " <br><i>⚠️ <b>DEGEN WARNING:</b> Spec play. Casino money only.</i>"
    return base

# ---------------- Rendering & Parsing ----------------

def mini_candle(ind, flag_info=None, pattern_lines=None):
    v = ind.tail(MINI_BARS).copy()
    fig = go.Figure(data=[go.Candlestick(x=v["Date"], open=v["Open"], high=v["High"], low=v["Low"], close=v["Close"], increasing_line_color="#4ade80", decreasing_line_color="#f87171")])
    if "SMA20" in v.columns:
        fig.add_trace(go.Scatter(x=v["Date"], y=v["SMA20"], line=dict(color="rgba(56,189,248,0.8)", width=1)))
    if flag_info:
        t2 = ind.tail(flag_info["win"])
        x = np.arange(len(t2))
        fig.add_trace(go.Scatter(x=t2["Date"], y=np.polyval(flag_info["hi"], x), line=dict(dash="dash", color="#a855f7")))
        fig.add_trace(go.Scatter(x=t2["Date"], y=np.polyval(flag_info["lo"], x), line=dict(dash="dash", color="#a855f7")))
    if pattern_lines:
        for ln in pattern_lines:
            if ln[0] == "h": fig.add_trace(go.Scatter(x=[ln[1], ln[2]], y=[ln[3], ln[3]], mode="lines", line=dict(color="#facc15", width=2, dash="dot")))
            elif ln[0] == "seg": fig.add_trace(go.Scatter(x=[ln[1], ln[3]], y=[ln[2], ln[4]], mode="lines", line=dict(color="#facc15", width=2)))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=130, width=280, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "staticPlot": True})

def mini_spark(ind):
    v = ind.tail(SPARK_DAYS)
    fig = go.Figure(go.Scatter(x=v["Date"], y=v["Close"], mode="lines", line=dict(width=1, color="#94a3b8")))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=50, width=120, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "staticPlot": True})

def parse_announcements(market_code):
    if market_code != "AUS": return pd.DataFrame(columns=["Date", "Ticker", "Type", "Headline"])
    rows = []
    
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

    def read_pdf_first_text(path):
        if not HAVE_PYPDF: return ""
        try: return re.sub(r'[ \t]+', ' ', PdfReader(path).pages[0].extract_text() or '')
        except: return ""

    today = datetime.now().date()
    if os.path.isdir(ANN_DIR):
        for fp in glob.glob(os.path.join(ANN_DIR, "*.pdf")):
            fname = os.path.basename(fp)
            tick = re.search(r"([A-Z]{3})[_-]", fname)
            tick = tick.group(1) if tick else "?"
            text = read_pdf_first_text(fp)
            _type, tag = next(( (l, t) for l, p, t in NEWS_TYPES_REGEX if re.search(p, fname + " " + text, re.I) ), ('Announcement', 'gen'))
            d_time = datetime.fromtimestamp(os.path.getmtime(fp))
            rows.append({"Date": d_time.strftime("%Y-%m-%d"), "Ticker": tick, "Type": _type, "Headline": _type, "Recent": (today - d_time.date()).days <= NEWS_WINDOW_DAYS})
            
    return pd.DataFrame(rows)

def process_market(m_code, m_conf):
    print(f"--> Analyzing {m_conf['name']}...")
    snaps = []
    frames = []
    
    news_df = parse_announcements(m_code)

    for t_key, t_meta in m_conf["tickers"].items():
        t_name, t_desc, t_cat = t_meta
        full_sym = f"{t_key}{m_conf['suffix']}"
        
        df = fetch_prices(full_sym, m_conf["tz"])
        if df.empty: continue
        df["Ticker"] = t_key
        frames.append(df)
        
        fundy = fetch_deep_fundamentals(full_sym)
        
        # RELAXED FILTER: No SMA200 required
        ind = indicators(df).dropna(subset=["High20", "RSI14", "EMA21", "Vol20", "ATR14"])
        if ind.empty: continue
        last = ind.iloc[-1]
        sig = label_row(last)
        
        flag_flag, flag_det = detect_flag(ind)
        pats = detect_double_bottom(ind) + detect_double_top(ind) + detect_inverse_hs(ind) + detect_hs(ind) + detect_triangles(ind)
        if PATTERNS_CONFIRMED_ONLY: pats = [p for p in pats if p.get("status") == "confirmed"]
        
        brk_ready, brk_info = False, {}
        if (dts := [p for p in pats if p["name"]=="Double Top"]):
            brk_ready, brk_info = breakout_ready_dt(ind, dts[0], BREAKOUT_RULES)
        
        if AUTO_UPGRADE_BREAKOUT and brk_ready: sig = "BUY"

        gate_flag, gate_det = auto_dca_gate(ind)
        
        pname = pats[0]["name"] if pats else ""
        pbias = pattern_bias(pname)
        sig_str = str(sig).lower()
        is_aligned = (pbias == "neutral" or sig_str == "watch") or (pbias == "bullish" and sig_str in ["buy", "dca"]) or (pbias == "bearish" and sig_str == "avoid")
        palign = "ALIGNED" if is_aligned else "CONFLICT"

        if t_cat != "Spec" and fundy["score"] < 4: sig = "AVOID"

        snaps.append({
            "Ticker": t_key, "Name": t_name, "Desc": t_desc, "Category": t_cat,
            "LastDate": last["Date"].strftime("%Y-%m-%d"), "LastClose": float(last["Close"]),
            "SMA200": float(last.get("SMA200", 0.0)), "RSI14": float(last["RSI14"]),
            "Dist_to_SMA200_%": float(last.get("Dist_to_SMA200_%", 0.0)), "Dist_to_52W_High_%": float(last["Dist_to_52W_High_%"]),
            "Signal": sig, "SignalAuto": False, "Comment": None,
            "Fundy_Score": fundy["score"], "Fundy_Tier": fundy["tier"], "Fundy": fundy,
            "Flag": flag_flag, "_flag_info": flag_det,
            "_pattern_lines": pats[0]["lines"] if pats else None,
            "_pattern_name": pname, "_pattern_status": pats[0]["status"] if pats else "", "_pattern_conf": pats[0]["confidence"] if pats else 0,
            "_pattern_align": palign,
            "AutoDCA_Flag": gate_flag, "AutoDCA_Gap_%": gate_det.get("gap_pct", 0), "AutoDCA_Fill_%": gate_det.get("gap_fill_%", 0),
            "AutoDCA_ReclaimMid": gate_det.get("reclaim_mid", False), "AutoDCA_AboveEMA21": gate_det.get("above_ema21", False),
            "BreakoutReady": brk_ready, "Breakout_Level": brk_info.get("ceiling", 0), "_ind": ind
        })

    snaps_df = pd.DataFrame(snaps)
    if not snaps_df.empty:
        comments, candles, sparks = [], [], []
        for _, r in snaps_df.iterrows():
            r["Comment"] = comment_for_row(r)
            if r["BreakoutReady"]: r["Comment"] += f" • BreakoutReady: > {r['Breakout_Level']:.2f}"
            if not news_df.empty:
                nd = news_df[(news_df["Ticker"] == r["Ticker"]) & (news_df["Recent"])]
                if not nd.empty: r["Comment"] += f" • News: {nd.iloc[-1]['Headline']}"
            comments.append(r["Comment"])
            candles.append(mini_candle(r["_ind"], r["_flag_info"] if r["Flag"] else None, r["_pattern_lines"]))
            sparks.append(mini_spark(r["_ind"]))
        snaps_df["Comment"] = comments
        snaps_df["_mini_candle"] = candles
        snaps_df["_mini_spark"] = sparks
        
    return snaps_df, news_df

# ---------------- HTML Construction ----------------

if __name__ == "__main__":
    print("Starting TraderBruh Global Hybrid v5.2...")
    market_htmls, tab_buttons = [], []
    for m, conf in MARKETS.items():
        df, news = process_market(m, conf)
        market_htmls.append(render_market_html(m, conf, df, news))
        act = "active" if m=="AUS" else ""
        tab_buttons.append(f"<button id='tab-{m}' class='market-tab {act}' onclick=\"switchMarket('{m}')\">{conf['name']}</button>")
    
    full = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>TraderBruh v5.2</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>{CSS}</style><script>{JS}</script></head><body><div class="market-tabs">{''.join(tab_buttons)}</div>{''.join(market_htmls)}</body></html>"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f: f.write(full)
    print("Done:", OUTPUT_HTML)