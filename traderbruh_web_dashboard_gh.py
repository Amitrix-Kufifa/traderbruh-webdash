# traderbruh_web_dashboard_gh.py
# TraderBruh — Global Web Dashboard (ASX / USA / INDIA)
# Version: Ultimate 6.6 (Logic/Playbook Upgrade)
# - Fixed breakout logic (20D/52W highs shifted to avoid self-referencing)
# - Added HOLD + TRIM signals (explicit hodl / take-profit guidance)
# - Optional split/dividend-adjusted indicator series (AdjClose) for cleaner long lookbacks
# - Generic breakout/breakdown detector for multiple patterns (volume + ATR confirmation)
# - Much more directive commentary "playbook" per ticker (entry/add/stop/trim rules)
# - Optional in-dashboard litmus stats (median forward returns after BUY/DCA)

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

# ---------------- Build / Deploy Metadata ----------------
# These are populated automatically in GitHub Actions, but will be blank when you run locally.
def get_build_meta() -> dict:
    sha = os.getenv("GITHUB_SHA", "") or ""
    run_id = os.getenv("GITHUB_RUN_ID", "") or ""
    ref = os.getenv("GITHUB_REF_NAME", "") or os.getenv("GITHUB_REF", "") or ""
    actor = os.getenv("GITHUB_ACTOR", "") or ""
    # Shorten sha for display
    sha7 = sha[:7] if sha else "local"
    return {"sha": sha, "sha7": sha7, "run_id": run_id or "-", "ref": ref or "-", "actor": actor or "-"}

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


# --- Data / Backtest Safety ---
# NOTE: yfinance 'Adj Close' is split/dividend-adjusted. Using it for indicators avoids split artifacts.
USE_ADJUSTED_FOR_INDICATORS = True

# NOTE: Appending intraday bars into a daily series can distort indicators. Kept for experimentation only.
APPEND_INTRADAY_BAR = False

# Optional: lightweight in-dashboard "litmus" stats (forward returns after signals, etc.)
ENABLE_LITMUS_STATS = True
LITMUS_SIGNAL_HORIZONS = (5, 20, 60)   # trading days
LITMUS_LOOKBACK_BARS   = 252 * 6       # ~6 years for stats window

RULES = {
    # BUY = momentum / trend continuation (breakout in a Stage-2 uptrend)
    "buy":     {"rsi_min": 45, "rsi_max": 70, "vol_mult": 1.0},

    # DCA = quality dip near 200DMA (intended for Core/Growth, not pure Spec)
    # allow_below_pct: allow a small undercut of 200DMA without instantly flipping to AVOID
    "dca":     {"rsi_max": 45, "sma200_proximity": 0.05, "allow_below_pct": 0.02},

    # AVOID = damage control regime (trend broken / death cross)
    "avoid":   {"death_cross": True},

    # TRIM = profit-taking regime (usually triggered by euphoria conditions)
    "trim":    {"enabled": True},

    # Risk model (used in commentary)
    "risk":    {"atr_stop_mult": 2.0, "atr_trail_mult": 3.0, "sma200_break_pct": 0.03},

    # Auto-DCA gate: gap-down reclaim setup
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
            # --- ETFs & Indices ---
            "A200": ("Betashares A200", "ASX 200 ETF.", "Core"),
            "VAS":  ("Vanguard AUS", "Australian Shares Index.", "Core"),
            
            # --- CORE: Banking & Financials ---
            "CBA": ("CommBank", "Largest AU Bank.", "Core"),
            "NAB": ("National Australia Bank", "Big-4 Business Bank.", "Core"),
            "WBC": ("Westpac", "Big-4 Bank.", "Core"),
            "ANZ": ("ANZ Group", "Big-4 Bank.", "Core"),
            "MQG": ("Macquarie", "Global Asset Mgmt.", "Core"),
            "IAG": ("Insurance Australia", "General Insurer.", "Core"),
            "QBE": ("QBE Insurance", "Global Insurer.", "Core"),
            "SUN": ("Suncorp", "Bank & Insurance.", "Core"),
            
            # --- CORE: Resources & Energy ---
            "BHP": ("BHP Group", "Big Australian Miner.", "Core"),
            "RIO": ("Rio Tinto", "Iron Ore & Diversified.", "Core"),
            "FMG": ("Fortescue", "Iron Ore & Green Hydrogen.", "Core"),
            "WDS": ("Woodside", "Oil & Gas Major.", "Core"),
            "STO": ("Santos", "LNG & Gas.", "Core"),
            "S32": ("South32", "Diversified Base Metals.", "Core"),
            "NST": ("Northern Star", "Gold Major.", "Core"),
            "PLS": ("Pilbara Minerals", "Lithium Major.", "Core"),

            # --- CORE: Industrials, Telco, Staples ---
            "TLS": ("Telstra", "Telecom Infrastructure.", "Core"),
            "WES": ("Wesfarmers", "Retail Conglomerate.", "Core"),
            "WOW": ("Woolworths", "Grocery Dominance.", "Core"),
            "COL": ("Coles Group", "Supermarkets.", "Core"),
            "TCL": ("Transurban", "Toll Roads.", "Core"),
            "APA": ("APA Group", "Gas Infrastructure.", "Core"),
            "ALL": ("Aristocrat", "Gaming Machines.", "Core"),
            "BXB": ("Brambles", "Logistics Pallets.", "Core"),
            "CSL": ("CSL Limited", "Biotech Giant.", "Core"),
            "COH": ("Cochlear", "Hearing Implants.", "Core"),
            "RMD": ("ResMed", "Sleep Apnea.", "Core"),

            # --- GROWTH: Tech / Software / Platform ---
            "XRO": ("Xero", "Cloud Accounting.", "Growth"),
            "WTC": ("WiseTech", "Logistics Software.", "Growth"),
            "TNE": ("TechnologyOne", "Enterprise SaaS.", "Core"),
            "NXT": ("NEXTDC", "Data Centers.", "Growth"),
            "PME": ("Pro Medicus", "Health Imaging AI.", "Growth"),
            "AD8": ("Audinate", "Audio Networking.", "Growth"),
            "HUB": ("HUB24", "Wealth Platform.", "Growth"),
            "NWL": ("Netwealth", "Wealth Platform.", "Growth"),
            "CAR": ("Carsales", "Auto Marketplace.", "Core"),
            "REA": ("REA Group", "Property Listings.", "Core"),
            "SEK": ("Seek", "Jobs Marketplace.", "Growth"),
            "360": ("Life360", "Family Safety App.", "Growth"),

            # --- GROWTH: Emerging Leaders ---
            "DRO": ("DroneShield", "Counter-Drone Defense.", "Growth"),
            "WEB": ("Webjet", "Online Travel.", "Growth"),
            "FLT": ("Flight Centre", "Travel Agency.", "Growth"),
            "LOV": ("Lovisa", "Fast Fashion Jewelry.", "Growth"),
            "TPW": ("Temple & Webster", "Online Furniture.", "Growth"),
            "GNP": ("GenusPlus", "Power Infrastructure.", "Growth"),
            "CVL": ("Civmec", "Engineering/Defense.", "Growth"),
            "TLX": ("Telix Pharm", "Radiopharmaceuticals.", "Growth"),
            "PNV": ("PolyNovo", "Wound Care.", "Growth"),
            "NAN": ("Nanosonics", "Infection Control.", "Growth"),

            # --- SPEC: Strategic Minerals / Explorers ---
            "LTR": ("Liontown", "Lithium Developer.", "Spec"),
            "VUL": ("Vulcan Energy", "Zero Carbon Lithium.", "Spec"),
            "SYR": ("Syrah", "Graphite.", "Spec"),
            "CXO": ("Core Lithium", "Lithium Producer (Volatile).", "Spec"),
            "LKE": ("Lake Resources", "Lithium Brine.", "Spec"),
            "CHN": ("Chalice Mining", "Nickel/PGE Discovery.", "Spec"),
            # DEG Removed due to download errors
            "GL1": ("Global Lithium", "Lithium Exploration.", "Spec"),
            "BOE": ("Boss Energy", "Uranium.", "Growth"),
            "PDN": ("Paladin Energy", "Uranium.", "Growth"),
            "DYL": ("Deep Yellow", "Uranium.", "Spec"),

            # --- SPEC: Small Caps / Degens ---
            "ZIP": ("Zip Co", "BNPL.", "Spec"),
            "BRN": ("BrainChip", "Neuromorphic AI.", "Spec"),
            "MSB": ("Mesoblast", "Stem Cells.", "Spec"),
            "IMM": ("Immuron", "Biotech.", "Spec"),
            "BUB": ("Bubs Aust", "Infant Formula.", "Spec"),
            "88E": ("88 Energy", "Oil Explorer.", "Spec"),
            "MP1": ("Megaport", "Network as a Service.", "Spec"),
            "EOS": ("Electro Optic", "Defense Systems.", "Spec"),
        },
    },
    "USA": {
        "name": "United States (Wall St)",
        "tz": "America/New_York",
        "currency": "U$",
        "suffix": "",
        "tickers": {
            # --- CORE: Mag 7 & Mega Tech ---
            "NVDA": ("NVIDIA", "AI Hardware King.", "Core"),
            "MSFT": ("Microsoft", "Cloud & AI.", "Core"),
            "AAPL": ("Apple", "Consumer Tech.", "Core"),
            "AMZN": ("Amazon", "E-comm & Cloud.", "Core"),
            "GOOG": ("Alphabet", "Search & Data.", "Core"),
            "META": ("Meta", "Social & Ads.", "Core"),
            "TSLA": ("Tesla", "EV & Robotics.", "Growth"),
            
            # --- CORE: Buffett / Dividend Aristocrats ---
            "BRK.B": ("Berkshire", "Conglomerate.", "Core"),
            "JNJ":   ("J&J", "Healthcare.", "Core"),
            "PG":    ("P&G", "Staples.", "Core"),
            "KO":    ("Coca-Cola", "Beverages.", "Core"),
            "PEP":   ("PepsiCo", "Snacks/Bev.", "Core"),
            "MCD":   ("McDonalds", "Fast Food.", "Core"),
            "WMT":   ("Walmart", "Retail.", "Core"),
            "HD":    ("Home Depot", "Home Imp.", "Core"),
            "JPM":   ("JPMorgan", "Banking.", "Core"),
            "V":     ("Visa", "Payments.", "Core"),
            "MA":    ("Mastercard", "Payments.", "Core"),
            
            # --- CORE: Healthcare & Pharma ---
            "LLY":  ("Eli Lilly", "Weight Loss/Pharma.", "Core"),
            "NVO":  ("Novo Nordisk", "Weight Loss/Pharma.", "Core"),
            "UNH":  ("UnitedHealth", "Insurance.", "Core"),
            "MRK":  ("Merck", "Oncology.", "Core"),
            "PFE":  ("Pfizer", "Pharma.", "Core"),
            "ABBV": ("AbbVie", "Pharma.", "Core"),
            "VRTX": ("Vertex", "Biotech.", "Core"),

            # --- CORE: Old Tech / Semis ---
            "AVGO": ("Broadcom", "Networking/AI.", "Core"),
            "ORCL": ("Oracle", "Cloud/DB.", "Core"),
            "CSCO": ("Cisco", "Networking.", "Core"),
            "ADBE": ("Adobe", "Creative Software.", "Core"),
            "TXN":  ("Texas Inst", "Analog Chips.", "Core"),
            "AMD":  ("AMD", "CPU/GPU.", "Growth"),
            "INTC": ("Intel", "Chips Turnaround.", "Core"),
            "TSM":  ("TSMC", "Foundry.", "Core"),

            # --- GROWTH: Cloud / SaaS / Data ---
            "PLTR": ("Palantir", "AI Data Ops.", "Growth"),
            "SNOW": ("Snowflake", "Data Cloud.", "Growth"),
            "DDOG": ("Datadog", "Observability.", "Growth"),
            "MDB":  ("MongoDB", "NoSQL DB.", "Growth"),
            "CRM":  ("Salesforce", "CRM.", "Growth"),
            "NOW":  ("ServiceNow", "Workflow.", "Core"),
            "PANW": ("Palo Alto", "Cybersec.", "Core"),
            "CRWD": ("CrowdStrike", "Cybersec.", "Growth"),
            "ZS":   ("Zscaler", "Security Cloud.", "Growth"),
            "NET":  ("Cloudflare", "Internet Infra.", "Growth"),
            "SHOP": ("Shopify", "E-commerce.", "Growth"),
            "SQ":   ("Block", "Fintech.", "Growth"),
            "HUBS": ("HubSpot", "Marketing SaaS.", "Growth"),
            "TEAM": ("Atlassian", "Collaboration.", "Growth"),
            
            # --- GROWTH: Consumer & Disruption ---
            "ABNB": ("Airbnb", "Travel.", "Growth"),
            "UBER": ("Uber", "Mobility.", "Growth"),
            "DASH": ("DoorDash", "Delivery.", "Growth"),
            "BKNG": ("Booking", "Travel.", "Core"),
            "NFLX": ("Netflix", "Streaming.", "Growth"),
            "SPOT": ("Spotify", "Audio.", "Growth"),
            "COIN": ("Coinbase", "Crypto Exchange.", "Growth"),
            "HOOD": ("Robinhood", "Trading.", "Spec"),
            "DKNG": ("DraftKings", "Betting.", "Spec"),

            # --- SPEC: EV / Clean Tech / Hype ---
            "RIVN": ("Rivian", "EV Trucks.", "Spec"),
            "LCID": ("Lucid", "Luxury EV.", "Spec"),
            "PLUG": ("Plug Power", "Hydrogen.", "Spec"),
            "QS":   ("QuantumScape", "Solid State Battery.", "Spec"),
            "ENVX": ("Enovix", "Battery Tech.", "Spec"),
            "SOUN": ("SoundHound", "Voice AI.", "Spec"),
            "AI":   ("C3.ai", "Enterprise AI.", "Spec"),
            "IONQ": ("IonQ", "Quantum Computing.", "Spec"),

            # --- SPEC: Meme & Crypto ---
            "GME":  ("GameStop", "Retailer/Meme.", "Spec"),
            "AMC":  ("AMC", "Cinema/Meme.", "Spec"),
            "MSTR": ("MicroStrategy", "Bitcoin Proxy.", "Spec"),
            "MARA": ("Marathon", "Bitcoin Mining.", "Spec"),
            "RIOT": ("Riot", "Bitcoin Mining.", "Spec"),
            "CLSK": ("CleanSpark", "Bitcoin Mining.", "Spec"),
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
        df = yf.download(symbol, period=f"{FETCH_DAYS}d", interval="1d", auto_adjust=False, progress=False, group_by="column", prepost=False)
        
        if df is None or df.empty:
            df = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False, group_by="column", prepost=False)

        if df is None or df.empty: return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns.get_level_values(-1):
                df = df.xs(symbol, axis=1, level=-1, drop_level=True)
            else:
                df.columns = df.columns.get_level_values(0)

        col_map = {}
        for col in df.columns:
            c = str(col).lower()
            if c == "open": col_map[col] = "Open"
            elif c == "high": col_map[col] = "High"
            elif c == "low": col_map[col] = "Low"
            elif c in ("close", "price"): col_map[col] = "Close"
            elif c in ("adj close", "adj_close", "adjclose", "adjusted close"): col_map[col] = "AdjClose"
            elif "volume" in c: col_map[col] = "Volume"
        
        df = df.rename(columns=col_map)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in needed): return pd.DataFrame()

        keep = needed + (["AdjClose"] if "AdjClose" in df.columns else [])
        df = df[keep].reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df["Date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")

        market_tz = zoneinfo.ZoneInfo(tz_name)
        now_mkt = datetime.now(market_tz)
        last_date_mkt = df["Date"].dt.tz_convert(market_tz).dt.date.max()

        if APPEND_INTRADAY_BAR and (now_mkt.time() >= time(10, 0)) and (last_date_mkt < now_mkt.date()):
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

# ---------------- NEW: Dynamic Fundamental Engine (Refined V6.2) ----------------

def fetch_dynamic_fundamentals(symbol: str, category: str):
    """
    Evaluates companies based on:
    1. Core : Buffett/Piotroski-style quality (ROE, margins, balance sheet, cash conversion)
    2. Growth: Rule of 40 + Mohanram-style intangibles + survival buffer (runway)
    3. Spec : Runway, lifestyle vs work, cash floor (survival)
    """
    try:
        tick = yf.Ticker(symbol)
        # --- Safely get info dict ---
        try: info = tick.info
        except Exception: info = {}
        if not isinstance(info, dict): info = {}

        # --- Safely get statements ---
        try: bs, is_, cf = tick.balance_sheet, tick.income_stmt, tick.cashflow
        except Exception: bs, is_, cf = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        def get_item(df, item_names, idx: int = 0, default: float = 0.0) -> float:
            if df is None or df.empty: return default
            for name in item_names:
                if name in df.index:
                    try:
                        row = df.loc[name]
                        if hasattr(row, "iloc"): return float(row.iloc[idx])
                        return float(row)
                    except Exception: return default
            return default

        # --- Parse profit margin & debt/equity once ---
        raw_profit_margin = info.get("profitMargins") or 0.0
        try: profit_margin = float(raw_profit_margin)
        except Exception: profit_margin = 0.0

        raw_debt_eq = info.get("debtToEquity") or 0.0
        try: debt_to_equity_val = float(raw_debt_eq)
        except Exception: debt_to_equity_val = 0.0

        score = 0.0
        tier = "Neutral"
        key_metric_str = ""

        # --- Common datapoints ---
        cash    = get_item(bs, ["Cash And Cash Equivalents", "Cash And Cash Equivalents And Short Term Investments", "Cash Financial"])
        lt_debt = get_item(bs, ["Long Term Debt", "Total Long Term Debt"])
        assets  = get_item(bs, ["Total Assets"])
        tot_rev = get_item(is_, ["Total Revenue", "Operating Revenue"])
        net_inc = get_item(is_, ["Net Income", "Net Income Common Stockholders"])
        ebitda  = get_item(is_, ["EBITDA", "Normalized EBITDA"])
        ocf     = get_item(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        equity  = get_item(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest"])

        # --- Burn / runway shared by Growth & Spec ---
        burn_annual = 0.0
        runway_months = None
        if (ocf < 0) or (net_inc < 0):
            if (ocf < 0) and (net_inc < 0): base_loss = min(ocf, net_inc)
            elif ocf < 0: base_loss = ocf
            else: base_loss = net_inc
            burn_annual = abs(base_loss)
            if burn_annual > 0 and cash > 0:
                runway_months = cash / (burn_annual / 12.0)

        # =================== MODE 1: CORE (BUFFETT / VALUE) ===================
        if category == "Core":
            # 1) ROE (weight ~3)
            if equity > 0:
                roe = net_inc / equity
                if roe > 0.15: score += 3
                elif roe > 0.10: score += 1.5
            else: roe = 0.0

            # 2) Profit margins (weight ~2)
            marg = profit_margin
            if marg > 0.15: score += 2
            elif marg > 0.08: score += 1

            # 3) Balance sheet: debt vs cash (weight ~3)
            debt_eq = debt_to_equity_val
            if debt_eq > 50: debt_eq = debt_eq / 100.0

            if cash > lt_debt and lt_debt > 0: score += 3
            elif debt_eq < 0.5: score += 2
            elif debt_eq < 1.0: score += 1

            # 4) Earnings quality (weight ~2)
            if ocf > net_inc > 0: score += 2

            tier = "Fortress" if score >= 7 else ("Stable" if score >= 4 else "Weak")
            key_metric_str = f"ROE: {roe*100:.1f}%"

        # =================== MODE 2: GROWTH (VC / MOHANRAM) ===================
        elif category == "Growth":
            # 1) Rule of 40 (weight ~4)
            rev_prev = get_item(is_, ["Total Revenue", "Operating Revenue"], idx=1)
            rev_growth = ((tot_rev - rev_prev) / rev_prev) * 100.0 if rev_prev > 0 else 0.0
            ebitda_marg = (ebitda / tot_rev) * 100.0 if tot_rev > 0 else 0.0
            rule_40 = rev_growth + ebitda_marg

            if rule_40 > 50: score += 4
            elif rule_40 > 40: score += 3
            elif rule_40 > 20: score += 1

            # 2) Mohanram-style intangibles: R&D intensity (weight ~2)
            rnd = get_item(is_, ["Research And Development"])
            sga = get_item(is_, ["Selling General And Administration", "General And Administrative Expense"])
            rnd_intensity = (rnd / assets) if assets > 0 else 0.0
            if rnd_intensity > 0.10: score += 2
            elif rnd_intensity > 0.05: score += 1

            # 3) Earnings quality: CFO > Net Income (weight ~1)
            if ocf > net_inc and ocf > 0: score += 1

            # 4) "Magic Number" proxy: rev added per $ of SGA (weight ~2)
            if sga > 0:
                efficiency = (tot_rev - rev_prev) / sga
                if efficiency > 1.0: score += 2
                elif efficiency > 0.7: score += 1

            # 5) Gross margin scalability (weight ~2)
            gp = get_item(is_, ["Gross Profit"])
            gm = gp / tot_rev if tot_rev > 0 else 0.0
            if gm > 0.60: score += 2
            elif gm > 0.40: score += 1

            # 6) Survival buffer: runway (penalty/bonus)
            if runway_months is not None:
                if runway_months < 6: score -= 2
                elif runway_months < 12: score -= 1
                elif runway_months > 24: score += 1

            tier = "Hyper-Growth" if score >= 7 else ("Scalable" if score >= 4 else "Burner")
            key_metric_str = f"Rule40: {rule_40:.0f}"

        # =================== MODE 3: SPEC (SURVIVAL / LIFESTYLE) ===================
        else:  # Spec
            # 1) Runway (weight ~4)
            if runway_months is not None or burn_annual > 0:
                rm = runway_months if runway_months is not None else 0.0
                if rm > 18: score += 4
                elif rm > 12: score += 3
                elif rm > 6: score += 1
                else: score -= 2  # imminent death

            # 2) Lifestyle vs work (weight ~3)
            sga = get_item(is_, ["Selling General And Administration", "General And Administrative Expense"])
            rnd = get_item(is_, ["Research And Development"])
            capex = abs(get_item(cf, ["Capital Expenditure", "Purchase Of PPE"]))
            work_spend = max(rnd, capex)

            if work_spend == 0 and sga == 0: pass
            elif work_spend > sga * 1.5: score += 3
            elif work_spend > sga: score += 2
            elif sga > work_spend * 2: score -= 2  # lifestyle company

            # 3) Cash floor (weight ~3)
            if cash > 20_000_000: score += 3
            elif cash > 5_000_000: score += 1

            # Key metric for card/table
            if runway_months is not None:
                rm = runway_months
                runway_str = f"{rm:.1f}m" if rm < 120 else ">10yr"
            else: runway_str = "n/a"

            tier = "Funded" if score >= 7 else ("Alive" if score >= 4 else "Zombie")
            key_metric_str = f"Runway: {runway_str}"
            if tot_rev < 1_000_000:
                key_metric_str = f"Cash: {cash/1e6:.1f}M / {runway_str}"

        # Clamp and return
        score = max(0.0, min(10.0, score))

        return {
            "score": round(score, 1),
            "tier": tier,
            "category_mode": category,
            "key_metric": key_metric_str,
            "roe": net_inc / equity if equity > 0 else 0.0,
            "margin": profit_margin,
            "debteq": debt_to_equity_val,
            "cash": cash,
            "runway_months": runway_months or 0.0,
            "burn_annual": burn_annual,
        }

    except Exception:
        return {"score": 0.0, "tier": "Error", "category_mode": category, "key_metric": "-", "roe": 0.0, "margin": 0.0, "debteq": 0.0, "cash": 0.0, "runway_months": 0.0, "burn_annual": 0.0}

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicator stack used by the dashboard.

    Key design choices:
    - Signals are computed on a single "Price" series.
    - If AdjClose is present (yfinance) we can use it to avoid split/dividend artifacts in long lookbacks.
    - Breakout highs/lows are shifted by 1 bar to avoid the "Close > rolling(max including today)" bug.
    """
    x = df.copy().sort_values("Date").reset_index(drop=True)

    # --- Price series for indicator calculations ---
    if USE_ADJUSTED_FOR_INDICATORS and ("AdjClose" in x.columns):
        # AdjClose is typically aligned to last close (factor ~1 on the latest bar)
        x["Price"] = pd.to_numeric(x["AdjClose"], errors="coerce")
        # Approximate adjusted OHLC using the same adjustment factor as close
        with np.errstate(divide="ignore", invalid="ignore"):
            adj_factor = x["Price"] / pd.to_numeric(x["Close"], errors="coerce")
        adj_factor = adj_factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        x["OpenI"] = pd.to_numeric(x["Open"], errors="coerce") * adj_factor
        x["HighI"] = pd.to_numeric(x["High"], errors="coerce") * adj_factor
        x["LowI"]  = pd.to_numeric(x["Low"],  errors="coerce") * adj_factor
    else:
        x["Price"] = pd.to_numeric(x["Close"], errors="coerce")
        x["OpenI"] = pd.to_numeric(x["Open"], errors="coerce")
        x["HighI"] = pd.to_numeric(x["High"], errors="coerce")
        x["LowI"]  = pd.to_numeric(x["Low"],  errors="coerce")

    # --- Moving averages ---
    x["SMA20"]  = x["Price"].rolling(20).mean()
    x["SMA50"]  = x["Price"].rolling(50).mean()
    x["SMA200"] = x["Price"].rolling(200).mean()
    x["EMA21"]  = x["Price"].ewm(span=21, adjust=False).mean()

    # Slope proxies (useful for Stage classification)
    x["SMA200_Slope"]    = x["SMA200"].diff(20)
    x["SMA200_Slope_%"]  = (x["SMA200_Slope"] / x["SMA200"]).replace([np.inf, -np.inf], np.nan) * 100.0
    x["SMA50_Slope_%"]   = (x["SMA50"].diff(10) / x["SMA50"]).replace([np.inf, -np.inf], np.nan) * 100.0

    # --- Breakout reference levels (shifted by 1 bar: we compare to prior highs) ---
    x["High20"]  = x["HighI"].shift(1).rolling(20).max()
    x["Low20"]   = x["LowI"].shift(1).rolling(20).min()
    x["High52W"] = x["HighI"].shift(1).rolling(252).max()
    x["Low52W"]  = x["LowI"].shift(1).rolling(252).min()

    # --- Volume baseline ---
    x["Vol20"] = pd.to_numeric(x["Volume"], errors="coerce").rolling(20).mean()

    # --- RSI(14) on Price ---
    chg = x["Price"].diff()
    gains = chg.clip(lower=0).rolling(14).mean()
    losses = (-chg).clip(lower=0).rolling(14).mean()
    RS = gains / losses
    x["RSI14"] = 100 - (100 / (1 + RS))

    # --- Distances ---
    x["Dist_to_52W_High_%"] = (x["Price"] / x["High52W"] - 1) * 100.0
    x["Dist_to_SMA200_%"]   = (x["Price"] / x["SMA200"]  - 1) * 100.0

    # --- ATR(14) on adjusted OHLC (HighI/LowI) vs prior Price ---
    x["H-L"] = x["HighI"] - x["LowI"]
    x["H-C"] = (x["HighI"] - x["Price"].shift(1)).abs()
    x["L-C"] = (x["LowI"]  - x["Price"].shift(1)).abs()
    x["TR"]  = x[["H-L", "H-C", "L-C"]].max(axis=1)
    x["ATR14"]   = x["TR"].rolling(14).mean()
    x["ATR14_%"] = (x["ATR14"] / x["Price"]).replace([np.inf, -np.inf], np.nan) * 100.0

    return x

def label_row(r: pd.Series) -> str:
    """
    Produces a single high-level action label for the latest bar.

    Philosophy:
    - AVOID  = chart is broken (damage control regime)
    - TRIM   = uptrend but stretched (profit-taking / tighten stops)
    - BUY    = breakout continuation in a healthy uptrend
    - DCA    = pullback-to-200DMA zone in a healthy uptrend
    - HOLD   = uptrend intact, but no fresh edge today
    - WATCH  = no-man's land / base-building / too early
    """
    price  = float(r.get("Price", np.nan))
    sma200 = float(r.get("SMA200", np.nan))
    sma50  = float(r.get("SMA50", np.nan))
    rsi    = float(r.get("RSI14", np.nan))
    high20 = float(r.get("High20", np.nan))

    vol   = float(r.get("Volume", np.nan))
    vol20 = float(r.get("Vol20", np.nan))

    sma200_slope_pct = float(r.get("SMA200_Slope_%", 0.0))

    # Basic sanity
    if not (np.isfinite(price) and np.isfinite(sma200) and np.isfinite(sma50) and np.isfinite(rsi)):
        return "WATCH"

    death_cross = sma50 < sma200

    # Trend context
    trend_up = (price > sma200) and (sma50 > sma200) and (sma200_slope_pct > 0)

    # Damage control: below 200DMA, especially with death cross
    break_pct = float(RULES.get("risk", {}).get("sma200_break_pct", 0.03))
    broke_200 = price < sma200 * (1.0 - break_pct)

    if RULES.get("avoid", {}).get("death_cross", True) and death_cross and (price < sma200 or broke_200):
        return "AVOID"

    # Profit-taking regime
    if RULES.get("trim", {}).get("enabled", True) and is_euphoria(r):
        return "TRIM"

    # BUY: breakout above prior 20-day high in a Stage-2 uptrend
    buy_ok = trend_up and np.isfinite(high20) and (price > high20) and (RULES["buy"]["rsi_min"] <= rsi <= RULES["buy"]["rsi_max"])
    if buy_ok and np.isfinite(vol) and np.isfinite(vol20) and vol20 > 0:
        buy_ok = vol >= RULES["buy"].get("vol_mult", 1.0) * vol20

    # DCA: dip near 200DMA in a healthy regime (no death-cross)
    allow_below = float(RULES["dca"].get("allow_below_pct", 0.02))
    dca_ok = (
        (price >= sma200 * (1.0 - allow_below)) and
        (price <= sma200 * (1.0 + RULES["dca"]["sma200_proximity"])) and
        (rsi <= RULES["dca"]["rsi_max"]) and
        (not death_cross)
    )

    if buy_ok: return "BUY"
    if dca_ok: return "DCA"
    if trend_up: return "HOLD"
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

def litmus_signal_stats(ind: pd.DataFrame, horizons=(5, 20, 60), lookback_bars=252 * 6) -> dict:
    """
    Lightweight "evidence check" for the dashboard.

    Computes forward returns after signal days to answer:
      "Historically, when this logic said BUY (or DCA), what tended to happen next?"

    Notes:
    - Uses close-to-close forward returns on the indicator Price series.
    - This is NOT a full trading backtest (no slippage/commissions/execution modelling).
    - Intended as a sanity check to spot dead signals / obvious anti-signals.
    """
    if ind is None or ind.empty:
        return {}

    df = ind.copy()

    # Require the minimum columns needed by label_row + forward returns
    required = ["Price", "SMA200", "SMA50", "RSI14", "High20", "Vol20", "Volume", "Dist_to_SMA200_%", "Dist_to_52W_High_%"]
    have = [c for c in required if c in df.columns]
    df = df.dropna(subset=have)
    if df.empty:
        return {}

    df = df.tail(int(lookback_bars)).copy()
    try:
        df["SignalLit"] = df.apply(label_row, axis=1)
    except Exception:
        return {}

    price = df["Price"]
    out = {}

    for s in ("BUY", "DCA", "TRIM"):
        mask = df["SignalLit"] == s
        out[f"n_{s.lower()}"] = int(mask.sum())
        for h in horizons:
            fwd = price.shift(-int(h)) / price - 1.0
            vals = fwd[mask].dropna()
            if len(vals) >= 10:
                out[f"{s}_{h}d_med"] = float(vals.median() * 100.0)
                out[f"{s}_{h}d_hit"] = float((vals > 0).mean() * 100.0)
            else:
                out[f"{s}_{h}d_med"] = np.nan
                out[f"{s}_{h}d_hit"] = np.nan

    return out

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
    """Simple bull-flag detector (meant as a *setup*, not a guarantee)."""
    if len(ind) < 60:
        return False, {}

    # Prefer adjusted indicator series if present
    look = ind.tail(40)
    look_p = look.get("Price", look["Close"])

    impulse = (look_p.max() / look_p.min() - 1) * 100
    if not np.isfinite(impulse) or impulse < 12:
        return False, {}

    win = 14
    tail = ind.tail(max(win, 8)).copy()
    t_price = tail.get("Price", tail["Close"])
    t_high  = tail.get("HighI", tail["High"])
    t_low   = tail.get("LowI", tail["Low"])

    x = np.arange(len(tail))
    hi = np.polyfit(x, t_high.values, 1)
    lo = np.polyfit(x, t_low.values, 1)

    slope_pct = (hi[0] / t_price.iloc[-1]) * 100
    ch = np.polyval(hi, x[-1]) - np.polyval(lo, x[-1])

    # Channel must be relatively tight vs the impulse move
    tight = ch <= max(0.4 * (look_p.max() - look_p.min()), 0.02 * t_price.iloc[-1])

    # Flag should drift sideways / gently down (not a steep collapse)
    gentle = (-0.006 <= slope_pct <= 0.002)

    return bool(tight and gentle), {"hi": hi.tolist(), "lo": lo.tolist(), "win": win}

def pattern_bias(name: str) -> str:
    if name in ("Double Bottom", "Inverse H&S", "Ascending Triangle", "Bull Flag"): return "bullish"
    if name in ("Double Top", "Head & Shoulders", "Descending Triangle"): return "bearish"
    return "neutral"

def breakout_ready(ind: pd.DataFrame, pat: dict, rules: dict, flag_info: dict = None):
    """
    Generic breakout/breakdown detector for dashboard use (not a backtest execution engine).

    We look for:
    - Price clears a pattern trigger level by a small buffer and ATR multiple
    - Volume confirms vs 20-day average

    Returns (ready: bool, info: dict)
      info["direction"] is "bull" or "bear"
    """
    if ind is None or ind.empty or not pat:
        return False, {}

    last = ind.iloc[-1]
    price = float(last.get("Price", last.get("Close", np.nan)))
    atr   = float(last.get("ATR14", np.nan))
    vol   = float(last.get("Volume", np.nan))
    vol20 = float(last.get("Vol20", np.nan))

    if not (np.isfinite(price) and np.isfinite(atr)):
        return False, {}

    name = str(pat.get("name", ""))
    levels = pat.get("levels", {}) or {}
    bias = pattern_bias(name)  # bullish / bearish / neutral

    # --- Determine trigger level ---
    level = np.nan
    level_type = ""

    # Bull Flag: trigger is the upper channel value on the last bar
    if name == "Bull Flag" and flag_info:
        try:
            win = int(flag_info.get("win", 14))
            t2 = ind.tail(max(win, 8))
            x = np.arange(len(t2))
            hi_line = float(np.polyval(flag_info.get("hi"), x[-1]))
            level = hi_line
            level_type = "flag_upper"
        except Exception:
            level = np.nan

    # Inverse H&S and H&S have 2 neckline points; compress into a single trigger
    if not np.isfinite(level) and name in ("Inverse H&S", "Head & Shoulders"):
        nl = float(levels.get("neck_left", np.nan))
        nr = float(levels.get("neck_right", np.nan))
        if np.isfinite(nl) and np.isfinite(nr):
            level = min(nl, nr) if bias == "bullish" else max(nl, nr)
            level_type = "neckline"

    # Default mapping based on bias
    if not np.isfinite(level):
        if bias == "bullish":
            for k in ("resistance", "neckline", "trigger", "ceiling"):
                v = levels.get(k, np.nan)
                if np.isfinite(v):
                    level = float(v); level_type = k; break
        elif bias == "bearish":
            for k in ("support", "neckline", "trigger", "floor"):
                v = levels.get(k, np.nan)
                if np.isfinite(v):
                    level = float(v); level_type = k; break
        else:
            return False, {}

    if not np.isfinite(level):
        return False, {}

    # --- Confirmation rules ---
    buffer_pct = float(rules.get("buffer_pct", 0.003))
    atr_mult   = float(rules.get("atr_mult", 0.50))
    vol_mult   = float(rules.get("vol_mult", 1.30))

    ok_vol = True
    if np.isfinite(vol) and np.isfinite(vol20) and vol20 > 0:
        ok_vol = vol >= vol_mult * vol20

    if bias == "bullish":
        ok_price = (price >= level * (1.0 + buffer_pct)) and (price >= level + atr_mult * atr)
        stop = price - atr
        direction = "bull"
    else:
        ok_price = (price <= level * (1.0 - buffer_pct)) and (price <= level - atr_mult * atr)
        stop = price + atr
        direction = "bear"

    ready = bool(ok_price and ok_vol)
    return ready, {
        "direction": direction,
        "pattern": name,
        "level": round(level, 4),
        "level_type": level_type,
        "atr": round(atr, 4),
        "stop": round(stop, 4),
        "vol_mult": vol_mult,
    }

# ---------------- Commentary Engine 2.2 (Setup Aware) ----------------

def is_euphoria(r):
    # Global helper required for both commentary and HTML rendering
    d200 = r.get("Dist_to_SMA200_%", 0.0)
    d52  = r.get("Dist_to_52W_High_%", 0.0)
    rsi  = r.get("RSI14", 50.0)
    return (d52 > -3.5) and (d200 > 40.0) and (rsi >= 70.0)

def comment_for_row(r: pd.Series):
    """
    Returns (summary_html, playbook_html)

    The goal is to be *directive*:
    - What to do if you DON'T own it
    - What to do if you DO own it
    - Where you're wrong (invalidations / stops)
    """
    # --- Unpack metrics safely ---
    sig   = str(r.get("Signal", "WATCH")).upper()
    cat   = str(r.get("Category", "Core"))
    score = float(r.get("Fundy_Score", 0.0))
    tier  = str(r.get("Fundy_Tier", "Neutral"))

    price = float(r.get("Price", r.get("LastClose", np.nan)))
    last_close = float(r.get("LastClose", price))

    rsi   = float(r.get("RSI14", np.nan))
    d200  = float(r.get("Dist_to_SMA200_%", 0.0))
    d52   = float(r.get("Dist_to_52W_High_%", 0.0))

    sma200 = float(r.get("SMA200", np.nan))
    sma50  = float(r.get("SMA50", np.nan))
    ema21  = float(r.get("EMA21", np.nan))
    atr    = float(r.get("ATR14", np.nan))
    atr_pct = float(r.get("ATR14_%", np.nan))

    high20 = float(r.get("High20", np.nan))
    high52 = float(r.get("High52W", np.nan))

    slope200 = float(r.get("SMA200_Slope_%", 0.0))
    death_cross = (np.isfinite(sma50) and np.isfinite(sma200) and sma50 < sma200)

    # Setups / patterns
    pat_name   = str(r.get("_pattern_name", "") or "")
    pat_status = str(r.get("_pattern_status", "") or "")
    pat_conf   = float(r.get("_pattern_conf", 0.0) or 0.0)
    pat_align  = str(r.get("_pattern_align", "") or "")

    is_autodca = bool(r.get("AutoDCA_Flag", False))
    is_breakout = bool(r.get("BreakoutReady", False))
    brk_lvl = float(r.get("Breakout_Level", np.nan))
    brk_dir = str(r.get("Breakout_Direction", "") or "")

    fundy = r.get("Fundy", {}) or {}
    key_metric = str(fundy.get("key_metric", "-"))
    runway = float(fundy.get("runway_months", 0.0) or 0.0)

    # --- Stage / regime classification ---
    stage = "Stage 1 (Base / Range)"
    if np.isfinite(d200) and np.isfinite(slope200):
        if (d200 > 5.0) and (slope200 > 0):
            stage = "Stage 2 (Uptrend)"
        elif (d200 < -5.0) and (slope200 < 0):
            stage = "Stage 4 (Downtrend)"
        elif (d200 < -2.0) and (slope200 > 0):
            stage = "Stage 3 (Transition / Topping)"
        elif (d200 > 2.0) and (slope200 < 0):
            stage = "Stage 3 (Transition / Topping)"

    # --- Risk levels (ATR based) ---
    atr_stop_mult  = float(RULES.get("risk", {}).get("atr_stop_mult", 2.0))
    atr_trail_mult = float(RULES.get("risk", {}).get("atr_trail_mult", 3.0))

    hard_stop = np.nan
    trail_stop = np.nan
    if np.isfinite(price) and np.isfinite(atr) and atr > 0:
        hard_stop = price - atr_stop_mult * atr
        trail_stop = price - atr_trail_mult * atr
        if np.isfinite(sma50):
            trail_stop = max(trail_stop, sma50)

    # --- Entry triggers ---
    entry_lvl = np.nan
    entry_reason = ""
    if is_breakout and np.isfinite(brk_lvl):
        entry_lvl = brk_lvl
        entry_reason = "pattern trigger"
    elif np.isfinite(high20):
        entry_lvl = high20
        entry_reason = "prior 20D high"
    elif np.isfinite(high52):
        entry_lvl = high52
        entry_reason = "prior 52W high"

    # --- Emoji / summary line ---
    emoji = {"BUY":"🚀","DCA":"🛒","HOLD":"🧘","TRIM":"🥂","WATCH":"👀","AVOID":"🧊"}.get(sig, "⚖️")

    # Spec sanity: no DCA on zombies
    is_zombie = (cat == "Spec" and runway > 0 and runway < 6.0)

    # --- Summary text ---
    fundy_line = f"{cat} • {score:.1f}/10 {tier} • {key_metric}"
    if cat == "Spec":
        fundy_line = f"{cat} • {score:.1f}/10 {tier} • {key_metric}"

    quick_flags = []
    if is_autodca: quick_flags.append("Auto-DCA setup")
    if is_breakout: quick_flags.append(f"Breakout {brk_dir or ''}".strip())
    if pat_name: quick_flags.append(f"{pat_name} ({pat_status}, {pat_conf:.2f})")
    if is_zombie: quick_flags.append("⚠️ Dilution risk")

    flags_html = (" • " + " • ".join(quick_flags)) if quick_flags else ""

    # Main directive headline per signal
    if sig == "BUY":
        headline = "Breakout / continuation candidate in a healthy uptrend."
    elif sig == "DCA":
        headline = "Dip zone near the 200DMA — accumulation only if this is a business you’d happily hold."
    elif sig == "HOLD":
        headline = "Trend intact — hold if owned; wait for a better entry if not."
    elif sig == "TRIM":
        headline = "Extended / euphoric — take some profit and tighten risk."
    elif sig == "AVOID":
        headline = "Damage-control regime — avoid new buys; reduce risk if holding."
    else:
        headline = "No clear edge today — keep on a watchlist with alerts."

    summary = f"<b>{emoji} {sig}:</b> {headline}<br><span style='color:var(--text-muted)'>{stage} • {fundy_line}{flags_html}</span>"

    # --- Playbook (actionable steps) ---
    def _fmt(x):
        return "-" if not np.isfinite(x) else f"{x:.2f}"

    def _fmt_pct(x):
        return "-" if not np.isfinite(x) else f"{x:.1f}%"

    bullets = []

    # Common context bullets
    if np.isfinite(rsi):
        bullets.append(f"<b>Momentum:</b> RSI {rsi:.0f} • Δ vs 200DMA {_fmt_pct(d200)} • Δ vs 52W High {_fmt_pct(d52)}")
    if np.isfinite(atr_pct):
        bullets.append(f"<b>Volatility:</b> ATR(14) {_fmt_pct(atr_pct)} (bigger ATR = wider stops / smaller size)")

    # Signal-specific guidance
    if sig == "BUY":
        if np.isfinite(entry_lvl):
            bullets.append(f"<b>If you DON'T own:</b> Consider a starter position only on strength: close &gt; <span class='mono'>{_fmt(entry_lvl)}</span> ({entry_reason}) with volume. Avoid buying into the middle of a range.")
        else:
            bullets.append("<b>If you DON'T own:</b> Starter position only if price holds above the 50DMA and prints a new swing high.")
        bullets.append(f"<b>If you DO own:</b> Hold winners. Prefer adding on pullbacks to EMA21 (<span class='mono'>{_fmt(ema21)}</span>) rather than chasing green candles.")
        if np.isfinite(trail_stop):
            bullets.append(f"<b>Risk (trailing):</b> Consider a trail near <span class='mono'>{_fmt(trail_stop)}</span> (max(50DMA, {atr_trail_mult:.0f}×ATR stop)).")

    elif sig == "DCA":
        if cat == "Spec":
            bullets.append("<b>Spec warning:</b> This is not a DCA instrument. If you buy, treat it as a trade with a hard stop (no averaging down).")
        if np.isfinite(sma200):
            bullets.append(f"<b>If you DON'T own:</b> Scale-in plan: 3 tranches near the 200DMA (<span class='mono'>{_fmt(sma200)}</span>). Start small; add only if it stabilizes (higher lows).")
        bullets.append("<b>If you DO own:</b> Add only if your thesis is intact AND the chart holds 200DMA. If 200DMA breaks and stays broken → stop the bleeding (no heroic averaging).")
        if np.isfinite(hard_stop):
            bullets.append(f"<b>Risk (hard stop):</b> A simple line-in-the-sand is <span class='mono'>{_fmt(hard_stop)}</span> ({atr_stop_mult:.0f}×ATR below price).")

    elif sig == "HOLD":
        bullets.append("<b>If you DON'T own:</b> Patience. Wait for either (1) pullback to EMA21/50DMA or (2) breakout above prior highs.")
        bullets.append("<b>If you DO own:</b> Do nothing. Move your stop up as the 50DMA rises. Add only on constructive pullbacks (not strength).")
        if np.isfinite(trail_stop):
            bullets.append(f"<b>Risk:</b> Trail near <span class='mono'>{_fmt(trail_stop)}</span> to avoid giving back a full trend leg.")

    elif sig == "TRIM":
        bullets.append("<b>If you DON'T own:</b> Do not chase. Wait for a pullback to EMA21 / 50DMA, or a multi-week consolidation.")
        bullets.append("<b>If you DO own:</b> Consider trimming 20–50% into strength. Raise your stop (protect gains).")
        if np.isfinite(trail_stop):
            bullets.append(f"<b>Risk:</b> Tighten trail near <span class='mono'>{_fmt(trail_stop)}</span>. A sharp reversal from euphoria can be fast.")

    elif sig == "AVOID":
        if is_zombie:
            bullets.append(f"<b>Zombie risk:</b> Runway ≈ {runway:.1f} months. High probability of dilution / capital raise. Avoid.")
        bullets.append("<b>If you DON'T own:</b> Stay away until it reclaims 200DMA and the 50DMA turns back up.")
        bullets.append("<b>If you DO own:</b> Consider reducing or exiting. If you insist on holding, define a hard invalidation (no 'hope trades').")
        if np.isfinite(sma200):
            bullets.append(f"<b>Recovery trigger:</b> A first step is price closing back above 200DMA (<span class='mono'>{_fmt(sma200)}</span>) and holding for multiple weeks.")

    else:  # WATCH
        if np.isfinite(entry_lvl):
            bullets.append(f"<b>Alert:</b> Set an alert at <span class='mono'>{_fmt(entry_lvl)}</span> ({entry_reason}). If it breaks, reassess for a momentum entry.")
        if death_cross:
            bullets.append("<b>Trend warning:</b> Death-cross regime. Treat any rally as suspect until 50DMA recovers above 200DMA.")
        bullets.append("<b>Plan:</b> No position is a position. Wait for trend + setup alignment.")

    # Pattern alignment note
    if pat_name:
        bullets.append(f"<b>Structure:</b> {pat_name} • {pat_status} • conf {pat_conf:.2f} • <span class='mono'>{pat_align}</span>")

    # Breakout ready note
    if is_breakout and np.isfinite(brk_lvl):
        bullets.append(f"<b>Breakout trigger:</b> {brk_dir.upper() if brk_dir else ''} &gt; <span class='mono'>{_fmt(brk_lvl)}</span> (needs volume).")

    playbook = "<div>" + "<br>".join([f"• {b}" for b in bullets]) + "</div>"

    return summary, playbook

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
            tick = re.search(r"([A-Z0-9]{3,4})[_-]", fname)
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

        raw_sym = f"{t_key}{m_conf['suffix']}"
        # Yahoo quirks: BRK.B => BRK-B, etc.
        dl_sym = raw_sym.replace(".", "-") if (m_code == "USA" and "." in raw_sym) else raw_sym

        df = fetch_prices(dl_sym, m_conf["tz"])
        if df.empty:
            continue

        df["Ticker"] = t_key
        frames.append(df)

        fundy = fetch_dynamic_fundamentals(dl_sym, t_cat)

        # Indicators (daily bars only)
        ind = indicators(df).dropna(subset=["High20", "RSI14", "EMA21", "Vol20", "ATR14", "SMA200", "SMA50", "Price"])
        if ind.empty:
            continue

        last = ind.iloc[-1]
        sig = label_row(last)

        # --- Flag detection (bull flag) ---
        flag_flag, flag_det = detect_flag(ind)

        # --- Pattern detection ---
        pats_all = (
            detect_double_bottom(ind) +
            detect_double_top(ind) +
            detect_inverse_hs(ind) +
            detect_hs(ind) +
            detect_triangles(ind)
        )

        # Treat a detected Bull Flag as a "pattern-like" structure for alignment / breakout checks
        flag_pat = None
        if flag_flag:
            confirmed = False
            upper = np.nan
            try:
                win = int(flag_det.get("win", 14))
                t2 = ind.tail(max(win, 8))
                x = np.arange(len(t2))
                upper = float(np.polyval(flag_det.get("hi"), x[-1]))
                confirmed = bool(float(last.get("Price", last.get("Close"))) > upper)
            except Exception:
                confirmed = False
            conf = 0.55 + (0.25 if confirmed else 0.0)
            flag_pat = {
                "name": "Bull Flag",
                "status": "confirmed" if confirmed else "forming",
                "confidence": round(min(conf, 1.0), 2),
                "levels": {"resistance": round(float(upper), 4) if np.isfinite(upper) else np.nan},
                "lines": None,
            }
            pats_all = [flag_pat] + pats_all

        # Patterns for display (optionally confirmed-only)
        pats_display = pats_all
        if PATTERNS_CONFIRMED_ONLY:
            pats_display = [p for p in pats_all if p.get("status") == "confirmed"]

        # --- Breakout / breakdown readiness (use ALL patterns, not only confirmed) ---
        brk_ready, brk_info = False, {}
        # Priority order: flag/triangle/dbottom/ihs ... then bearish breakdown patterns
        for p in pats_all:
            ready, info = breakout_ready(ind, p, BREAKOUT_RULES, flag_det if (p.get("name") == "Bull Flag") else None)
            if ready:
                brk_ready, brk_info = True, info
                break

        # Auto-upgrade: bullish breakout => BUY, bearish breakdown => AVOID
        if AUTO_UPGRADE_BREAKOUT and brk_ready:
            if brk_info.get("direction") == "bull" and sig in ("WATCH", "HOLD", "DCA", "TRIM"):
                sig = "BUY"
            elif brk_info.get("direction") == "bear":
                sig = "AVOID"

        # --- Auto-DCA gate (gap reclaim) ---
        gate_flag, gate_det = auto_dca_gate(ind)
        if gate_flag and sig in ("WATCH", "HOLD"):
            sig = "DCA"

        # --- Pattern alignment ---
        pname = pats_display[0]["name"] if pats_display else (flag_pat["name"] if flag_pat else "")
        pbias = pattern_bias(pname)
        sig_str = str(sig).lower()

        # For alignment purposes, HOLD/TRIM are treated as "not fighting the tape"
        bullish_ok = sig_str in ["buy", "dca", "hold", "trim", "watch"]
        bearish_ok = sig_str in ["avoid", "watch"]

        is_aligned = (
            pbias == "neutral" or
            (pbias == "bullish" and bullish_ok) or
            (pbias == "bearish" and bearish_ok)
        )
        palign = "ALIGNED" if is_aligned else "CONFLICT"

        # --- Fundamental safety gates (prevents "trash BUY" on nice charts) ---
        if t_cat == "Core" and fundy["score"] < 4:
            sig = "AVOID"
        if t_cat == "Growth" and fundy["score"] < 3:
            sig = "AVOID"
        if t_cat == "Spec" and fundy["score"] < 3:
            sig = "AVOID"

        # --- Optional litmus stats (forward returns after signals) ---
        litmus = {}
        if ENABLE_LITMUS_STATS:
            try:
                litmus = litmus_signal_stats(
                    ind,
                    horizons=LITMUS_SIGNAL_HORIZONS,
                    lookback_bars=LITMUS_LOOKBACK_BARS,
                )
            except Exception:
                litmus = {}

        # Pick a primary visible pattern (if any)
        p0 = pats_display[0] if pats_display else (flag_pat if flag_pat else None)

        snaps.append({
            "Ticker": t_key,
            "Name": t_name,
            "Desc": t_desc,
            "Category": t_cat,

            "LastDate": last["Date"].strftime("%Y-%m-%d"),
            "LastClose": float(last.get("Close", np.nan)),
            "Price": float(last.get("Price", last.get("Close", np.nan))),

            "SMA50": float(last.get("SMA50", np.nan)),
            "SMA200": float(last.get("SMA200", np.nan)),
            "EMA21": float(last.get("EMA21", np.nan)),
            "SMA200_Slope_%": float(last.get("SMA200_Slope_%", 0.0)),

            "RSI14": float(last.get("RSI14", np.nan)),
            "ATR14": float(last.get("ATR14", np.nan)),
            "ATR14_%": float(last.get("ATR14_%", np.nan)),

            "High20": float(last.get("High20", np.nan)),
            "High52W": float(last.get("High52W", np.nan)),

            "Dist_to_SMA200_%": float(last.get("Dist_to_SMA200_%", np.nan)),
            "Dist_to_52W_High_%": float(last.get("Dist_to_52W_High_%", np.nan)),

            "Signal": sig,
            "SignalAuto": False,

            # Commentary placeholders (filled later)
            "Comment": None,
            "Playbook": None,

            # Fundamentals
            "Fundy_Score": fundy.get("score", 0.0),
            "Fundy_Tier": fundy.get("tier", "Neutral"),
            "Fundy": fundy,

            # Patterns / flags
            "Flag": bool(flag_flag),
            "_flag_info": flag_det,
            "_pattern_lines": (p0.get("lines") if p0 else None),
            "_pattern_name": (p0.get("name") if p0 else ""),
            "_pattern_status": (p0.get("status") if p0 else ""),
            "_pattern_conf": (p0.get("confidence") if p0 else 0.0),
            "_pattern_align": palign,

            # AutoDCA
            "AutoDCA_Flag": bool(gate_flag),
            "AutoDCA_Gap_%": float(gate_det.get("gap_pct", 0.0) or 0.0),
            "AutoDCA_Fill_%": float(gate_det.get("gap_fill_%", 0.0) or 0.0),
            "AutoDCA_ReclaimMid": bool(gate_det.get("reclaim_mid", False)),
            "AutoDCA_AboveEMA21": bool(gate_det.get("above_ema21", False)),

            # Breakout
            "BreakoutReady": bool(brk_ready),
            "Breakout_Level": float(brk_info.get("level", np.nan)),
            "Breakout_Direction": str(brk_info.get("direction", "")),

            # Litmus
            "_litmus": litmus,

            # Full indicator frame for charting
            "_ind": ind,
        })

    snaps_df = pd.DataFrame(snaps)

    if not snaps_df.empty:
        comments, playbooks, candles, sparks = [], [], [], []

        for _, r in snaps_df.iterrows():
            summary, playbook = comment_for_row(r)

            # Append litmus stats (if available) into playbook (keeps cards actionable + evidence-based)
            lit = r.get("_litmus", {}) or {}
            if lit:
                parts = []
                for h in LITMUS_SIGNAL_HORIZONS:
                    m = lit.get(f"BUY_{h}d_med", np.nan)
                    hit = lit.get(f"BUY_{h}d_hit", np.nan)
                    if np.isfinite(m) and np.isfinite(hit):
                        parts.append(f"{h}d: med {m:+.1f}%, hit {hit:.0f}%")
                if parts:
                    playbook += f"<br>• <b>Litmus (BUY outcomes):</b> " + " • ".join(parts)

            # News (AUS only) — surface inside the comment so the card gets the 'News' tag
            if not news_df.empty:
                nd = news_df[(news_df["Ticker"] == r["Ticker"]) & (news_df["Recent"])]
                if not nd.empty:
                    headline = str(nd.iloc[-1]["Headline"])
                    summary += f" • News: {headline}"
                    playbook += f"<br>• <b>News:</b> {headline}"

            comments.append(summary)
            playbooks.append(playbook)
            candles.append(mini_candle(r["_ind"], r["_flag_info"] if r["Flag"] else None, r["_pattern_lines"]))
            sparks.append(mini_spark(r["_ind"]))

        snaps_df["Comment"] = comments
        snaps_df["Playbook"] = playbooks
        snaps_df["_mini_candle"] = candles
        snaps_df["_mini_spark"] = sparks

    return snaps_df, news_df

# ---------------- HTML Construction ----------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --bg: #0f172a; --surface-1: #1e293b; --surface-2: #334155; --primary: #3b82f6; --text-main: #f1f5f9; --text-muted: #94a3b8; --accent-green: #10b981; --accent-amber: #f59e0b; --accent-red: #ef4444; --accent-purple: #a855f7; --glass: rgba(30, 41, 59, 0.7); --border: rgba(148, 163, 184, 0.1); }
* { box-sizing: border-box; -webkit-font-smoothing: antialiased; }
body { background: var(--bg); background-image: radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.1) 0px, transparent 50%), radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.1) 0px, transparent 50%); background-attachment: fixed; color: var(--text-main); font-family: 'Inter', sans-serif; margin: 0; padding-bottom: 60px; font-size: 14px; }
.mono { font-family: 'JetBrains Mono', monospace; }
.text-green { color: var(--accent-green); } .text-red { color: var(--accent-red); } .text-amber { color: var(--accent-amber); } .text-purple { color: var(--accent-purple); } .text-primary { color: var(--primary); } .hidden { display: none !important; }
.market-tabs { position: sticky; top: 0; z-index: 200; background: #020617; border-bottom: 1px solid var(--border); display: flex; justify-content: center; gap: 10px; padding: 10px; }
.market-tab { background: transparent; border: 1px solid var(--text-muted); color: var(--text-muted); padding: 8px 20px; border-radius: 999px; cursor: pointer; font-weight: 600; transition: 0.2s; }
.market-tab.active { background: var(--primary); border-color: var(--primary); color: white; }
.nav-wrapper { position: sticky; top: 53px; z-index: 100; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border); padding: 10px 16px; }
.nav-inner { display: flex; align-items: center; gap: 12px; max-width: 1200px; margin: 0 auto; overflow-x: auto; scrollbar-width: none; }
.nav-inner::-webkit-scrollbar { display: none; }
.nav-link { white-space: nowrap; color: var(--text-muted); text-decoration: none; padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 500; background: rgba(255,255,255,0.03); border: 1px solid transparent; transition: all 0.2s; }
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
.metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 8px; }
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
.badge.hold { background: rgba(59, 130, 246, 0.10); color: var(--primary); border: 1px solid rgba(59, 130, 246, 0.18); }
.badge.trim { background: rgba(168, 85, 247, 0.15); color: var(--accent-purple); border: 1px solid rgba(168, 85, 247, 0.2); }
.badge.avoid { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.2); }
.badge.news { background: rgba(168, 85, 247, 0.15); color: var(--accent-purple); }
.badge.shield-high { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.2); }
.badge.shield-low { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.2); }
.badge.venture-high { background: rgba(168, 85, 247, 0.15); color: var(--accent-purple); border: 1px solid rgba(168, 85, 247, 0.2); }
.badge.spec-high { background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.2); }
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
    
    score = r['Fundy_Score']
    cat = r['Category']
    
    s_badge, s_icon = "watch", "⚖️"
    if cat == "Core":
        s_badge = "shield-high" if score >= 7 else ("shield-low" if score < 4 else "watch")
        s_icon = "🛡️"
    elif cat == "Growth":
        s_badge = "venture-high" if score >= 7 else "watch"
        s_icon = "🚀"
    elif cat == "Spec":
        s_badge = "spec-high" if score >= 7 else "shield-low"
        s_icon = "❤️"

    fundy_html = f'<span class="badge {s_badge}" style="margin-left:6px">{s_icon} {score}/10 {r["Fundy_Tier"]}</span>'

    return f"""
    <div class="card searchable-item {euphoria_cls}">
        <div class="card-header">
            <div>
                <span class="ticker-badge mono">{r['Ticker']}</span>
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
            <div class="metric"><label>ATR%</label><span class="mono">{r['ATR14_%']:.1f}%</span></div>
            <div class="metric"><label>Key Metric</label><span class="mono">{r['Fundy']['key_metric']}</span></div>
        </div>
        <div class="comment-box">{r['Comment']}</div>
        <div class="playbook">{r['Playbook']}</div>
        <div class="chart-container">{r['_mini_candle']}</div>
    </div>
    """

def render_kpi(label, val, color_cls):
    return f"""<div class="kpi-card"><div class="kpi-lbl">{label}</div><div class="kpi-val {color_cls}">{val}</div></div>"""

def render_market_html(m_code, m_conf, snaps_df, news_df):
    if snaps_df.empty: return f"<div id='cont-{m_code}' class='market-container'><div style='padding:50px'>No Data</div></div>"
    
    mask_spec = snaps_df['Category'] == 'Spec'
    DEGEN = snaps_df[mask_spec].sort_values(['Fundy_Score', 'RSI14'], ascending=[True, False])
    CORE  = snaps_df[~mask_spec]

    BUY   = CORE[CORE.Signal == 'BUY'].sort_values(['Fundy_Score'], ascending=False)
    DCA   = CORE[CORE.Signal == 'DCA'].sort_values(['Fundy_Score'], ascending=False)
    HOLD  = CORE[CORE.Signal == 'HOLD'].sort_values(['Fundy_Score'], ascending=False)
    TRIM  = CORE[CORE.Signal == 'TRIM'].sort_values(['Fundy_Score'], ascending=False)
    WATCH = CORE[CORE.Signal == 'WATCH'].sort_values(['Fundy_Score'], ascending=False)
    AVOID = CORE[CORE.Signal == 'AVOID'].sort_values(['Fundy_Score'], ascending=True)

    GATE  = CORE[CORE['AutoDCA_Flag']==True]
    PATS  = CORE[CORE['_pattern_name']!='']
    
    def mk_card(df, badge):
        if df.empty: return ""
        h = f"<h2 id='{m_code}-{badge}' style='color:var(--text-muted);margin-top:30px'>{badge.upper()}</h2><div class='grid'>"
        for _, r in df.iterrows(): h += render_card(r, badge, m_conf['currency'])
        return h + "</div>"

    html_cards = mk_card(BUY,'buy') + mk_card(DCA,'dca') + mk_card(HOLD,'hold') + mk_card(TRIM,'trim') + mk_card(WATCH,'watch') + mk_card(AVOID,'avoid')

    degen_rows = "".join([f"<tr class='searchable-item'><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td><span class='badge shield-low'>{r['Fundy_Score']} {r['Fundy_Tier']}</span></td><td>{r['Signal']}</td><td class='mono'>{r['Fundy']['key_metric']}</td><td>{r['_mini_spark']}</td></tr>" for _, r in DEGEN.iterrows()]) if not DEGEN.empty else "<tr><td colspan='5' style='text-align:center'>No Degens.</td></tr>"
    
    gate_rows = "".join([f"<tr class='searchable-item'><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td class='mono text-red'>{r['AutoDCA_Gap_%']:.1f}%</td><td class='mono'>{'Yes' if r['AutoDCA_ReclaimMid'] else 'No'}</td><td class='mono'>{'Yes' if r['AutoDCA_AboveEMA21'] else 'No'}</td><td class='mono'>{r['AutoDCA_Fill_%']:.1f}%</td><td>{r['_mini_spark']}</td></tr>" for _, r in GATE.iterrows()]) if not GATE.empty else "<tr><td colspan='6' style='text-align:center'>No setups.</td></tr>"
    
    pat_rows = "".join([f"<tr class='searchable-item'><td>{r['_pattern_name']}</td><td><span class='ticker-badge mono'>{r['Ticker']}</span></td><td class='mono'>{r['_pattern_status']}</td><td class='mono'>{r['_pattern_conf']:.2f}</td><td class='mono'>{r['_pattern_align']}</td><td>{r['_mini_candle']}</td></tr>" for _, r in PATS.iterrows()]) if not PATS.empty else "<tr><td colspan='6' style='text-align:center'>No patterns.</td></tr>"
    
    news_rows = "".join([f"<tr class='searchable-item'><td class='mono' style='color:var(--text-muted)'>{r['Date']}</td><td><b>{r['Ticker']}</b></td><td><span class='badge news'>{r['Type']}</span></td><td>{r['Headline']}</td></tr>" for _, r in news_df.sort_values('Date', ascending=False).iterrows()]) if not news_df.empty else "<tr><td colspan='4' style='text-align:center'>No news.</td></tr>"

    f_rows = ""
    for _, r in snaps_df.sort_values('Fundy_Score', ascending=False).iterrows():
        sc = r['Fundy_Score']
        cl = 'shield-high' if sc>=7 else ('watch' if sc>=4 else 'shield-low')
        icon = "🛡️"
        if r['Category'] == 'Growth': icon = "🚀"
        elif r['Category'] == 'Spec': icon = "❤️"
        
        f_rows += f"<tr class='searchable-item'><td>{r['Ticker']}</td><td><span class='badge {cl}'>{icon} {sc} {r['Fundy_Tier']}</span></td><td class='mono'>{r['Fundy']['key_metric']}</td><td class='mono'>{r['Fundy']['roe']*100:.1f}%</td><td class='mono'>{r['Fundy']['debteq']:.2f}</td><td class='mono'>{r['Category']}</td></tr>"

    kpi_html = f"""<div class="kpi-scroll">
        {render_kpi('Buy', len(BUY), 'text-green')}
        {render_kpi('DCA', len(DCA), 'text-amber')}
        {render_kpi('Hold', len(HOLD), 'text-primary')}
        {render_kpi('Trim', len(TRIM), 'text-purple')}
        {render_kpi('Watch', len(WATCH), 'text-primary')}
        {render_kpi('Avoid', len(AVOID), 'text-red')}
        {render_kpi('Degens', len(DEGEN), 'text-purple')}
        {render_kpi('Auto-DCA', len(GATE), 'text-main')}
    </div>"""

    nav = f"""<div class="nav-wrapper"><div class="nav-inner">
    <a href="#{m_code}-top" class="nav-link">Main</a>
    <a href="#{m_code}-degen" class="nav-link">Degenerate Radar</a>
    <a href="#{m_code}-gate" class="nav-link">Auto-DCA</a>
    <a href="#{m_code}-patterns" class="nav-link">Patterns</a>
    <a href="#{m_code}-news" class="nav-link">News</a>
    <a href="#{m_code}-fundy" class="nav-link">Deep Fundamentals</a>
    </div></div>"""

    return f"""
    <div id="cont-{m_code}" class="market-container {'active' if m_code=='AUS' else ''}">
        {nav}
        <div class="search-container"><input type="text" id="search-{m_code}" class="search-input" placeholder="Search {m_conf['name']}..."></div>
        <div class="container">
            <h1 id="{m_code}-top" style="margin-bottom:20px">{m_conf['name']}</h1>
            {kpi_html}
            {html_cards}
            
            <h2 id="{m_code}-degen" style="margin-top:40px">Degenerate Radar (Spec Only)</h2>
            <div class="card"><div class="table-responsive"><table><thead><tr><th>Ticker</th><th>Score</th><th>Sig</th><th>Metric</th><th>Spark</th></tr></thead><tbody>{degen_rows}</tbody></table></div></div>
            
            <h2 id="{m_code}-gate" style="margin-top:40px">Auto-DCA Candidates</h2>
            <div class="card"><div class="table-responsive"><table><thead><tr><th>Ticker</th><th>Gap %</th><th>Reclaim?</th><th>EMA21?</th><th>Fill %</th><th>Trend</th></tr></thead><tbody>{gate_rows}</tbody></table></div></div>
            
            <h2 id="{m_code}-patterns" style="margin-top:40px">Patterns & Structures</h2>
            <div class="card"><div class="table-responsive"><table><thead><tr><th>Pattern</th><th>Ticker</th><th>Status</th><th>Conf</th><th>Align</th><th>Mini</th></tr></thead><tbody>{pat_rows}</tbody></table></div></div>
            
            <h2 id="{m_code}-news" style="margin-top:40px">News</h2>
            <div class="card" style="padding:0"><div class="table-responsive"><table><thead><tr><th>Date</th><th>Ticker</th><th>Type</th><th>Headline</th></tr></thead><tbody>{news_rows}</tbody></table></div></div>
            
            <h2 id="{m_code}-fundy" style="margin-top:40px">Deep Fundamentals</h2>
            <div class="card"><div class="table-responsive"><table><thead><tr><th>Ticker</th><th>Score</th><th>Key Metric</th><th>ROE</th><th>Debt/Eq</th><th>Cat</th></tr></thead><tbody>{f_rows}</tbody></table></div></div>
            
            <div style="height:50px"></div>
        </div>
    </div>
    """

if __name__ == "__main__":
    print("Starting TraderBruh Global Hybrid v6.6...")
    market_htmls, tab_buttons = [], []
    
    # Force Sydney Time
    au_tz = zoneinfo.ZoneInfo("Australia/Sydney")
    gen_time = datetime.now(au_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    build_meta = get_build_meta()
    
    for m, conf in MARKETS.items():
        df, news = process_market(m, conf)
        market_htmls.append(render_market_html(m, conf, df, news))
        act = "active" if m=="AUS" else ""
        tab_buttons.append(f"<button id='tab-{m}' class='market-tab {act}' onclick=\"switchMarket('{m}')\">{conf['name']}</button>")
    
    full = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>TraderBruh v6.6</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>{CSS}</style><script>{JS}</script></head><body>
    <div style="text-align:center; padding:10px 0 5px 0; color:#64748b; font-size:11px; font-family:'JetBrains Mono', monospace;">Built: {gen_time} · Commit: {build_meta['sha7']} · Run: {build_meta['run_id']} · Ref: {build_meta['ref']} · Actor: {build_meta['actor']} · Script: Ultimate 6.6</div>
    <div class="market-tabs">{''.join(tab_buttons)}</div>{''.join(market_htmls)}</body></html>"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f: f.write(full)
    print("Done:", OUTPUT_HTML)

