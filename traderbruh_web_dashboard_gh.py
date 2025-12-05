# traderbruh_web_dashboard_gh.py
# TraderBruh — Global Web Dashboard (ASX / USA / INDIA)
# Version: Ultimate 6.4 (Final Polish)
# - Massive Ticker Expansion
# - Academic Logic (Rule of 40, Mohanram, Runway)
# - Actionable Commentary Engine 2.1

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
            "ALU": ("Altium", "PCB Design Software.", "Growth"),
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
            "DEG": ("De Grey", "Gold Exploration.", "Spec"),
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
            elif "volume" in c: col_map[col] = "Volume"
        
        df = df.rename(columns=col_map)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in needed): return pd.DataFrame()

        df = df[needed].reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df["Date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")

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

# ---------------- Commentary Engine 2.1 (Actionable + Deep Logic) ----------------

def comment_for_row(r: pd.Series) -> str:
    d200 = r.get("Dist_to_SMA200_%", 0.0)
    d52  = r.get("Dist_to_52W_High_%", 0.0)
    dist_high = abs(d52)
    rsi  = r.get("RSI14", 50.0)
    sig  = str(r.get("Signal", "")).upper()
    score = r.get("Fundy_Score", 0)
    
    fundy = r.get("Fundy", {})
    cat   = fundy.get("category_mode", "Core")
    runway = fundy.get("runway_months", 999.0)
    
    is_elite = (score >= 7)
    is_trash = (score < 4)
    is_weak  = is_trash # Alias fix
    is_zombie = (cat == "Spec" and runway < 6.0 and runway > 0)
    is_euphoria = (d52 > -3.5) and (d200 > 40.0) and (rsi >= 70.0)
    is_oversold = (rsi <= 35.0)

    mode_label = ""
    if cat == "Core": mode_label = "Fortress"
    elif cat == "Growth": mode_label = "Venture"
    elif cat == "Spec": mode_label = "Survival"

    base = ""

    if sig == "BUY":
        if is_zombie:
            base = (f"<b>💀 DILUTION TRAP:</b> Chart is up, but cash is critically low (<{runway:.1f}m). "
                    f"They will likely use this rally to raise capital. Take quick profits or avoid.")
        elif is_elite:
            base = (f"<b>🚀 ROCKET FUEL:</b> Strong uptrend in {mode_label} name ({score}/10). "
                    f"High conviction setup. Build position on pullbacks to EMA21.")
        elif is_weak:
            if cat == "Spec":
                base = (f"<b>🎲 LOTTO TICKET:</b> Momentum is hot but fundamentals are thin ({score}/10). "
                        f"Trade the chart only – use tight trailing stops. Casino money.")
            else:
                base = (f"<b>🗑️ TRASH RALLY:</b> Price moving up, but business quality is low ({score}/10). "
                        f"Rent the trade, don't own the company. Exit on first sign of weakness.")
        else:
            base = (f"<b>✅ STANDARD BUY:</b> Healthy trend > 200DMA. Fundamentals decent ({score}/10). "
                    f"Starter size now, look to add if it holds support.")

    elif sig == "DCA":
        if is_zombie:
            base = (f"<b>🩸 CATCHING KNIVES:</b> Don't do it. Stock is falling and they have <{runway:.1f}m cash. "
                    f"Bankruptcy or massive dilution risk. Stay away.")
        elif is_trash:
            base = (f"<b>💣 VALUE TRAP:</b> It looks cheap, but quality is poor ({score}/10). "
                    f"Do not average down. Wait for a confirmed reversal or cut loss.")
        elif is_elite:
            if is_oversold:
                base = (f"<b>💎 GOLDEN BUCKET:</b> Elite {mode_label} ({score}/10) is deeply oversold (RSI {rsi:.0f}). "
                        f"High probability zone for patient accumulation. Scale in.")
            else:
                base = (f"<b>🛒 QUALITY ON SALE:</b> Rare dip in a {score}/10 business. "
                        f"Price near 200DMA (Δ{d200:.1f}%). Good spot to start a long-term position.")
        else:
            base = (f"<b>📉 SWING ZONE:</b> Testing 200DMA support. "
                    f"Play the bounce with a stop below the line. Fundamental conviction is average.")

    elif sig == "WATCH":
        if is_euphoria:
            if is_weak:
                base = (f"<b>🚨 EXIT SCAM WARNING:</b> Weak stock ({score}/10) gone vertical. "
                        f"RSI {rsi:.0f} is dangerous. Lock in profits immediately or tighten stops.")
            else:
                base = (f"<b>🍾 EUPHORIA:</b> Great company, terrible entry. RSI {rsi:.0f} is stretched. "
                        f"Do not chase. Set alert for a pullback to EMA21.")
        elif is_elite:
            base = (f"<b>🎯 SNIPER LIST:</b> Elite {mode_label} ({score}/10) consolidating. "
                    f"Stalking mode: Wait for a breakout of highs or a dip to the 50DMA.")
        else:
            if dist_high < 5.0:
                base = (f"<b>🚪 KNOCKING:</b> Coiling tight, just {dist_high:.1f}% below highs. "
                        f"Set buy-stop alert above resistance for a breakout trade.")
            elif dist_high > 25.0:
                base = (f"<b>🤕 REPAIRING:</b> Down {dist_high:.1f}% from highs. Trend is broken. "
                        f"Dead money until it reclaims the 200DMA. Ignore.")
            elif rsi > 60:
                base = (f"<b>🔥 HEATING UP:</b> Momentum building (RSI {rsi:.0f}), but no clear entry. "
                        f"Watch for a 'Bull Flag' pattern to form.")
            else:
                base = (f"<b>💤 CHOP CITY:</b> Sideways action. No edge here currently. "
                        f"Focus capital elsewhere.")

    elif sig == "AVOID":
        if is_elite:
            base = (f"<b>🥶 VALUE TRAP:</b> Great {mode_label} business ({score}/10) but chart is broken. "
                    f"Don't fight the trend. Wait for a Stage 1 base to form.")
        else:
            base = (f"<b>💀 RADIOACTIVE:</b> Broken chart + weak fundamentals ({score}/10). "
                    f"High opportunity cost. Remove from watchlist.")

    else:
        base = "Neutral – no clear edge."

    if cat == "Growth" and score > 6:
        base += " <br><i>💡 Note: Efficient Growth Machine (Rule of 40).</i>"
    if cat == "Spec" and score > 6 and runway >= 18:
        base += " <br><i>💡 Note: Well Funded (Runway > 18m).</i>"

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
        
        fundy = fetch_dynamic_fundamentals(full_sym, t_cat)
        
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

        if t_cat == "Core" and fundy["score"] < 4: sig = "AVOID"
        if t_cat == "Spec" and fundy["score"] < 3: sig = "AVOID"

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
            <div class="metric"><label>Key Metric</label><span class="mono">{r['Fundy']['key_metric']}</span></div>
        </div>
        <div class="comment-box">{r['Comment']}</div>
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
    WATCH = CORE[CORE.Signal == 'WATCH'].sort_values(['Fundy_Score'], ascending=False)
    AVOID = CORE[CORE.Signal == 'AVOID'].sort_values(['Fundy_Score'], ascending=True)

    GATE  = CORE[CORE['AutoDCA_Flag']==True]
    PATS  = CORE[CORE['_pattern_name']!='']
    
    def mk_card(df, badge):
        if df.empty: return ""
        h = f"<h2 id='{m_code}-{badge}' style='color:var(--text-muted);margin-top:30px'>{badge.upper()}</h2><div class='grid'>"
        for _, r in df.iterrows(): h += render_card(r, badge, m_conf['currency'])
        return h + "</div>"

    html_cards = mk_card(BUY,'buy') + mk_card(DCA,'dca') + mk_card(WATCH,'watch') + mk_card(AVOID,'avoid')

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
        {render_kpi('Watch', len(WATCH), 'text-primary')}
        {render_kpi('Avoid', len(AVOID), 'text-red')}
        {render_kpi('Degens', len(DEGEN), 'text-purple')}
        {render_kpi('Auto-DCA', len(GATE), 'text-main')}
    </div>"""

    nav = f"""<div class="nav-wrapper"><div class="nav-inner">
    <a href="#{m_code}-buy" class="nav-link">Main</a>
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
            <h1 style="margin-bottom:20px">{m_conf['name']}</h1>
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
    print("Starting TraderBruh Global Hybrid v6.4...")
    market_htmls, tab_buttons = [], []
    for m, conf in MARKETS.items():
        df, news = process_market(m, conf)
        market_htmls.append(render_market_html(m, conf, df, news))
        act = "active" if m=="AUS" else ""
        tab_buttons.append(f"<button id='tab-{m}' class='market-tab {act}' onclick=\"switchMarket('{m}')\">{conf['name']}</button>")
    
    full = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>TraderBruh v6.4</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>{CSS}</style><script>{JS}</script></head><body><div class="market-tabs">{''.join(tab_buttons)}</div>{''.join(market_htmls)}</body></html>"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f: f.write(full)
    print("Done:", OUTPUT_HTML)

