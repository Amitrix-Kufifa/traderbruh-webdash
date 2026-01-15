# traderbruh_web_dashboard_gh.py
# TraderBruh — Global Web Dashboard (ASX / USA / INDIA)
# Version: Ultimate 8.0 (Institutional robustness: Wilder ATR/RSI, fundamentals cache + unverified fallback, vol-adaptive patterns, ATR-risk backtest sizing, parallel price fetch)
# - Fixed breakout logic (20D/52W highs shifted to avoid self-referencing)
# - Added HOLD + TRIM signals (explicit hodl / take-profit guidance)
# - Optional split/dividend-adjusted indicator series (AdjClose) for cleaner long lookbacks
# - Generic breakout/breakdown detector for multiple patterns (volume + ATR confirmation)
# - Much more directive commentary "playbook" per ticker (entry/add/stop/trim rules)
# - Optional in-dashboard litmus stats (median forward returns after BUY/DCA)
# - NEW: Market regime filter (benchmark trend) + Relative Strength (leader filtering)
# - NEW: Comment action line now includes concrete levels (entry/stop/trim)
# - Renamed 'Litmus' → 'Past results' with simpler explanation

from datetime import datetime, time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
PRICE_TOL           = 0.03  # fallback only (if ATR ref unavailable)
_SIM_ATR_REF        = np.nan  # set per ticker during pattern scanning
PATTERNS_CONFIRMED_ONLY = True


# --- Data / Backtest Safety ---
# NOTE: yfinance 'Adj Close' is split/dividend-adjusted. Using it for indicators avoids split artifacts.
USE_ADJUSTED_FOR_INDICATORS = True

# NOTE: Appending intraday bars into a daily series can distort indicators. Kept for experimentation only.
APPEND_INTRADAY_BAR = False


# Chart cleanliness toggles
CHART_SHOW_PATTERN_LINES = False   # show pattern levels on mini charts (can get busy)
CHART_SHOW_FLAG_CHANNEL  = False   # show bull-flag channel on mini charts
CHART_KEY_IN_CARD         = True   # show a small chart key under each mini chart
# Optional: lightweight in-dashboard "litmus" stats (forward returns after signals, etc.)
ENABLE_LITMUS_STATS = True
LITMUS_SIGNAL_HORIZONS = (5, 20, 60)   # trading days
LITMUS_LOOKBACK_BARS   = 252 * 6       # ~6 years for stats window

# Optional: investor-mode "real answer" backtest (Variant 2)
# Variant 2 = start from cash; only enter on BUY / DCA Dip / DCA Reclaim; exit on AVOID; trim on TRIM.
# This is NOT a guarantee of future performance—just a structured historical sanity-check.
ENABLE_BACKTEST_REPORT = True
BACKTEST_WINDOWS = {"3m": 63, "6m": 126}   # trading days
BACKTEST_MAX_POSITIONS = 12                # portfolio capacity (equal-lot sizing)
BACKTEST_DCA_MAX_LOTS = 2                  # cap adds per name (2 lots max)
BACKTEST_DCA_ADD_FRACTION = 0.5            # add-size vs base lot (0.5 = half-lot adds)
BACKTEST_TRIM_FRACTION = 0.33              # when TRIM triggers, sell this fraction of position
BACKTEST_FEE_BPS = 10                      # 10 bps per trade side (rough friction estimate)

# Backtest sizing upgrade (volatility / ATR targeting)
BACKTEST_RISK_PCT = 0.005              # risk budget per new position (0.5% of equity)
BACKTEST_STOP_ATR_MULT = 3.0           # assume an ATR-based stop distance
BACKTEST_MAX_POS_VALUE_PCT = 0.20      # cap position notional (avoid concentration)
BACKTEST_MIN_CASH_BUFFER_PCT = 0.01    # keep a small cash buffer

# --- Fundamentals caching (robustness + speed) ---
# Fundamentals don't change daily; yfinance info/statements can be slow or rate-limited.
# This cache becomes most valuable if you persist it across runs (e.g., via GitHub Actions cache).
ENABLE_FUNDAMENTALS_CACHE = True
FUNDAMENTALS_CACHE_PATH = os.environ.get("TRADERBRUH_FUND_CACHE", ".cache/traderbruh_fundamentals.json")
FUNDAMENTALS_CACHE_TTL_DAYS = 14           # refresh cadence
FUNDAMENTALS_CACHE_STALE_OK_DAYS = 90      # if live fetch fails, allow stale cache up to this age
FUNDAMENTALS_FAIL_POLICY = "unverified"    # keeps technical signal + flags fundamentals as unverified

# --- Parallel price fetching (kept conservative to reduce rate-limit risk) ---
ENABLE_PARALLEL_PRICE_FETCH = True
PRICE_FETCH_MAX_WORKERS = 6

# --- Volatility-adaptive pattern tolerance ---
# Instead of a fixed % tolerance, compare pivots using a fraction of recent ATR.
SIMILAR_ATR_MULT = 0.6   # abs(a-b) <= SIMILAR_ATR_MULT * median_ATR

# --- Internal caches (in-memory) ---
_FUND_CACHE = None
_FUND_CACHE_DIR_READY = False

def _ensure_cache_dir():
    global _FUND_CACHE_DIR_READY
    if _FUND_CACHE_DIR_READY:
        return
    try:
        d = os.path.dirname(FUNDAMENTALS_CACHE_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    _FUND_CACHE_DIR_READY = True

def _load_fund_cache():
    global _FUND_CACHE
    if _FUND_CACHE is not None:
        return _FUND_CACHE
    _ensure_cache_dir()
    try:
        if os.path.exists(FUNDAMENTALS_CACHE_PATH):
            with open(FUNDAMENTALS_CACHE_PATH, "r", encoding="utf-8") as f:
                _FUND_CACHE = json.load(f)
        else:
            _FUND_CACHE = {}
    except Exception:
        _FUND_CACHE = {}
    if not isinstance(_FUND_CACHE, dict):
        _FUND_CACHE = {}
    return _FUND_CACHE

def _save_fund_cache():
    if not ENABLE_FUNDAMENTALS_CACHE:
        return
    _ensure_cache_dir()
    try:
        with open(FUNDAMENTALS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_FUND_CACHE or {}, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        pass

def _cache_key(symbol: str, category: str) -> str:
    return f"{symbol}||{category}"

def _cache_age_days(ts: float) -> float:
    try:
        return (datetime.utcnow().timestamp() - float(ts)) / 86400.0
    except Exception:
        return 1e9

def _get_cached_fund(symbol: str, category: str):
    cache = _load_fund_cache()
    v = cache.get(_cache_key(symbol, category))
    if not isinstance(v, dict):
        return None, None
    age = _cache_age_days(v.get("ts", 0.0))
    return v, age

def _set_cached_fund(symbol: str, category: str, payload: dict):
    if not ENABLE_FUNDAMENTALS_CACHE:
        return
    cache = _load_fund_cache()
    payload = dict(payload or {})
    payload["ts"] = datetime.utcnow().timestamp()
    cache[_cache_key(symbol, category)] = payload



# --- Market regime + relative strength (leader filtering) ---
# These are optional but very useful for "robust logic":
# - Market regime: avoid most BUY/DCA signals when the broader market is risk-off.
# - Relative strength: prefer stocks outperforming their benchmark (leaders, not laggards).
ENABLE_MARKET_REGIME_FILTER = True
MARKET_FILTER_MODE = "hard"   # "hard" blocks BUY/DCA when market is risk-off; "soft" only adds caution text

ENABLE_RELATIVE_STRENGTH = True
RS_LOOKBACK_BARS = 63         # ~3 months (trading days)
RS_SLOPE_WINDOW  = 20
RS_MIN_OUTPERF_PCT = 0.0      # require >=0% outperformance over lookback for BUY

# --- Fundamental gating (quality bias) ---
# DCA is "averaging into a position" — we bias this toward higher-quality businesses to avoid value traps.
# These are soft knobs; if your scoring model changes, tweak these thresholds.
MIN_SHIELD_FOR_DCA_CORE   = 4.5
MIN_SHIELD_FOR_DCA_GROWTH = 5.0


# Benchmarks used for regime + relative strength. (We try in order; first that downloads wins.)
BENCHMARKS = {
    "AUS":   ["A200.AX", "^AXJO"],
    "USA":   ["SPY", "^GSPC"],
    "INDIA": ["^NSEI", "NIFTYBEES.NS", "INDA"],
}

RULES = {
    # BUY = momentum / trend continuation (breakout in a Stage-2 uptrend)
    # buffer_pct: require a small "clearance" above prior high to reduce false breaks
    # max_dist200_pct: don't chase if already extremely stretched vs 200DMA
    "buy": {
        "rsi_min": 50,
        "rsi_max": 75,
        "vol_mult": 1.10,
        "buffer_pct": 0.003,          # +0.3% above prior 20D high
        "max_dist200_pct": 25.0,      # avoid chasing >25% above 200DMA
    },

    # DCA = quality dip near 200DMA (intended for Core/Growth, not pure Spec)
    # allow_below_pct: allow a small undercut of 200DMA without instantly flipping to AVOID
    "dca": {
        "rsi_max": 55,
        "sma200_proximity": 0.04,     # within +4% of 200DMA
        "allow_below_pct": 0.02,      # allow -2% undercut
    },

    # AVOID = damage control regime (trend broken / downtrend)
    "avoid": {"death_cross": True},

    # TRIM = profit-taking regime (usually triggered by "extended" conditions)
    "trim": {"enabled": True},

    # Risk model (used in commentary)
    "risk": {"atr_stop_mult": 2.0, "atr_trail_mult": 3.0, "sma200_break_pct": 0.03},

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

            "NKE":  ("Nike", "Athletic Apparel.", "Core"),
            "PYPL": ("PayPal", "Payments/Fintech.", "Growth"),
            "CELH": ("Celsius", "Energy Drinks.", "Growth"),
            "SOFI": ("SoFi", "Fintech Bank.", "Growth"),
            "RKT":  ("Rocket", "Mortgage/Fintech.", "Spec"),
            "FUBO": ("fuboTV", "Streaming/Sports.", "Spec"),
            "BMNU": ("T-REX 2X Long BMNR ETF", "Leveraged ETF.", "Spec"),
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
    # --- Cache (avoid yfinance SPOF) ---
    if ENABLE_FUNDAMENTALS_CACHE:
        cached, age = _get_cached_fund(symbol, category)
        if cached is not None and age is not None and age <= float(FUNDAMENTALS_CACHE_TTL_DAYS):
            out = dict(cached)
            out["verified"] = bool(out.get("verified", True))
            out["source"] = "cache"
            out["cache_age_days"] = float(age)
            return out


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

        out = {
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
        out["verified"] = True
        out["source"] = "live"
        _set_cached_fund(symbol, category, out)
        _save_fund_cache()
        return out

    except Exception:
        # Fallback: if live fetch fails, use stale cache if available (up to FUNDAMENTALS_CACHE_STALE_OK_DAYS)
        if ENABLE_FUNDAMENTALS_CACHE:
            cached, age = _get_cached_fund(symbol, category)
            if cached is not None and age is not None and age <= float(FUNDAMENTALS_CACHE_STALE_OK_DAYS):
                out = dict(cached)
                out["source"] = "stale-cache"
                out["cache_age_days"] = float(age)
                # mark as unverified if beyond TTL
                out["verified"] = bool(age <= float(FUNDAMENTALS_CACHE_TTL_DAYS))
                return out

        # No cache available — do not punish the technical signal; mark fundamentals as unverified
        return {
            "score": 0.0,
            "tier": "Unverified",
            "category_mode": str(category),
            "key_metric": "-",
            "roe": 0.0,
            "gross": 0.0,
            "oper": 0.0,
            "debteq": 0.0,
            "cash": 0.0,
            "runway_months": 0.0,
            "burn_annual": 0.0,
            "verified": False,
            "source": "none",
        }
def indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicator stack used by the dashboard.

    Key design choices:
    - Signals are computed on a single "Price" series.
    - If AdjClose is present (yfinance) we can use it to avoid split/dividend artifacts in long lookbacks.
    - Breakout highs/lows are shifted by 1 bar to avoid the "Close > rolling(max including today)" bug.
    """
    x = df.copy().sort_values("Date").reset_index(drop=True)

    # Defensive: yfinance can occasionally return MultiIndex or duplicate columns.
    # If duplicates exist, pandas can return a DataFrame when selecting a "single" column (e.g. AdjClose),
    # which breaks pd.to_numeric(). We keep the first occurrence.
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    if hasattr(x.columns, 'duplicated') and x.columns.duplicated().any():
        x = x.loc[:, ~x.columns.duplicated()]

    # --- Price series for indicator calculations ---
    if USE_ADJUSTED_FOR_INDICATORS and ("AdjClose" in x.columns):
        # AdjClose is typically aligned to last close (factor ~1 on the latest bar)
        adj = x["AdjClose"]
        # Pandas returns a DataFrame if there are duplicate 'AdjClose' columns
        # (or a MultiIndex selection). Take the first column defensively.
        if isinstance(adj, pd.DataFrame):
            adj = adj.iloc[:, 0]
        x["Price"] = pd.to_numeric(adj, errors="coerce")
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

    # --- Simple reclaim markers (used to split DCA into Dip vs Reclaim)
    # Reclaim21: price crosses back above the 21EMA today.
    # Reclaim200: price crosses back above the 200DMA today.
    x["Reclaim21"]  = (x["Price"] > x["EMA21"])  & (x["Price"].shift(1) <= x["EMA21"].shift(1))
    x["Reclaim200"] = (x["Price"] > x["SMA200"]) & (x["Price"].shift(1) <= x["SMA200"].shift(1))

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
    x["DollarVol20"] = x["Vol20"] * x["Price"]


    # --- RSI(14) on Price (Wilder smoothing / RMA) ---
    chg = x["Price"].diff()
    up = chg.clip(lower=0)
    dn = (-chg).clip(lower=0)
    avg_up = up.ewm(alpha=1/14, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1/14, adjust=False).mean()
    RS = avg_up / avg_dn.replace(0, np.nan)
    x["RSI14"] = 100 - (100 / (1 + RS))
    # --- Distances ---
    x["Dist_to_52W_High_%"] = (x["Price"] / x["High52W"] - 1) * 100.0
    x["Dist_to_SMA200_%"]   = (x["Price"] / x["SMA200"]  - 1) * 100.0
    x["Dist_EMA21_%"]     = (x["Price"] / x["EMA21"]  - 1) * 100.0
    x["Dist_SMA50_%"]     = (x["Price"] / x["SMA50"]  - 1) * 100.0

    # --- ATR(14) on adjusted OHLC (HighI/LowI) vs prior Price ---
    x["H-L"] = x["HighI"] - x["LowI"]
    x["H-C"] = (x["HighI"] - x["Price"].shift(1)).abs()
    x["L-C"] = (x["LowI"]  - x["Price"].shift(1)).abs()
    x["TR"]  = x[["H-L", "H-C", "L-C"]].max(axis=1)
    x["ATR14"]   = x["TR"].ewm(alpha=1/14, adjust=False).mean()  # Wilder RMA
    x["ATR14_%"] = (x["ATR14"] / x["Price"]).replace([np.inf, -np.inf], np.nan) * 100.0

    return x

def label_row(r: pd.Series) -> str:
    """Map the latest bar to ONE clear action label.

    Labels are intentionally opinionated and mutually exclusive:
    - AVOID  = trend is broken / downtrend (damage control)
    - BUY         = breakout continuation (trend + volume + leadership)
    - DCA_DIP     = pullback-to-200DMA zone in an uptrend (quality dip)
    - DCA_RECLAIM = a bounce day that reclaims key MAs (safer confirmation)
    - TRIM   = uptrend but *extended* (take partials / tighten stops)
    - HOLD   = uptrend intact, but no fresh edge today
    - WATCH  = no-man's land / base-building / too early

    Notes:
    - We optionally gate BUY/DCA by market regime (benchmark above rising 200DMA).
    - We optionally gate BUY by relative strength (leaders outperforming the benchmark).
    """
    price  = float(r.get("Price", np.nan))
    sma200 = float(r.get("SMA200", np.nan))
    sma50  = float(r.get("SMA50", np.nan))
    rsi    = float(r.get("RSI14", np.nan))
    high20 = float(r.get("High20", np.nan))
    vol    = float(r.get("Volume", np.nan))
    vol20  = float(r.get("Vol20", np.nan))
    d200   = float(r.get("Dist_to_SMA200_%", np.nan))
    sma200_slope_pct = float(r.get("SMA200_Slope_%", 0.0))

    # Market / RS context (may be NaN if benchmark wasn't available)
    market_up = bool(r.get("Market_Uptrend", True))
    rs3m      = float(r.get("RS_3M_%", np.nan))
    rs_slope  = float(r.get("RS_Slope20_%", np.nan))

    reclaim21  = bool(r.get("Reclaim21", False))
    reclaim200 = bool(r.get("Reclaim200", False))

    if not (np.isfinite(price) and np.isfinite(sma200) and np.isfinite(sma50) and np.isfinite(rsi)):
        return "WATCH"

    death_cross = sma50 < sma200

    # --- Damage control: Stage-4 downtrend (below & falling 200DMA) ---
    if (price < sma200) and (sma200_slope_pct < 0):
        return "AVOID"

    # --- Additional 'line in the sand' under the 200DMA ---
    break_pct = float(RULES.get("risk", {}).get("sma200_break_pct", 0.03))
    broke_200 = price < sma200 * (1.0 - break_pct)

    if RULES.get("avoid", {}).get("death_cross", True) and death_cross and (price < sma200 or broke_200):
        return "AVOID"

    # --- Trend context (Stage-2 style) ---
    trend_up = (price > sma200) and (sma50 > sma200) and (sma200_slope_pct > 0)

    # --- Market regime gate ---
    market_ok = True
    if ENABLE_MARKET_REGIME_FILTER and (str(MARKET_FILTER_MODE).lower() == "hard"):
        market_ok = market_up

    # --- Relative strength gate (leaders) ---
    rs_ok = True
    if ENABLE_RELATIVE_STRENGTH and np.isfinite(rs3m):
        rs_ok = rs3m >= RS_MIN_OUTPERF_PCT
        if np.isfinite(rs_slope):
            rs_ok = rs_ok and (rs_slope > 0)

    # --- BUY: breakout above prior 20-day high, with a small clearance buffer ---
    buffer_pct = float(RULES.get("buy", {}).get("buffer_pct", 0.0))
    trigger = high20 * (1.0 + buffer_pct) if np.isfinite(high20) else np.nan
    max_d200 = float(RULES.get("buy", {}).get("max_dist200_pct", 1e9))

    buy_ok = (
        market_ok and rs_ok and trend_up and
        np.isfinite(trigger) and (price > trigger) and
        (RULES["buy"]["rsi_min"] <= rsi <= RULES["buy"]["rsi_max"])
    )

    # Don't chase super-extended moves
    if buy_ok and np.isfinite(d200):
        buy_ok = d200 <= max_d200

    if buy_ok and np.isfinite(vol20) and vol20 > 0:
        if not np.isfinite(vol):
            buy_ok = False
        else:
            buy_ok = vol >= RULES["buy"].get("vol_mult", 1.0) * vol20

    # --- DCA: dip near 200DMA, only when the 200DMA is rising ---
    allow_below = float(RULES["dca"].get("allow_below_pct", 0.02))
    dca_ok = (
        market_ok and
        (sma200_slope_pct > 0) and
        (price >= sma200 * (1.0 - allow_below)) and
        (price <= sma200 * (1.0 + RULES["dca"]["sma200_proximity"])) and
        (rsi <= RULES["dca"]["rsi_max"]) and
        (not death_cross)
    )

    # Split DCA into:
    # - DCA_RECLAIM: we have a reclaim signal (crossing back above 21EMA or 200DMA)
    # - DCA_DIP: price is in the dip zone but no reclaim confirmation today
    dca_reclaim_ok = bool(dca_ok and (reclaim21 or reclaim200))
    dca_dip_ok     = bool(dca_ok and (not dca_reclaim_ok))


    # --- TRIM: only if NOT a fresh breakout (avoid blocking BUY) ---
    trim_ok = False
    if RULES.get("trim", {}).get("enabled", True) and trend_up and is_euphoria(r):
        trim_ok = True
        if np.isfinite(trigger) and (price <= trigger * 1.02):
            trim_ok = False

    if buy_ok:
        return "BUY"
    if dca_reclaim_ok:
        return "DCA_RECLAIM"
    if dca_dip_ok:
        return "DCA_DIP"
    if trim_ok:
        return "TRIM"
    if trend_up:
        return "HOLD"
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
    required = [
        "Price", "SMA200", "SMA50", "EMA21", "RSI14", "High20",
        "Vol20", "Volume", "Dist_to_SMA200_%", "Dist_to_52W_High_%",
        "Reclaim21", "Reclaim200",
    ]
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

    # IMPORTANT: keep this aligned with label_row() outputs.
    for s in ("BUY", "DCA_DIP", "DCA_RECLAIM", "TRIM"):
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


def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def backtest_investor_variant2(
    ind_map: dict,
    score_map: dict,
    bench_ind: pd.DataFrame = None,
    windows: dict = None,
):
    """Investor-mode backtest (Variant 2): start from cash; enter on signals; manage with TRIM/AVOID.

    Implementation notes:
    - Trades execute on next day's Open (simple + avoids same-bar lookahead).
    - Uses the dashboard's *label_row* signals (no pattern-based auto-upgrades to avoid lookahead).
    - Volatility-targeted sizing: risk-budget per position based on ATR (dollar-risk targeting).
    - DCA adds are capped by BACKTEST_DCA_MAX_LOTS.
    - Very simple friction model via BACKTEST_FEE_BPS.
    """
    if windows is None:
        windows = BACKTEST_WINDOWS

    # Build per-ticker lookup frames indexed by Date
    data = {}
    for t, df in (ind_map or {}).items():
        if df is None or df.empty:
            continue
        dfx = df.copy()
        if "Date" not in dfx.columns:
            continue
        dfx = dfx.sort_values("Date").drop_duplicates("Date")
        dfx = dfx.set_index("Date")
        # Ensure required fields exist
        for c in ["Open", "Close", "Price"]:
            if c not in dfx.columns:
                # fall back to Close for Price if needed
                if c == "Price" and "Close" in dfx.columns:
                    dfx[c] = dfx["Close"]
        data[t] = dfx

    if not data:
        return None

    # Determine a master calendar (prefer benchmark dates if available)
    if bench_ind is not None and (not bench_ind.empty) and ("Date" in bench_ind.columns):
        cal = bench_ind[["Date", "Price"]].dropna().sort_values("Date")
        cal = cal.drop_duplicates("Date").set_index("Date")
        dates_all = list(cal.index)
    else:
        # union of all dates
        dates_all = sorted(set().union(*[set(df.index) for df in data.values()]))

    if len(dates_all) < 10:
        return None

    # Precompute signal series for each ticker (vectorized-ish)
    sigs = {}
    for t, df in data.items():
        try:
            # label_row expects a Series; apply on each row
            sigs[t] = df.apply(label_row, axis=1)
        except Exception:
            continue

    # Rank helper
    pri = {"BUY": 3, "DCA_RECLAIM": 2, "DCA_DIP": 1}
    def _rank(t, sig, asof_row):
        s = pri.get(str(sig), 0)
        rs = float(asof_row.get("RS_3M_%", 0.0)) if asof_row is not None else 0.0
        sc = float((score_map or {}).get(t, 0.0))
        return (s, rs, sc)

    fee = float(BACKTEST_FEE_BPS) / 10000.0

    out = {"windows": {}, "meta": {"mode": "Investor Variant 2", "max_positions": int(BACKTEST_MAX_POSITIONS)}}

    # Benchmark stats per window
    bench_stats = {}

    for w_name, n_days in windows.items():
        try:
            n_days = int(n_days)
        except Exception:
            continue
        if n_days < 5:
            continue

        # Use the last n_days+1 bars so we can trade on next open
        dates = dates_all[-(n_days + 2):]
        if len(dates) < 10:
            continue

        equity0 = 100000.0
        cash = equity0
        positions = {}  # t -> dict(shares, lots, avg_cost)
        trade_wins = 0
        trade_count = 0
        exposure_acc = 0.0
        exposure_n = 0

        eq_series = []

        for i in range(len(dates) - 1):
            d = dates[i]
            d_next = dates[i + 1]

            # Mark-to-market at close on d
            equity = cash
            invested = 0.0
            for t, pos in list(positions.items()):
                df = data.get(t)
                if df is None or d not in df.index:
                    continue
                px = float(df.loc[d].get("Close", df.loc[d].get("Price", np.nan)))
                if not np.isfinite(px):
                    continue
                invested += pos["shares"] * px
            equity += invested
            eq_series.append((d, equity))

            exposure_acc += (invested / equity) if equity > 0 else 0.0
            exposure_n += 1

            # Signals on d
            exit_list = []
            trim_list = []
            add_list = []
            entry_cands = []

            # Existing positions: decide exits/trims/adds
            for t in list(positions.keys()):
                df = data.get(t)
                if df is None or d not in df.index:
                    continue
                sig = str(sigs.get(t, pd.Series()).get(d, "WATCH"))
                if sig == "AVOID":
                    exit_list.append(t)
                elif sig == "TRIM":
                    trim_list.append(t)
                elif sig in ("DCA_DIP", "DCA_RECLAIM") and positions[t].get("lots", 1) < int(BACKTEST_DCA_MAX_LOTS):
                    add_list.append(t)

            # New entries
            for t, df in data.items():
                if t in positions:
                    continue
                if d not in df.index:
                    continue
                sig = str(sigs.get(t, pd.Series()).get(d, "WATCH"))
                if sig in ("BUY", "DCA_DIP", "DCA_RECLAIM"):
                    entry_cands.append((t, sig, df.loc[d]))

            # Execute exits at next open
            for t in exit_list:
                df = data.get(t)
                if df is None or d_next not in df.index:
                    continue
                open_px = float(df.loc[d_next].get("Open", np.nan))
                if not np.isfinite(open_px):
                    continue
                sell_px = open_px * (1.0 - fee)
                sh = positions[t]["shares"]
                cash += sh * sell_px
                # Win/loss vs avg_cost
                avg_cost = float(positions[t].get("avg_cost", open_px))
                pnl = (sell_px - avg_cost) / avg_cost if avg_cost > 0 else 0.0
                trade_count += 1
                trade_wins += 1 if pnl > 0 else 0
                positions.pop(t, None)

            # Execute trims at next open
            for t in trim_list:
                if t not in positions:
                    continue
                df = data.get(t)
                if df is None or d_next not in df.index:
                    continue
                open_px = float(df.loc[d_next].get("Open", np.nan))
                if not np.isfinite(open_px):
                    continue
                sell_px = open_px * (1.0 - fee)
                sh = positions[t]["shares"]
                sell_sh = sh * float(BACKTEST_TRIM_FRACTION)
                if sell_sh <= 0:
                    continue
                positions[t]["shares"] = sh - sell_sh
                cash += sell_sh * sell_px
                # keep avg_cost unchanged

            # Execute adds at next open (half-lot by default)
            for t in add_list:
                if t not in positions:
                    continue
                df = data.get(t)
                if df is None or d_next not in df.index:
                    continue
                open_px = float(df.loc[d_next].get("Open", np.nan))
                if not np.isfinite(open_px):
                    continue
                buy_px = open_px * (1.0 + fee)
                add_value = float(positions[t].get("shares", 0.0)) * open_px * float(BACKTEST_DCA_ADD_FRACTION)
                if cash < add_value:
                    continue
                add_sh = add_value / buy_px
                # update avg_cost
                old_sh = positions[t]["shares"]
                old_cost = float(positions[t].get("avg_cost", buy_px))
                new_sh = old_sh + add_sh
                new_cost = (old_sh * old_cost + add_sh * buy_px) / new_sh
                positions[t]["shares"] = new_sh
                positions[t]["avg_cost"] = new_cost
                positions[t]["lots"] = positions[t].get("lots", 1) + float(BACKTEST_DCA_ADD_FRACTION)
                cash -= add_value

            # Execute entries at next open (ranked) until capacity / cash
            slots_left = int(BACKTEST_MAX_POSITIONS) - len(positions)
            if slots_left > 0 and entry_cands:
                entry_cands.sort(key=lambda x: _rank(x[0], x[1], x[2]), reverse=True)
                for (t, sig, row) in entry_cands[:slots_left]:
                    df = data.get(t)
                    if df is None or d_next not in df.index:
                        continue

                    open_px = float(df.loc[d_next].get("Open", np.nan))
                    if not np.isfinite(open_px) or open_px <= 0:
                        continue
                    buy_px = open_px * (1.0 + fee)

                    # Risk-based sizing (ATR targeting)
                    try:
                        atr = float(df.loc[d].get("ATR14", np.nan))
                    except Exception:
                        atr = np.nan

                    if not np.isfinite(atr) or atr <= 0:
                        # Fallback: small starter position if ATR unavailable
                        pos_value = min(cash * 0.10, equity * float(BACKTEST_MAX_POS_VALUE_PCT))
                    else:
                        risk_dollars = equity * float(BACKTEST_RISK_PCT)
                        risk_per_share = atr * float(BACKTEST_STOP_ATR_MULT)
                        shares = risk_dollars / risk_per_share
                        pos_value = shares * buy_px
                        # concentration cap
                        pos_value = min(pos_value, equity * float(BACKTEST_MAX_POS_VALUE_PCT))

                    # Cash cap + buffer
                    cash_buffer = equity * float(BACKTEST_MIN_CASH_BUFFER_PCT)
                    max_afford = max(0.0, cash - cash_buffer)
                    if pos_value > max_afford:
                        pos_value = max_afford
                    if pos_value <= 0:
                        continue

                    sh = pos_value / buy_px
                    positions[t] = {"shares": sh, "avg_cost": buy_px, "lots": 1.0, "entry_sig": str(sig)}
                    cash -= pos_value

# Final equity at last close
        d_last = dates[-1]
        equity = cash
        for t, pos in positions.items():
            df = data.get(t)
            if df is None or d_last not in df.index:
                continue
            px = float(df.loc[d_last].get("Close", df.loc[d_last].get("Price", np.nan)))
            if not np.isfinite(px):
                continue
            equity += pos["shares"] * px
        eq_series.append((d_last, equity))

        eq = pd.Series([v for _, v in eq_series], index=[d for d, _ in eq_series])
        ret = (eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else 0.0
        mdd = _max_drawdown(eq)
        win_rate = (trade_wins / trade_count) if trade_count > 0 else np.nan
        avg_exposure = (exposure_acc / exposure_n) if exposure_n > 0 else 0.0

        out["windows"][w_name] = {
            "return_pct": round(ret * 100.0, 2),
            "max_drawdown_pct": round(mdd * 100.0, 2),
            "trades": int(trade_count),
            "win_rate_pct": (round(win_rate * 100.0, 1) if np.isfinite(win_rate) else None),
            "avg_invested_pct": round(avg_exposure * 100.0, 1),
        }

        # Benchmark stats
        if bench_ind is not None and (not bench_ind.empty) and ("Date" in bench_ind.columns) and ("Price" in bench_ind.columns):
            b = bench_ind[["Date", "Price"]].dropna().sort_values("Date")
            b = b.drop_duplicates("Date").set_index("Date")
            b = b.loc[(b.index >= dates[0]) & (b.index <= dates[-1])]
            if len(b) > 5:
                bret = b["Price"].iloc[-1] / b["Price"].iloc[0] - 1.0
                bmdd = _max_drawdown(b["Price"])
                bench_stats[w_name] = {
                    "return_pct": round(bret * 100.0, 2),
                    "max_drawdown_pct": round(bmdd * 100.0, 2),
                }

    out["benchmark"] = bench_stats
    return out


def render_backtest_block(bt: dict, bench_symbol: str = "") -> str:
    """Render an investor-style *portfolio simulation* summary.

    IMPORTANT:
    - Return is the total portfolio % change over the window (cash included).
    - Trade win% is the share of CLOSED exits that were profitable.
      It is NOT a 'success probability' for the whole strategy.
    """
    if not bt or not bt.get("windows"):
        return ""

    meta = bt.get("meta", {}) or {}
    max_pos = int(meta.get("max_positions", BACKTEST_MAX_POSITIONS))
    fee_bps = float(BACKTEST_FEE_BPS)

    rows = ""
    for w, s in bt.get("windows", {}).items():
        b = (bt.get("benchmark", {}) or {}).get(w, {})
        btxt = ""
        if b:
            btxt = (
                f"<div class='muted' style='font-size:12px'>"
                f"Benchmark ({bench_symbol}): {b.get('return_pct','')}% return, {b.get('max_drawdown_pct','')}% maxDD"
                f"</div><div class=\"chart-key-sub\">EMA21 hugs candles · 200DMA is long-term trend</div></div>"
            )

        win = s.get("win_rate_pct")
        win_txt = "—" if win is None else f"{win}%"

        rows += f"""<tr>
            <td class='mono'>{w}</td>
            <td class='mono'>{s.get('return_pct','')}%</td>
            <td class='mono'>{s.get('max_drawdown_pct','')}%</td>
            <td class='mono'>{s.get('trades','')}</td>
            <td class='mono'>{win_txt}</td>
            <td class='mono'>{s.get('avg_invested_pct','')}%</td>
        </tr>""" + f"<tr><td colspan='6' style='padding-top:0;border-top:none'>{btxt}</td></tr>"

    return f"""<div class='card' style='margin-top:10px'>
        <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap'>
            <div>
                <div style='font-weight:700'>Simulation (Investor Variant 2)</div>
                <div class='muted' style='font-size:12px;margin-top:2px'>
                    Start from cash. Buy next open on <span class='mono'>BUY / DCA Dip / DCA Reclaim</span> · trim on <span class='mono'>TRIM</span> · exit on <span class='mono'>AVOID</span>.
                </div>
            </div>
            <div class='muted' style='font-size:12px'>Equal-lot · max {max_pos} positions · fee {fee_bps:.0f} bps/side</div>
        </div>
        <div class='table-responsive' style='margin-top:10px'>
        <table>
            <thead><tr>
                <th>Window</th><th>Return</th><th>Max DD</th><th>Closed trades</th><th>Trade win%</th><th>Avg invested</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
        <div class='muted' style='font-size:12px;margin-top:8px;line-height:1.35'>
            How to read: <span class='mono'>Return</span> is the portfolio % change over the window (cash included). <span class='mono'>Trade win%</span> is the share of completed exits that were profitable (not “chance of making money”). <span class='mono'>Avg invested</span> shows how much capital was deployed on average.
        </div>
        <div class='muted' style='font-size:12px;margin-top:6px'>Note: excludes pattern-based upgrades to avoid lookahead bias; uses current signal rules on historical prices.</div>
    </div>"""

def _pivots(ind, window=PIVOT_WINDOW):
    v = ind.tail(PATTERN_LOOKBACK).reset_index(drop=True).copy()
    v["PH"] = (v["High"] == v["High"].rolling(window * 2 + 1, center=True).max()).fillna(False)
    v["PL"] = (v["Low"]  == v["Low"].rolling(window * 2 + 1, center=True).min()).fillna(False)
    return v

def _similar(a, b, tol=PRICE_TOL, atr_ref=None):
    """Volatility-adaptive similarity test.

    Primary: treat two pivot prices as 'similar' if their difference is small relative to recent ATR.
    Fallback: fixed % tolerance (PRICE_TOL) if ATR isn't available.
    """
    if atr_ref is None:
        try:
            atr_ref = globals().get("_SIM_ATR_REF", np.nan)
        except Exception:
            atr_ref = np.nan
    try:
        if atr_ref is not None and np.isfinite(float(atr_ref)):
            return abs(a - b) <= (SIMILAR_ATR_MULT * float(atr_ref))
    except Exception:
        pass
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
    """Returns True when the chart looks *extended* (a good time to think about trimming).

    This is NOT a prediction. It's a risk-management flag:
    - near highs
    - stretched vs the 200DMA and/or far above the 21EMA in ATR terms
    """
    try:
        price = float(r.get("Price", r.get("Close", np.nan)))
        d200  = float(r.get("Dist_to_SMA200_%", np.nan))
        d52   = float(r.get("Dist_to_52W_High_%", np.nan))
        rsi   = float(r.get("RSI14", np.nan))
        ema21 = float(r.get("EMA21", np.nan))
        atr   = float(r.get("ATR14", np.nan))

        z_ema = np.nan
        if np.isfinite(price) and np.isfinite(ema21) and np.isfinite(atr) and atr > 0:
            z_ema = (price - ema21) / atr

        # "Classic" euphoria: very stretched + overbought + near highs
        cond1 = (np.isfinite(d52) and d52 > -5) and (np.isfinite(d200) and d200 > 25) and (np.isfinite(rsi) and rsi >= 75)

        # "Vertical" euphoria: price far above 21EMA in ATR terms (often snaps back)
        cond2 = (np.isfinite(d52) and d52 > -2) and (np.isfinite(z_ema) and z_ema >= 3.0) and (np.isfinite(rsi) and rsi >= 70)

        return bool(cond1 or cond2)
    except Exception:
        return False


def comment_for_row(r: pd.Series):
    """
    Returns (summary_html, playbook_html)

    Goal: translate indicators into *specific actions* (levels + rules),
    without drowning the user in verbose commentary.
    """
    # --- Helpers ---
    def f(x, d=2):
        return f"{x:.{d}f}" if np.isfinite(x) else "—"

    sig_raw = str(r.get("Signal", "WATCH")).upper()
    sig = sig_raw
    cat   = str(r.get("Category", "Core"))
    score = float(r.get("Fundy_Score", 0.0))
    tier  = str(r.get("Fundy_Tier", "Neutral"))

    price = float(r.get("Price", r.get("LastClose", np.nan)))
    rsi   = float(r.get("RSI14", np.nan))
    d200  = float(r.get("Dist_to_SMA200_%", np.nan))
    d52   = float(r.get("Dist_to_52W_High_%", np.nan))

    sma200 = float(r.get("SMA200", np.nan))
    sma50  = float(r.get("SMA50", np.nan))
    ema21  = float(r.get("EMA21", np.nan))
    slope200 = float(r.get("SMA200_Slope_%", 0.0))

    high20 = float(r.get("High20", np.nan))
    high52 = float(r.get("High52W", np.nan))

    atr   = float(r.get("ATR14", np.nan))
    atrp  = float(r.get("ATR14_%", np.nan))

    rs3m  = float(r.get("RS_3M_%", np.nan))
    rsslp = float(r.get("RS_Slope20_%", np.nan))
    bench = str(r.get("Benchmark", "") or "")
    market_up = bool(r.get("Market_Uptrend", True))
    dca_mode = str(r.get("DCA_Mode", "200") or "200").upper()

    fundy = r.get("Fundy", {}) or {}
    key_metric = str(fundy.get("key_metric", "-"))

    # --- Stage / regime classification (simple Weinstein-style) ---
    stage = "Stage 1 (Base)"
    if np.isfinite(d200) and np.isfinite(slope200):
        if (d200 > 5.0) and (slope200 > 0):
            stage = "Stage 2 (Uptrend)"
        elif (d200 < -5.0) and (slope200 < 0):
            stage = "Stage 4 (Downtrend)"
        elif (slope200 < 0) or (slope200 > 0 and d200 < 0):
            stage = "Stage 3 (Transition)"

    # --- Display label + emoji ---
    sig_disp = {
        "DCA_DIP": "DCA Dip",
        "DCA_RECLAIM": "DCA Reclaim",
    }.get(sig, sig)
    emoji = {
        "BUY": "🟢",
        "DCA_DIP": "🧱",
        "DCA_RECLAIM": "🧱",
        "HOLD": "🟦",
        "TRIM": "🟡",
        "WATCH": "⚪",
        "AVOID": "🔴",
    }.get(sig, "⚪")

    # --- Key levels (used in action line + bullets) ---
    buy_buf = float(RULES.get("buy", {}).get("buffer_pct", 0.0))
    buy_trigger = high20 * (1.0 + buy_buf) if np.isfinite(high20) else np.nan

    allow_below = float(RULES.get("dca", {}).get("allow_below_pct", 0.02))
    prox = float(RULES.get("dca", {}).get("sma200_proximity", 0.04))
    dca_low  = sma200 * (1.0 - allow_below) if np.isfinite(sma200) else np.nan
    dca_high = sma200 * (1.0 + prox)        if np.isfinite(sma200) else np.nan

    # Stops (rule-of-thumb; NOT a guarantee)
    atr_stop_mult  = float(RULES.get("risk", {}).get("atr_stop_mult", 2.0))
    atr_trail_mult = float(RULES.get("risk", {}).get("atr_trail_mult", 3.0))
    break200 = float(RULES.get("risk", {}).get("sma200_break_pct", 0.03))

    buy_stop = (buy_trigger - atr_stop_mult * atr) if (np.isfinite(buy_trigger) and np.isfinite(atr)) else np.nan

    dca_stop = np.nan
    if np.isfinite(dca_low):
        dca_stop = dca_low - (0.5 * atr if np.isfinite(atr) else 0.0)
    if np.isfinite(sma200):
        dca_stop = np.nanmin([dca_stop, sma200 * (1.0 - break200)]) if np.isfinite(dca_stop) else sma200 * (1.0 - break200)

    trail_stop = np.nan
    if np.isfinite(ema21) and np.isfinite(atr):
        trail_stop = ema21 - atr_trail_mult * atr
    if np.isfinite(sma50):
        trail_stop = np.nanmax([trail_stop, sma50 * (1.0 - break200)]) if np.isfinite(trail_stop) else sma50 * (1.0 - break200)

    # --- Market / leadership note (kept for logic, hidden in cards to reduce noise) ---
    market_note = ""

    # --- Action line (specific + numeric) ---
    action = "Wait; set alerts."
    if sig == "BUY":
        first_trim = (buy_trigger + 3 * atr) if (np.isfinite(buy_trigger) and np.isfinite(atr)) else np.nan
        action = f"Entry > {f(buy_trigger)} • Stop {f(buy_stop)} • First trim ~ {f(first_trim)}"
    elif sig in ("DCA_DIP", "DCA_RECLAIM"):
        if dca_mode == "AUTO":
            auto_mid, auto_stop = np.nan, np.nan
            ind0 = r.get("_ind", None)
            if isinstance(ind0, pd.DataFrame) and len(ind0) >= 3:
                D0 = ind0.iloc[-1]
                D1 = ind0.iloc[-2]
                hi = float(D1.get("HighI", D1.get("High", np.nan)))
                lo = float(D1.get("LowI", D1.get("Low", np.nan)))
                auto_mid = (hi + lo) / 2.0 if (np.isfinite(hi) and np.isfinite(lo)) else np.nan
                d0lo = float(D0.get("LowI", D0.get("Low", np.nan)))
                base_lo = np.nanmin([lo, d0lo]) if np.isfinite(d0lo) else lo
                auto_stop = base_lo - (0.5 * atr if np.isfinite(atr) else 0.0)
            action = f"Auto‑DCA: Starter > {f(auto_mid)} • Add ~ {f(ema21)} • Cut < {f(auto_stop)}"
        else:
            if sig == "DCA_RECLAIM":
                action = f"Reclaim > {f(ema21)} • Starter {f(dca_low)}–{f(dca_high)} • Cut if close < {f(dca_stop)}"
            else:
                action = f"Start zone {f(dca_low)}–{f(dca_high)} • Cut if close < {f(dca_stop)}"
    elif sig == "HOLD":
        action = f"Hold • Add: pullback ~ {f(ema21)} or breakout > {f(buy_trigger)} • Stop < {f(trail_stop)}"
    elif sig == "TRIM":
        action = f"Trim 20–33% • Trail stop < {f(trail_stop)}"
    elif sig == "AVOID":
        action = f"Avoid/exit • Re‑enter only after reclaim > 200DMA ({f(sma200)})"
    else:
        action = f"Alerts: breakout > {f(buy_trigger)} • Dip zone ~ 200DMA ({f(sma200)})"

    # --- Fundamental line (short) ---
    verified = bool(fundy.get("verified", True))
    if not verified:
        # Do not penalize signal — just flag fundamentals as unverified/stale
        fundy_line = f"{cat} • Fundamentals unverified"
        if np.isfinite(score) and score > 0:
            fundy_line += f" (last {score:.1f}/10)"
    else:
        fundy_line = f"{cat} • {score:.1f}/10 {tier}"
    if key_metric and key_metric != "-":
        fundy_line += f" • {key_metric}"

    # Quick flags (very short)
    quick_flags = []
    if bool(r.get("AutoDCA_Flag", False)): quick_flags.append("Auto‑DCA")
    if bool(r.get("BreakoutReady", False)): quick_flags.append("Breakout ready")

    pat_name = str(r.get("_pattern_name", "") or "")
    pat_status = str(r.get("_pattern_status", "") or "")
    pat_conf = float(r.get("_pattern_conf", 0.0) or 0.0)
    palign = str(r.get("_pattern_align", "") or "")
    if pat_name:
        if pat_status:
            quick_flags.append(f"{pat_name} ({pat_status}, {pat_conf:.2f})")
        else:
            quick_flags.append(pat_name)
    if palign:
        quick_flags.append(palign)

    flags_html = (" • " + " • ".join(quick_flags)) if quick_flags else ""

    sig_css = sig_raw.lower().replace("_", "-")

    summary = (
        f"<div class='c-action'>"
        f"<span class='c-sig c-{sig_css}'>{emoji} {sig_disp}</span>"
        f"<span class='c-act'>{action}</span>"
        f"</div>"
        f"<div class='c-meta'>{stage} · {fundy_line}{flags_html}</div>"
    )

    # --- Playbook (tight + readable) ---
    def pb(label: str, text: str) -> str:
        return (
            "<div class='pb-row'>"
            f"<span class='pb-lbl'>{label}</span>"
            f"<span class='pb-txt'>{text}</span>"
            "</div>"
        )

    rows = []

    # Context (one line)
    ctx = []
    if np.isfinite(rsi):  ctx.append(f"RSI {rsi:.0f}")
    if np.isfinite(d200): ctx.append(f"vs200 {d200:+.1f}%")
    if np.isfinite(atrp): ctx.append(f"ATR {atrp:.1f}%/day")
    if np.isfinite(rs3m): ctx.append(f"RS(3m) {rs3m:+.1f}% vs mkt")
    if ctx:
        rows.append(pb("Context", " · ".join(ctx)))

    if sig == "BUY":
        rows.append(pb("New", f"Enter only after a close > {f(buy_trigger)} (breakout)."))
        rows.append(pb("Add", f"Add on first pullback that holds EMA21 (~{f(ema21)})."))
        rows.append(pb("Risk", f"Cut if close < {f(buy_stop)} (failed breakout)."))

    elif sig in ("DCA_DIP", "DCA_RECLAIM"):
        if dca_mode == "AUTO":
            auto_mid, auto_stop = np.nan, np.nan
            ind0 = r.get("_ind", None)
            if isinstance(ind0, pd.DataFrame) and len(ind0) >= 3:
                D0 = ind0.iloc[-1]
                D1 = ind0.iloc[-2]
                hi = float(D1.get("HighI", D1.get("High", np.nan)))
                lo = float(D1.get("LowI", D1.get("Low", np.nan)))
                auto_mid = (hi + lo) / 2.0 if (np.isfinite(hi) and np.isfinite(lo)) else np.nan
                d0lo = float(D0.get("LowI", D0.get("Low", np.nan)))
                base_lo = np.nanmin([lo, d0lo]) if np.isfinite(d0lo) else lo
                auto_stop = base_lo - (0.5 * atr if np.isfinite(atr) else 0.0)

            rows.append(pb("New", f"Starter only if price reclaims > {f(auto_mid)} (Auto‑DCA midpoint)."))
            rows.append(pb("Add", f"Next add is first pullback toward EMA21 (~{f(ema21)})."))
            rows.append(pb("Risk", f"Cut if close < {f(auto_stop)} (reclaim failed)."))

        else:
            if sig == "DCA_RECLAIM":
                rows.append(pb("New", f"Starter only after reclaim > EMA21 (~{f(ema21)}) while holding 200DMA."))
                rows.append(pb("Add", f"Add on first pullback that respects EMA21 (~{f(ema21)})."))
                rows.append(pb("Risk", f"Cut if close < {f(dca_stop)} (reclaim failed / 200DMA broke)."))
            else:
                rows.append(pb("New", f"Starter buys in 200DMA zone {f(dca_low)}–{f(dca_high)} (scale in)."))
                rows.append(pb("If no dip", f"Don’t chase. Wait for reclaim > EMA21 (~{f(ema21)}) or breakout > {f(buy_trigger)}."))
                rows.append(pb("Risk", f"Cut if close < {f(dca_stop)} (200DMA break)."))

    elif sig == "HOLD":
        rows.append(pb("New", f"Wait. Buy only on (1) pullback to ~{f(ema21)} or (2) breakout > {f(buy_trigger)}."))
        rows.append(pb("If owned", f"Hold; trail risk under ~{f(trail_stop)}."))

    elif sig == "TRIM":
        rows.append(pb("If owned", "Trim 20–33% into strength; don’t add here."))
        rows.append(pb("Risk", f"Trail stop under ~{f(trail_stop)}; re‑add only after pullback toward ~{f(ema21)}."))

    elif sig == "AVOID":
        rows.append(pb("New", "No buys while the trend is broken / 200DMA is failing."))
        rows.append(pb("If owned", f"Reduce risk. Re‑enter only after reclaiming 200DMA (~{f(sma200)}) and stabilising."))

    else:
        rows.append(pb("Plan", f"Set alerts: breakout > {f(buy_trigger)} and dip near 200DMA (~{f(sma200)})."))

    playbook = "".join(rows)
    return summary, playbook




def mini_candle(ind, flag_info=None, pattern_lines=None):
    v = ind.tail(MINI_BARS).copy()

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=v["Date"],
                open=v["Open"],
                high=v["High"],
                low=v["Low"],
                close=v["Close"],
                increasing_line_color="rgba(74,222,128,0.9)",
                decreasing_line_color="rgba(248,113,113,0.9)",
                increasing_fillcolor="rgba(74,222,128,0.35)",
                decreasing_fillcolor="rgba(248,113,113,0.35)",
                line=dict(width=1),
            )
        ]
    )

    # Clean overlays only (avoid clutter)
    if "EMA21" in v.columns:
        fig.add_trace(
            go.Scatter(
                x=v["Date"],
                y=v["EMA21"],
                mode="lines",
                line=dict(color="rgba(56,189,248,0.85)", width=1),
            )
        )

    if "SMA50" in v.columns:
        fig.add_trace(
            go.Scatter(
                x=v["Date"],
                y=v["SMA50"],
                mode="lines",
                line=dict(color="rgba(203,213,225,0.55)", width=1, dash="dot"),
            )
        )
    if "SMA200" in v.columns:
        fig.add_trace(
            go.Scatter(
                x=v["Date"],
                y=v["SMA200"],
                mode="lines",
                line=dict(color="rgba(148,163,184,0.55)", width=1),
            )
        )

    # Optional: pattern overlays (OFF by default)
    if CHART_SHOW_FLAG_CHANNEL and flag_info:
        t2 = ind.tail(flag_info["win"])
        x = np.arange(len(t2))
        fig.add_trace(
            go.Scatter(
                x=t2["Date"],
                y=np.polyval(flag_info["hi"], x),
                mode="lines",
                line=dict(dash="dash", color="rgba(168,85,247,0.55)", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t2["Date"],
                y=np.polyval(flag_info["lo"], x),
                mode="lines",
                line=dict(dash="dash", color="rgba(168,85,247,0.55)", width=1),
            )
        )

    if CHART_SHOW_PATTERN_LINES and pattern_lines:
        for ln in pattern_lines[-2:]:
            if ln[0] == "h":
                fig.add_trace(
                    go.Scatter(
                        x=[ln[1], ln[2]],
                        y=[ln[3], ln[3]],
                        mode="lines",
                        line=dict(color="rgba(148,163,184,0.35)", width=1, dash="dash"),
                    )
                )
            elif ln[0] == "seg":
                fig.add_trace(
                    go.Scatter(
                        x=[ln[1], ln[3]],
                        y=[ln[2], ln[4]],
                        mode="lines",
                        line=dict(color="rgba(148,163,184,0.35)", width=1, dash="dash"),
                    )
                )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=145,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config={"displayModeBar": False, "staticPlot": True},
    )

def mini_spark(ind):
    v = ind.tail(SPARK_DAYS)
    fig = go.Figure(go.Scatter(x=v["Date"], y=v["Close"], mode="lines", line=dict(width=1, color="#94a3b8")))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=50, width=120, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "staticPlot": True})

def mini_rs_spark(ind):
    """Relative Strength sparkline vs benchmark (RS_Line), normalized to 100 at the start of the window.

    Only shown in Advanced mode to avoid clutter.
    """
    try:
        if ind is None or ind.empty or ("RS_Line" not in ind.columns):
            return ""
        v = ind.tail(SPARK_DAYS).copy()
        if v["RS_Line"].notna().sum() < 3:
            return ""
        rs = v["RS_Line"].astype(float).replace([np.inf, -np.inf], np.nan)
        base = rs.iloc[0]
        if not np.isfinite(base) or base == 0:
            return ""
        rsn = (rs / base) * 100.0

        fig = go.Figure(
            go.Scatter(
                x=v["Date"],
                y=rsn,
                mode="lines",
                line=dict(width=1, color="rgba(250,204,21,0.75)"),
            )
        )
        # baseline at 100
        fig.add_trace(
            go.Scatter(
                x=[v["Date"].iloc[0], v["Date"].iloc[-1]],
                y=[100, 100],
                mode="lines",
                line=dict(width=1, color="rgba(148,163,184,0.25)", dash="dash"),
            )
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=55,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
        return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "staticPlot": True})
    except Exception:
        return ""


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



def fetch_benchmark(m_code: str, m_conf: dict):
    """Fetch a market benchmark once per market.

    Returns:
      dict(symbol, ind, uptrend, asof)

    Notes:
    - GitHub Actions 'schedule' runs in UTC, but the dashboard itself can still use local market time.
    - If benchmark data fails, we default to 'uptrend=True' so we don't accidentally block everything.
    """
    if not (ENABLE_MARKET_REGIME_FILTER or ENABLE_RELATIVE_STRENGTH):
        return {"symbol": None, "ind": None, "uptrend": True, "asof": None}

    for sym in BENCHMARKS.get(m_code, []):
        try:
            dfb = fetch_prices(sym, m_conf["tz"])
            if dfb is None or dfb.empty:
                continue
            indb = indicators(dfb)
            if indb is None or indb.empty:
                continue
            last = indb.iloc[-1]
            p = float(last.get("Price", np.nan))
            s200 = float(last.get("SMA200", np.nan))
            slope = float(last.get("SMA200_Slope_%", 0.0))
            up = bool(np.isfinite(p) and np.isfinite(s200) and (p > s200) and (slope > 0))
            asof = last.get("Date", None)
            return {"symbol": sym, "ind": indb, "uptrend": up, "asof": asof}
        except Exception:
            continue

    return {"symbol": None, "ind": None, "uptrend": True, "asof": None}



def add_market_regime(ind: pd.DataFrame, bench_ind: pd.DataFrame) -> pd.DataFrame:
    """Adds Market_Uptrend to a stock dataframe based on benchmark regime.

    Market_Uptrend = benchmark Price > benchmark SMA200 AND benchmark SMA200 slope > 0.
    We merge by Date. If benchmark isn't available, we default to True (don't block signals).
    """
    if (not ENABLE_MARKET_REGIME_FILTER) or bench_ind is None or ind is None or ind.empty:
        # Keep column for downstream logic consistency
        if ind is not None and ("Market_Uptrend" not in ind.columns):
            ind = ind.copy()
            ind["Market_Uptrend"] = True
        return ind

    try:
        a = ind[["Date"]].copy()
        b = bench_ind[["Date", "Price", "SMA200", "SMA200_Slope_%"]].copy().rename(
            columns={"Price": "BenchPrice", "SMA200": "BenchSMA200", "SMA200_Slope_%": "BenchSMA200_Slope_%"}
        )
        m = a.merge(b, on="Date", how="left").sort_values("Date")
        up = (m["BenchPrice"] > m["BenchSMA200"]) & (m["BenchSMA200_Slope_%"] > 0)
        ind2 = ind.copy()
        ind2 = ind2.merge(m[["Date"]].assign(Market_Uptrend=up.fillna(True).astype(bool)), on="Date", how="left")
        ind2["Market_Uptrend"] = ind2["Market_Uptrend"].fillna(True).astype(bool)
        return ind2
    except Exception:
        ind2 = ind.copy()
        ind2["Market_Uptrend"] = True
        return ind2



def add_relative_strength(ind: pd.DataFrame, bench_ind: pd.DataFrame) -> pd.DataFrame:
    """Adds simple relative strength fields to a stock dataframe.

    We compute:
    - RS_Line        = Price / BenchPrice
    - RS_Slope20_%   = % change in RS_Line over RS_SLOPE_WINDOW
    - RS_3M_%        = (stock 3M return - benchmark 3M return) in percentage points

    If benchmark isn't available, we simply return the original dataframe.
    """
    if (not ENABLE_RELATIVE_STRENGTH) or bench_ind is None or ind is None or ind.empty:
        return ind

    try:
        a = ind[["Date", "Price"]].copy()
        b = bench_ind[["Date", "Price"]].copy().rename(columns={"Price": "BenchPrice"})
        m = a.merge(b, on="Date", how="inner").sort_values("Date")
        if m.empty:
            return ind

        rs_line = (m["Price"] / m["BenchPrice"]).replace([np.inf, -np.inf], np.nan)
        m["RS_Line"] = rs_line

        # RS slope (positive => improving leadership)
        m["RS_Slope20_%"] = (rs_line.diff(RS_SLOPE_WINDOW) / rs_line.shift(RS_SLOPE_WINDOW)).replace([np.inf, -np.inf], np.nan) * 100.0

        # Relative return (stock outperformance vs benchmark)
        m["RS_3M_%"] = (m["Price"].pct_change(RS_LOOKBACK_BARS) - m["BenchPrice"].pct_change(RS_LOOKBACK_BARS)).replace([np.inf, -np.inf], np.nan) * 100.0

        ind2 = ind.merge(m[["Date", "RS_Line", "RS_Slope20_%", "RS_3M_%"]], on="Date", how="left")
        return ind2
    except Exception:
        return ind

def process_market(m_code, m_conf):
    print(f"--> Analyzing {m_conf['name']}...")
    snaps = []
    frames = []
    ind_map = {}  # for backtest
    score_map = {}

    news_df = parse_announcements(m_code)

    # --- Market benchmark (regime + relative strength) ---
    bench = fetch_benchmark(m_code, m_conf)
    bench_ind = bench.get("ind", None)
    market_up = bool(bench.get("uptrend", True))
    m_conf["_bench"] = {
        "symbol": bench.get("symbol", ""),
        "uptrend": market_up,
        "asof": bench.get("asof", None),
    }
    if bench.get("symbol"):
        mood = "RISK-ON" if market_up else "RISK-OFF"
        print(f"    Benchmark: {bench['symbol']} → {mood}")


    # --- Price history prefetch (parallel) ---
    hist_map = {}
    if ENABLE_PARALLEL_PRICE_FETCH:
        tasks = {}
        with ThreadPoolExecutor(max_workers=int(PRICE_FETCH_MAX_WORKERS)) as ex:
            for _t_key, _t_meta in m_conf["tickers"].items():
                _raw = f"{_t_key}{m_conf['suffix']}"
                _dl = _raw.replace(".", "-") if (m_code == "USA" and "." in _raw) else _raw
                tasks[ex.submit(fetch_prices, _dl, m_conf.get("tz", "Australia/Sydney"))] = _t_key
            for fut in as_completed(tasks):
                _t_key = tasks[fut]
                try:
                    hist_map[_t_key] = fut.result()
                except Exception:
                    hist_map[_t_key] = pd.DataFrame()
    else:
        for _t_key, _t_meta in m_conf["tickers"].items():
            _raw = f"{_t_key}{m_conf['suffix']}"
            _dl = _raw.replace(".", "-") if (m_code == "USA" and "." in _raw) else _raw
            hist_map[_t_key] = fetch_prices(_dl, m_conf.get("tz", "Australia/Sydney"))

    for t_key, t_meta in m_conf["tickers"].items():
        t_name, t_desc, t_cat = t_meta

        raw_sym = f"{t_key}{m_conf['suffix']}"
        # Yahoo quirks: BRK.B => BRK-B, etc.
        dl_sym = raw_sym.replace(".", "-") if (m_code == "USA" and "." in raw_sym) else raw_sym

        df = hist_map.get(t_key)
        if df is None:
            df = fetch_prices(dl_sym, m_conf["tz"])
        if df.empty:
            continue

        df["Ticker"] = t_key
        frames.append(df)

        fundy = fetch_dynamic_fundamentals(dl_sym, t_cat)

        # Indicators (daily bars only)
        ind = indicators(df)
        ind = add_market_regime(ind, bench_ind)
        ind = add_relative_strength(ind, bench_ind)
        ind = ind.dropna(subset=["High20", "RSI14", "EMA21", "Vol20", "ATR14", "SMA200", "SMA50", "Price"])
        if ind.empty:
            continue
        if ENABLE_BACKTEST_REPORT:
            ind_map[t_key] = ind.copy()
            score_map[t_key] = float(fundy.get("score", 0.0))

        last = ind.iloc[-1]
        sig = label_row(last)
        sig_auto = False

        # --- Flag detection (bull flag) ---
        flag_flag, flag_det = detect_flag(ind)


        # Set ATR reference for volatility-adaptive pattern tolerance (median of recent ATR)
        try:
            globals()["_SIM_ATR_REF"] = float(np.nanmedian(ind["ATR14"].tail(60)))
        except Exception:
            globals()["_SIM_ATR_REF"] = np.nan

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
        # NOTE: handle split DCA labels too.
        if AUTO_UPGRADE_BREAKOUT and brk_ready:
            if brk_info.get("direction") == "bull" and (sig in ("WATCH", "HOLD", "TRIM") or str(sig).startswith("DCA")):
                sig = "BUY"
                sig_auto = True
            elif brk_info.get("direction") == "bear":
                sig = "AVOID"
                sig_auto = True

        # --- Auto-DCA gate (gap reclaim) ---
        gate_flag, gate_det = auto_dca_gate(ind)
        if gate_flag and sig in ("WATCH", "HOLD"):
            # Treat this as a reclaim-style DCA (confirmation day)
            sig = "DCA_RECLAIM"
            sig_auto = True

        # --- Pattern alignment ---
        pname = pats_display[0]["name"] if pats_display else (flag_pat["name"] if flag_pat else "")
        pbias = pattern_bias(pname)
        sig_str = str(sig).lower()

        # For alignment purposes, HOLD/TRIM are treated as "not fighting the tape"
        bullish_ok = (sig_str in ["buy", "hold", "trim", "watch"]) or sig_str.startswith("dca")
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


        # --- DCA quality gate (avoid averaging into weak businesses) ---
        if str(sig).startswith("DCA"):
            min_shield = MIN_SHIELD_FOR_DCA_CORE if t_cat == "Core" else MIN_SHIELD_FOR_DCA_GROWTH
            if float(fundy.get("score", 0.0)) < float(min_shield):
                # Downgrade rather than 'AVOID': the chart may be tradable, but not a "DCA-quality" business.
                p_now = float(last.get("Price", last.get("Close", np.nan)))
                s200_now = float(last.get("SMA200", np.nan))
                if np.isfinite(p_now) and np.isfinite(s200_now) and (p_now > s200_now):
                    sig = "HOLD"
                else:
                    sig = "WATCH"
                sig_auto = False

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

            # Market context (benchmark + relative strength)
            "Benchmark": str(m_conf.get("_bench", {}).get("symbol", "")),
            "Market_Uptrend": bool(m_conf.get("_bench", {}).get("uptrend", True)),
            "RS_3M_%": float(last.get("RS_3M_%", np.nan)),
            "RS_Slope20_%": float(last.get("RS_Slope20_%", np.nan)),

            "Signal": sig,
            "SignalAuto": bool(sig_auto),
            "DCA_Mode": (
                "AUTO" if (sig == "DCA_RECLAIM" and sig_auto and gate_flag)
                else ("200" if str(sig).startswith("DCA") else "")
            ),

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

    
    # --- Backtest (Investor Variant 2) ---
    if ENABLE_BACKTEST_REPORT and ind_map:
        try:
            m_conf["_backtest"] = backtest_investor_variant2(ind_map, score_map, bench_ind)
        except Exception:
            m_conf["_backtest"] = None
    else:
        m_conf["_backtest"] = None

    snaps_df = pd.DataFrame(snaps)

    if not snaps_df.empty:
        comments, playbooks, candles, sparks, rs_sparks = [], [], [], [], []

        for _, r in snaps_df.iterrows():
            summary, playbook = comment_for_row(r)
            # Append a quick "past results" check (Investor-mode sanity check, not a full backtest)
            lit = r.get("_litmus", {}) or {}
            if lit:
                # Show outcomes for the *relevant* signal, otherwise default to BUY.
                sig_now = str(r.get("Signal", "BUY")).upper()
                if sig_now.startswith("DCA"):
                    s = sig_now
                elif sig_now in ("BUY", "TRIM"):
                    s = sig_now
                else:
                    s = "BUY"

                n = lit.get(f"n_{s.lower()}", None)

                rows = []
                for h in LITMUS_SIGNAL_HORIZONS:
                    mret = lit.get(f"{s}_{h}d_med", np.nan)
                    hit = lit.get(f"{s}_{h}d_hit", np.nan)
                    if np.isfinite(mret) and np.isfinite(hit):
                        rows.append((int(h), float(mret), float(hit)))

                if rows:
                    # Past results → plain-English 'History' line (investor-friendly)
                    horizon_order = [20, 60, 5]
                    by_h = {int(h): (float(med), float(hit)) for h, med, hit in rows}

                    def fmt_one(h):
                        med, hit = by_h[h]
                        hl = {5: "~1 week", 20: "~1 month", 60: "~3 months"}.get(int(h), f"{int(h)}d")
                        return f"{hl}: typical {med:+.1f}%, higher {hit:.0f}%"

                    chosen = [h for h in horizon_order if h in by_h][:2]
                    details = " · ".join(fmt_one(h) for h in chosen) if chosen else ""

                    edge_h = 20 if 20 in by_h else (chosen[0] if chosen else None)
                    edge_txt = ""
                    if edge_h is not None:
                        med, hit = by_h[edge_h]
                        if hit >= 60 and med > 0:
                            edge_txt = "Supportive history"
                        elif hit <= 45 and med < 0:
                            edge_txt = "Weak history"
                        else:
                            edge_txt = "No clear edge"

                    n_txt = f" (n={int(n)})" if isinstance(n, (int, np.integer)) else ""
                    s_disp = {
                        "BUY": "BUY",
                        "TRIM": "TRIM",
                        "DCA_DIP": "DCA Dip",
                        "DCA_RECLAIM": "DCA Reclaim",
                    }.get(s, s)

                    playbook += (
                        "<div class='pb-row'>"
                        "<span class='pb-lbl'>History</span>"
                        f"<span class='pb-txt'>{edge_txt}{n_txt} — {s_disp}. {details}.</span>"
                        "</div>"
                    )

            # News (AUS only) — surface inside the comment so the card gets the 'News' tag
            if not news_df.empty:
                nd = news_df[(news_df["Ticker"] == r["Ticker"]) & (news_df["Recent"])]
                if not nd.empty:
                    headline = str(nd.iloc[-1]["Headline"])
                    summary += f" <span class='c-news'>News: {headline}</span>"
                    playbook += f"<div class='pb-row'><span class='pb-lbl'>News</span><span class='pb-txt'>{headline}</span></div>"

            comments.append(summary)
            playbooks.append(playbook)
            candles.append(mini_candle(r["_ind"], r["_flag_info"] if r["Flag"] else None, r["_pattern_lines"]))
            sparks.append(mini_spark(r["_ind"]))
            rs_sparks.append(mini_rs_spark(r["_ind"]))

        snaps_df["Comment"] = comments
        snaps_df["Playbook"] = playbooks
        snaps_df["_mini_candle"] = candles
        snaps_df["_mini_spark"] = sparks
        snaps_df["_rs_spark"] = rs_sparks

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
.comment-box { font-size: 13px; line-height: 1.5; color: #cbd5e1; margin-bottom: 12px; padding-top: 8px; border-top: 1px solid var(--border); }/* --- Cleaner commentary --- */
.c-action { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.c-sig { padding:3px 8px; border-radius:999px; font-weight:700; font-size:12px; letter-spacing:0.02em; color:white;
         background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.12); }
.c-sig.c-buy { background: rgba(34,197,94,0.18); border-color: rgba(34,197,94,0.30); }
.c-sig.c-dca-dip, .c-sig.c-dca-reclaim, .c-sig.c-dca { background: rgba(245,158,11,0.18); border-color: rgba(245,158,11,0.30); }
.c-sig.c-hold { background: rgba(59,130,246,0.18); border-color: rgba(59,130,246,0.30); }
.c-sig.c-trim { background: rgba(168,85,247,0.18); border-color: rgba(168,85,247,0.30); }
.c-sig.c-avoid { background: rgba(239,68,68,0.18); border-color: rgba(239,68,68,0.30); }
.c-sig.c-watch { background: rgba(255,255,255,0.10); border-color: rgba(255,255,255,0.12); }
.c-act { font-size:14px; line-height:1.35; color: rgba(255,255,255,0.92); }
.c-meta { margin-top:6px; font-size:12.5px; color: var(--text-muted); line-height:1.3; }
.c-news { margin-left:10px; padding:2px 6px; border-radius:8px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); }

.rs-block { margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.06); }
.rs-title { font-size:11px; color: var(--text-muted); margin: 0 0 2px 0; }
.why-list { margin: 4px 0 0 0; padding-left: 16px; color: rgba(226,232,240,0.95); }
.why-list li { margin: 2px 0; }
.playbook { margin-top: 10px; }
.pb-row { display:flex; gap:10px; align-items:flex-start; padding:6px 0; border-top: 1px solid rgba(255,255,255,0.06); }
.pb-row:first-child { border-top: none; padding-top:0; }
.pb-lbl { flex: 0 0 auto; min-width: 64px; font-size:12px; font-weight:700; letter-spacing:0.02em;
          color: rgba(255,255,255,0.75); }
.pb-txt { font-size:13.5px; line-height:1.35; color: rgba(255,255,255,0.90); }

.chart-container { margin-top: 10px; }
.chart-key { display:flex; gap:12px; align-items:center; margin-top:6px; font-size:12px; color: var(--text-muted); flex-wrap:wrap; }
.key-item { display:inline-flex; align-items:center; gap:6px; }
.sw { width:18px; height:0; border-top:2px solid rgba(255,255,255,0.35); display:inline-block; }
.sw-ema { border-top-color: rgba(56,189,248,0.85); }
.sw-50 { border-top-color: rgba(203,213,225,0.55); border-top-style:dotted; }
.sw-200 { border-top-color: rgba(148,163,184,0.65); }
.chart-key-wrap { margin-top: 8px; }
.chart-key-sub { font-size:11px; color: rgba(148,163,184,0.85); margin-top:4px; }

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

/* Mode toggles */
.mode-standard .advanced-only { display: none !important; }
.mode-advanced .advanced-only { display: block; }
.mode-advanced .standard-only { display: none !important; }
.mode-standard .standard-only { display: block; }

.topbar { display:flex; align-items:center; justify-content:space-between; gap:12px; padding: 10px 16px 6px 16px; }
.build-stamp { text-align:left; color:#94a3b8; font-family:'JetBrains Mono', monospace; font-size:12px; }
.mode-switch { display:flex; align-items:center; gap:10px; color:#cbd5e1; font-size:12px; }
.mode-label { opacity:0.9; }
.switch { position: relative; display:inline-block; width:44px; height:24px; }
.switch input { opacity:0; width:0; height:0; }
.slider { position:absolute; cursor:pointer; top:0; left:0; right:0; bottom:0; background: rgba(148,163,184,0.25); transition:.2s; border-radius: 999px; border: 1px solid rgba(255,255,255,0.10); }
.slider:before { position:absolute; content:""; height:18px; width:18px; left:3px; top:2px; background: rgba(255,255,255,0.85); transition:.2s; border-radius: 999px; }
input:checked + .slider { background: rgba(56,189,248,0.35); }
input:checked + .slider:before { transform: translateX(20px); background: rgba(255,255,255,0.95); }

.adv-details { margin-top: 10px; background: rgba(0,0,0,0.18); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 8px 10px; }
.adv-details > summary { cursor:pointer; list-style:none; color:#e2e8f0; font-weight:600; font-size:12px; }
.adv-details > summary::-webkit-details-marker { display:none; }
.adv-grid { margin-top: 8px; display:flex; flex-direction:column; gap:6px; }
.adv-row { display:flex; gap:10px; align-items:baseline; }
.adv-k { min-width: 80px; color: var(--text-muted); font-size:12px; }
.adv-v { color: var(--text-main); font-size:12px; }
.adv-v .mono { font-family:'JetBrains Mono', monospace; }

.chart-key .key-item { white-space: nowrap; }
.chart-key .key-item .note { opacity:0.7; font-size:11px; margin-left:4px; }

"""

JS = """
function switchMarket(code) {
    document.querySelectorAll('.market-container').forEach(el => el.classList.remove('active'));
    document.getElementById('cont-'+code).classList.add('active');
    document.querySelectorAll('.market-tab').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-'+code).classList.add('active');
}

function setMode(mode) {
    const body = document.body;
    body.classList.remove('mode-standard','mode-advanced');
    body.classList.add(mode === 'advanced' ? 'mode-advanced' : 'mode-standard');
    try { localStorage.setItem('tb_mode', mode); } catch(e) {}
    const cb = document.getElementById('modeToggle');
    if (cb) cb.checked = (mode === 'advanced');
}

function initMode() {
    let mode = 'standard';
    try { mode = localStorage.getItem('tb_mode') || 'standard'; } catch(e) {}
    setMode(mode);
    const cb = document.getElementById('modeToggle');
    if (cb) {
        cb.addEventListener('change', () => setMode(cb.checked ? 'advanced' : 'standard'));
    }
}

function init() {
    initMode();
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


def why_fired(r: pd.Series):
    """Return a short checklist explaining why the current Signal fired.

    This is intentionally 'rule aligned' with label_row(), so Advanced users can audit logic quickly.
    """
    try:
        sig = str(r.get("Signal", "WATCH")).upper()
        price  = float(r.get("Price", np.nan))
        sma200 = float(r.get("SMA200", np.nan))
        sma50  = float(r.get("SMA50", np.nan))
        ema21  = float(r.get("EMA21", np.nan))
        rsi    = float(r.get("RSI14", np.nan))
        high20 = float(r.get("High20", np.nan))
        vol    = float(r.get("Volume", np.nan))
        vol20  = float(r.get("Vol20", np.nan))
        sma200_slope = float(r.get("SMA200_Slope_%", 0.0))
        rs3m   = float(r.get("RS_3M_%", np.nan))
        rs_sl  = float(r.get("RS_Slope20_%", np.nan))
        market_up = bool(r.get("Market_Uptrend", True))
        reclaim21  = bool(r.get("Reclaim21", False))
        reclaim200 = bool(r.get("Reclaim200", False))

        lines = []
        death_cross = np.isfinite(sma50) and np.isfinite(sma200) and (sma50 < sma200)
        trend_up = (np.isfinite(price) and np.isfinite(sma200) and np.isfinite(sma50) and
                    (price > sma200) and (sma50 > sma200) and (sma200_slope > 0))

        # gates
        if ENABLE_MARKET_REGIME_FILTER and (str(MARKET_FILTER_MODE).lower() == "hard"):
            lines.append(f"✓ Market gate: {'PASS' if market_up else 'FAIL'}")
        if ENABLE_RELATIVE_STRENGTH and np.isfinite(rs3m):
            rs_ok = (rs3m >= RS_MIN_OUTPERF_PCT) and ((not np.isfinite(rs_sl)) or (rs_sl > 0))
            lines.append(f"✓ Leadership: {'PASS' if rs_ok else 'FAIL'} (RS(3m) {rs3m:+.1f}%, slope {rs_sl:+.1f}%)")

        if sig == "BUY":
            if trend_up:
                lines.append("✓ Trend up: price > 200DMA, 50DMA > 200DMA, 200DMA rising")
            if np.isfinite(high20):
                buffer_pct = float(RULES.get('buy', {}).get('buffer_pct', 0.0))
                trig = high20 * (1.0 + buffer_pct)
                lines.append(f"✓ Breakout: price {price:.2f} > 20D high trigger {trig:.2f}")
            if np.isfinite(vol20) and vol20 > 0 and np.isfinite(vol):
                vm = float(RULES.get('buy', {}).get('vol_mult', 1.0))
                lines.append(f"✓ Volume: {vol/vol20:.2f}× (need ≥ {vm:.2f}×)")
            if np.isfinite(rsi):
                lines.append(f"✓ RSI: {rsi:.0f} within buy range")

        elif sig in ("DCA_DIP", "DCA_RECLAIM"):
            if np.isfinite(sma200) and sma200_slope > 0:
                lines.append("✓ 200DMA rising (dip-buying only in uptrend)")
            if np.isfinite(price) and np.isfinite(sma200):
                prox = float(RULES['dca'].get('sma200_proximity', 0.04))
                allow_below = float(RULES['dca'].get('allow_below_pct', 0.02))
                lo = sma200 * (1.0 - allow_below)
                hi = sma200 * (1.0 + prox)
                lines.append(f"✓ Near 200DMA zone: {lo:.2f}–{hi:.2f}")
            if np.isfinite(rsi):
                lines.append(f"✓ RSI: {rsi:.0f} ≤ {float(RULES['dca'].get('rsi_max', 55)):.0f}")
            if death_cross:
                lines.append("✓ No death-cross: FAIL")
            else:
                lines.append("✓ No death-cross: PASS")

            if sig == "DCA_RECLAIM":
                if reclaim21:
                    lines.append(f"✓ Reclaim: closed back above EMA21 ({ema21:.2f})")
                if reclaim200:
                    lines.append(f"✓ Reclaim: closed back above 200DMA ({sma200:.2f})")
                if not (reclaim21 or reclaim200):
                    lines.append("✓ Reclaim signal: (none) — check rules")
            else:
                lines.append("✓ Dip day (no reclaim confirmation today)")

        elif sig == "TRIM":
            lines.append("✓ Uptrend intact but extended (take partials / tighten stop)")

        elif sig == "AVOID":
            if np.isfinite(price) and np.isfinite(sma200) and (price < sma200) and (sma200_slope < 0):
                lines.append("✓ Below falling 200DMA (stage-4)")
            if death_cross:
                lines.append("✓ Death cross (50DMA < 200DMA)")
            if np.isfinite(price) and np.isfinite(sma200) and price < sma200:
                lines.append("✓ Price below 200DMA")

        elif sig == "HOLD":
            if trend_up:
                lines.append("✓ Trend up, but no breakout/dip signal today")
            else:
                lines.append("✓ Holding zone (trend not clearly broken)")

        else:
            lines.append("✓ No strong edge (base / early / mixed signals)")

        # keep it short
        return lines[:7]
    except Exception:
        return []

def render_card(r, badge_type, curr):
    euphoria_cls = "euphoria-glow" if is_euphoria(r) else ""
    euphoria_tag = '<span class="badge" style="background:rgba(245,158,11,0.2);color:#fbbf24;margin-left:6px">Euphoria</span>' if is_euphoria(r) else ""
    news_tag = '<span class="badge news" style="margin-left:6px">News</span>' if "News:" in (r['Comment'] or "") else ""
    chart_key = ""
    if CHART_KEY_IN_CARD:
        def _fmt(x):
            try:
                return f"{float(x):.2f}" if np.isfinite(float(x)) else "—"
            except Exception:
                return "—"

        ema = r.get("EMA21", np.nan)
        s50 = r.get("SMA50", np.nan)
        s200 = r.get("SMA200", np.nan)

        chart_key = (
            f"<div class=\"chart-key-wrap\"><div class=\"chart-key\">"
            f"<span class='key-item'><span class='sw sw-ema'></span>EMA21 <span class='mono'>{_fmt(ema)}</span></span>"
            f"<span class='key-item'><span class='sw sw-50'></span>SMA50<span class='note'>(dot)</span> <span class='mono'>{_fmt(s50)}</span></span>"
            f"<span class='key-item'><span class='sw sw-200'></span>200DMA <span class='mono'>{_fmt(s200)}</span></span>"
            f"</div>"
            f"<div class='muted' style='font-size:11px;margin-top:2px'>EMA21 hugs the candles; 200DMA is the slow long-term line.</div>"
        )
    
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

    sig_txt = {
        "DCA_DIP": "DCA Dip",
        "DCA_RECLAIM": "DCA Reclaim",
    }.get(r.get("Signal"), r.get("Signal"))

    
    # Advanced (technical) block — shown only in Advanced mode
    def _p(x):
        try:
            x = float(x)
            return f"{curr}{x:.2f}" if np.isfinite(x) else "—"
        except Exception:
            return "—"

    def _n(x, suf=""):
        try:
            x = float(x)
            return f"{x:.2f}{suf}" if np.isfinite(x) else "—"
        except Exception:
            return "—"

    hi20 = r.get("High20", np.nan)
    hi52 = r.get("High52W", np.nan)
    lo52 = r.get("Low52W", np.nan)
    vol = r.get("Volume", np.nan)
    vol20 = r.get("Vol20", np.nan)
    vol_ratio = (float(vol) / float(vol20)) if (vol is not None and vol20 not in (None, 0) and np.isfinite(float(vol)) and np.isfinite(float(vol20)) and float(vol20) != 0) else np.nan

    
    why_lines = why_fired(r)
    why_html = ""
    if why_lines:
        items = "".join(f"<li>{w}</li>" for w in why_lines)
        why_html = f"<div class='adv-row'><span class='adv-k'>Why</span><span class='adv-v'><ul class='why-list'>{items}</ul></span></div>"
    advanced_html = (
        "<details class='adv-details advanced-only'>"
        "<summary>Advanced</summary>"
        "<div class='adv-grid'>" f"{why_html}"
        f"<div class='adv-row'><span class='adv-k'>Levels</span><span class='adv-v mono'>20D High {_p(hi20)} · 52W High {_p(hi52)} · 52W Low {_p(lo52)}</span></div>"
        f"<div class='adv-row'><span class='adv-k'>Volume</span><span class='adv-v mono'>Today {_n(vol/1e6,'M')} · Avg20 {_n(vol20/1e6,'M')} · Ratio {_n(vol_ratio,'x')}</span></div>"
        f"<div class='adv-row'><span class='adv-k'>Distances</span><span class='adv-v mono'>vs EMA21 {_n(r.get('Dist_EMA21_%', np.nan),'%')} · vs SMA50 {_n(r.get('Dist_SMA50_%', np.nan),'%')} · vs 200DMA {_n(r.get('Dist_to_SMA200_%', np.nan),'%')}</span></div>"
        f"<div class='adv-row'><span class='adv-k'>Risk</span><span class='adv-v mono'>ATR(14) {_n(r.get('ATR14', np.nan))} ({_n(r.get('ATR14_%', np.nan),'%/day')})</span></div>"
        "</div>"
        "</details>"
    )

    return f"""
    <div class="card searchable-item {euphoria_cls}">
        <div class="card-header">
            <div>
                <span class="ticker-badge mono">{r['Ticker']}</span>
                <span class="badge {badge_type}" style="margin-left:8px">{sig_txt}</span>
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
        {advanced_html}
        <div class="chart-container">{r['_mini_candle']}{chart_key}<div class="rs-block advanced-only"><div class="rs-title">RS vs market (normalized)</div>{r.get('_rs_spark','')}</div></div>
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
    DCA_DIP     = CORE[CORE.Signal == 'DCA_DIP'].sort_values(['Fundy_Score'], ascending=False)
    DCA_RECLAIM = CORE[CORE.Signal == 'DCA_RECLAIM'].sort_values(['Fundy_Score'], ascending=False)
    HOLD  = CORE[CORE.Signal == 'HOLD'].sort_values(['Fundy_Score'], ascending=False)
    TRIM  = CORE[CORE.Signal == 'TRIM'].sort_values(['Fundy_Score'], ascending=False)
    WATCH = CORE[CORE.Signal == 'WATCH'].sort_values(['Fundy_Score'], ascending=False)
    AVOID = CORE[CORE.Signal == 'AVOID'].sort_values(['Fundy_Score'], ascending=True)

    GATE  = CORE[CORE['AutoDCA_Flag']==True]
    PATS  = CORE[CORE['_pattern_name']!='']
    
    def mk_section(df, anchor, title, badge_cls):
        if df.empty:
            return ""
        h = f"<h2 id='{m_code}-{anchor}' style='color:var(--text-muted);margin-top:30px'>{title}</h2><div class='grid'>"
        for _, r in df.iterrows():
            h += render_card(r, badge_cls, m_conf['currency'])
        return h + "</div>"

    # Keep the legacy anchor "...-dca" for Dip so existing links don't break.
    html_cards = (
        mk_section(BUY, 'buy', 'BUY', 'buy')
        + mk_section(DCA_DIP, 'dca', 'DCA Dip', 'dca')
        + mk_section(DCA_RECLAIM, 'dca-reclaim', 'DCA Reclaim', 'dca')
        + mk_section(HOLD, 'hold', 'HOLD', 'hold')
        + mk_section(TRIM, 'trim', 'TRIM', 'trim')
        + mk_section(WATCH, 'watch', 'WATCH', 'watch')
        + mk_section(AVOID, 'avoid', 'AVOID', 'avoid')
    )

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
        {render_kpi('DCA Dip', len(DCA_DIP), 'text-amber')}
        {render_kpi('DCA Reclaim', len(DCA_RECLAIM), 'text-amber')}
        {render_kpi('Hold', len(HOLD), 'text-primary')}
        {render_kpi('Trim', len(TRIM), 'text-purple')}
        {render_kpi('Watch', len(WATCH), 'text-primary')}
        {render_kpi('Avoid', len(AVOID), 'text-red')}
        {render_kpi('Degens', len(DEGEN), 'text-purple')}
        {render_kpi('Auto-DCA', len(GATE), 'text-main')}
    </div>"""

    bench = m_conf.get("_bench", {}) or {}
    bench_sym = str(bench.get("symbol", "") or "")
    bench_up  = bool(bench.get("uptrend", True))
    bt_html = render_backtest_block(m_conf.get("_backtest"), bench_sym)
    # Market regime is used in logic, but we keep the UI clean (no benchmark banner).
    bench_line = ""

    bt_section = f"<h2 id='{m_code}-sim' style='margin-top:40px'>Simulation</h2>" + bt_html if bt_html else ""

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
            {bench_line}
            {kpi_html}
            {html_cards}
            
            <h2 id="{m_code}-degen" style="margin-top:40px">Degenerate Radar (Spec Only)</h2>
            <div class="card"><div class="table-responsive"><table><thead><tr><th>Ticker</th><th>Score</th><th>Sig</th><th>Metric</th><th>Spark</th></tr></thead><tbody>{degen_rows}</tbody></table></div><div class='bt-note'>Note: This simulation uses the current watchlist (survivorship bias) and simplified assumptions. It is a historical sanity-check, not a prediction.</div></div>
            
            <h2 id="{m_code}-gate" style="margin-top:40px">Auto-DCA Candidates</h2>
            <div class="card"><div class="table-responsive"><table><thead><tr><th>Ticker</th><th>Gap %</th><th>Reclaim?</th><th>EMA21?</th><th>Fill %</th><th>Trend</th></tr></thead><tbody>{gate_rows}</tbody></table></div></div>
            
            <h2 id="{m_code}-patterns" style="margin-top:40px">Patterns & Structures</h2>
            <div class="card"><div class="table-responsive"><table><thead><tr><th>Pattern</th><th>Ticker</th><th>Status</th><th>Conf</th><th>Align</th><th>Mini</th></tr></thead><tbody>{pat_rows}</tbody></table></div></div>
            
            <h2 id="{m_code}-news" style="margin-top:40px">News</h2>
            <div class="card" style="padding:0"><div class="table-responsive"><table><thead><tr><th>Date</th><th>Ticker</th><th>Type</th><th>Headline</th></tr></thead><tbody>{news_rows}</tbody></table></div></div>
            
            <h2 id="{m_code}-fundy" style="margin-top:40px">Deep Fundamentals</h2>
            <div class="card"><div class="table-responsive"><table><thead><tr><th>Ticker</th><th>Score</th><th>Key Metric</th><th>ROE</th><th>Debt/Eq</th><th>Cat</th></tr></thead><tbody>{f_rows}</tbody></table></div></div>
            
            {bt_section}
            <div style=\"height:50px\"></div>
        </div>
    </div>
    """

if __name__ == "__main__":
    print("Starting TraderBruh Global Hybrid v8.0")
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
    
    full = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>TraderBruh v7.3</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>{CSS}</style><script>{JS}</script></head><body class="mode-standard">
    <div class="topbar">
        <div class="build-stamp">Built: {gen_time} · Copyright @Amitesh</div>
        <div class="mode-switch">
            <span class="mode-label">Standard</span>
            <label class="switch" title="Toggle Standard/Advanced">
                <input type="checkbox" id="modeToggle">
                <span class="slider"></span>
            </label>
            <span class="mode-label">Advanced</span>
        </div>
    </div>
    <div class="market-tabs">{''.join(tab_buttons)}</div>{''.join(market_htmls)}</body></html>"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f: f.write(full)
    print("Done:", OUTPUT_HTML)