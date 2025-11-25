# traderbruh_global_hybrid.py
# TraderBruh — Global Web Dashboard (ASX / USA / INDIA)
# Version: Ultimate 4.2-hybrid (Multi-Market + Degen Radar + Full TA/Fundy Stack)

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

# --- Market Definitions with Categories (Core / Growth / Spec) ---
MARKETS = {
    "AUS": {
        "name": "Australia (ASX)",
        "tz": "Australia/Sydney",
        "currency": "A$",
        "suffix": ".AX",
        "tickers": {
            # -- WAAAX & Tech --
            "XRO": ("Xero", "Cloud accounting globally.", "Growth"),
            "WTC": ("WiseTech", "Logistics software (CargoWise).", "Growth"),
            "TNE": ("TechnologyOne", "Gov/Edu Enterprise SaaS.", "Core"),
            "NXT": ("NEXTDC", "Data Centers (AI Infrastructure).", "Growth"),
            "PME": ("Pro Medicus", "Radiology AI software.", "Growth"),
            "MP1": ("Megaport", "Network-as-a-Service.", "Spec"),
            "CDA": ("Codan", "Comms & Metal Detection.", "Core"),
            "HUB": ("HUB24", "Investment Platform.", "Growth"),
            "NWL": ("Netwealth", "Wealth Platform.", "Growth"),
            
            # -- Defense & Aero --
            "DRO": ("DroneShield", "Counter-UAS/Drone defense.", "Growth"),
            "EOS": ("Electro Optic", "Space & Defense systems.", "Spec"),
            "ASB": ("Austal", "Shipbuilding (US Navy).", "Core"),
            
            # -- Strategic Minerals (Lithium/Uranium/Rare Earths) --
            "PLS": ("Pilbara Minerals", "Hard-rock Lithium.", "Growth"),
            "MIN": ("Mineral Resources", "Mining services + Lithium.", "Growth"),
            "IGO": ("IGO Ltd", "Nickel & Lithium clean energy.", "Growth"),
            "LTR": ("Liontown", "Lithium developer.", "Spec"),
            "LYC": ("Lynas", "Rare Earths (Non-China supply).", "Growth"),
            "BOE": ("Boss Energy", "Uranium producer.", "Growth"),
            "PDN": ("Paladin", "Uranium (Namibia).", "Growth"),
            "DYL": ("Deep Yellow", "Uranium exploration.", "Spec"),
            
            # -- Healthcare & Bio --
            "CSL": ("CSL Limited", "Blood plasma & Vaccines.", "Core"),
            "COH": ("Cochlear", "Hearing implants.", "Core"),
            "RMD": ("ResMed", "Sleep apnea/Digital health.", "Core"),
            "TLX": ("Telix", "Radiopharmaceuticals.", "Growth"),
            "PNV": ("PolyNovo", "Synthetic skin/Wound care.", "Growth"),
            "NAN": ("Nanosonics", "Infection prevention.", "Growth"),
            "NEU": ("Neuren", "Neurodevelopmental drugs.", "Spec"),
            
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
        },
    },
    "USA": {
        "name": "United States (Wall St)",
        "tz": "America/New_York",
        "currency": "U$",
        "suffix": "",
        "tickers": {
            # -- The Magnificent 7 (AI & Big Tech) --
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

# ---------------- Helper Functions ----------------

def fetch_prices(symbol: str, tz_name: str) -> pd.DataFrame:
    """
    Fetch OHLCV history and stitch today's intraday bar if the local market is open.
    """
    try:
        df = yf.download(
            symbol,
            period=f"{FETCH_DAYS}d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            prepost=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        # Flatten multi-index if needed
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(symbol, axis=1, level=-1, drop_level=True)
            except Exception:
                df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df["Date"] = pd.to_datetime(df[date_col], utc=True)

        market_tz = zoneinfo.ZoneInfo(tz_name)
        now_mkt = datetime.now(market_tz)
        last_date_mkt = df["Date"].dt.tz_convert(market_tz).dt.date.max()

        # If market is active (after 10am local) and today's bar isn't present, approximate with 60m data
        if (now_mkt.time() >= time(10, 0)) and (last_date_mkt < now_mkt.date()):
            try:
                intr = yf.download(
                    symbol,
                    period="5d",
                    interval="60m",
                    auto_adjust=False,
                    progress=False,
                    prepost=False,
                    group_by="column",
                )
                if intr is not None and not intr.empty:
                    if isinstance(intr.columns, pd.MultiIndex):
                        try:
                            intr = intr.xs(symbol, axis=1, level=-1, drop_level=True)
                        except Exception:
                            intr.columns = intr.columns.get_level_values(0)
                    intr = intr.reset_index()
                    intr["Date"] = pd.to_datetime(intr[intr.columns[0]], utc=True)
                    last = intr.tail(1).iloc[0]
                    top = pd.DataFrame(
                        [
                            {
                                "Date": last["Date"],
                                "Open": float(last["Open"]),
                                "High": float(last["High"]),
                                "Low": float(last["Low"]),
                                "Close": float(last["Close"]),
                                "Volume": float(last["Volume"]),
                            }
                        ]
                    )
                    df = pd.concat([df, top], ignore_index=True)
            except Exception:
                pass

        # Convert to local tz then drop tz info for Plotly
        df["Date"] = df["Date"].dt.tz_convert(market_tz).dt.tz_localize(None)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def fetch_deep_fundamentals(symbol: str):
    """
    Buffett/Piotroski-style quality score on 0-10 scale.
    """
    try:
        tick = yf.Ticker(symbol)
        info = tick.info
        try:
            bs = tick.balance_sheet
            is_ = tick.income_stmt
            cf = tick.cashflow
        except Exception:
            bs, is_, cf = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        def get_item(df, item_names, idx=0):
            if df.empty:
                return 0
            for name in item_names:
                if name in df.index:
                    try:
                        return float(df.loc[name].iloc[idx])
                    except Exception:
                        return 0
            return 0

        def get_cagr(df, item_names, years=3):
            if df.empty or df.shape[1] < years:
                return 0
            curr = get_item(df, item_names, 0)
            past = get_item(df, item_names, years - 1)
            if past <= 0 or curr <= 0:
                return 0
            return (curr / past) ** (1 / (years - 1)) - 1

        # 1. Profitability (ROE over last ~3 years)
        roe_3y = 0
        try:
            if not is_.empty and not bs.empty:
                roes = []
                for i in range(min(3, len(is_.columns), len(bs.columns))):
                    ni = get_item(is_, ["Net Income"], i)
                    eq = get_item(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest"], i)
                    if eq > 0:
                        roes.append(ni / eq)
                if roes:
                    roe_3y = sum(roes) / len(roes)
        except Exception:
            pass

        ocf = get_item(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        net_inc = get_item(is_, ["Net Income"])
        high_quality_earnings = ocf > net_inc
        marg_curr = info.get("profitMargins", 0)

        score = 0.0
        if roe_3y > 0.15:
            score += 2
        elif roe_3y > 0.10:
            score += 1
        if marg_curr > 0.10:
            score += 1
        if high_quality_earnings:
            score += 0.5

        # 2. Balance Sheet
        curr_ratio = info.get("currentRatio", 0) or 0
        debt_eq = info.get("debtToEquity", 999)
        if debt_eq and debt_eq > 50:
            # Convert basis points-ish monstrous D/E to ratio
            debt_eq = debt_eq / 100.0

        cash = get_item(bs, ["Cash And Cash Equivalents", "Cash Financial"])
        lt_debt = get_item(bs, ["Long Term Debt"])

        if cash > lt_debt:
            score += 1.5
        elif debt_eq < 0.5:
            score += 1

        if curr_ratio > 1.5:
            score += 1
        elif curr_ratio > 1.1:
            score += 0.5

        # 3. Capital and buybacks
        shares_curr = get_item(bs, ["Share Issued", "Ordinary Shares Number"], 0)
        shares_old = get_item(bs, ["Share Issued", "Ordinary Shares Number"], 2)
        is_buyback = False
        if shares_old > 0:
            change = (shares_curr - shares_old) / shares_old
            if change < -0.01:
                score += 1.5
                is_buyback = True
            elif change < 0.05:
                score += 1

        # 4. Growth
        rev_cagr = get_cagr(is_, ["Total Revenue", "Operating Revenue"], 3)
        if rev_cagr > 0.10:
            score += 1

        peg = info.get("pegRatio", 0) or 0
        pe = info.get("trailingPE", 0) or 0
        if (peg and 0 < peg < 2.0) or (pe and 0 < pe < 20):
            score += 1

        score = min(score, 10)
        tier = "Fortress" if score >= 7 else ("Quality" if score >= 4 else "High-Risk")

        return {
            "score": round(score, 1),
            "tier": tier,
            "roe_3y": roe_3y,
            "margins": marg_curr,
            "debt_eq": debt_eq,
            "rev_cagr": rev_cagr,
            "is_buyback": is_buyback,
            "pe": pe,
            "cash": cash,
        }
    except Exception:
        return {
            "score": 0,
            "tier": "Error",
            "roe_3y": 0,
            "margins": 0,
            "debt_eq": 0,
            "rev_cagr": 0,
            "is_buyback": False,
            "pe": 0,
            "cash": 0,
        }


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

    # Proper True Range using previous close
    x["H-L"] = x["High"] - x["Low"]
    x["H-C"] = (x["High"] - x["Close"].shift(1)).abs()
    x["L-C"] = (x["Low"]  - x["Close"].shift(1)).abs()
    x["TR"]  = x[["H-L", "H-C", "L-C"]].max(axis=1)
    x["ATR14"] = x["TR"].rolling(14).mean()
    return x


def label_row(r: pd.Series) -> str:
    buy_ok = (
        (r["Close"] > r["SMA200"])
        and (r["Close"] > r["High20"])
        and (r["SMA50"] > r["SMA200"])
        and (RULES["buy"]["rsi_min"] <= r["RSI14"] <= RULES["buy"]["rsi_max"])
    )
    dca_ok = (
        (r["Close"] >= r["SMA200"])
        and (r["RSI14"] < RULES["dca"]["rsi_max"])
        and (r["Close"] <= r["SMA200"] * (1 + RULES["dca"]["sma200_proximity"]))
    )
    avoid = (r["SMA50"] < r["SMA200"]) if RULES["avoid"]["death_cross"] else False
    if buy_ok:
        return "BUY"
    if dca_ok:
        return "DCA"
    if avoid:
        return "AVOID"
    return "WATCH"


def auto_dca_gate(ind: pd.DataFrame):
    """
    Look for gap-down then recovery structures suitable for auto-DCA.
    """
    if len(ind) < 3:
        return False, {"reason": "insufficient data"}
    D0, D1, D2 = ind.iloc[-1], ind.iloc[-2], ind.iloc[-3]
    gap_pct = (D1["Open"] / D2["Close"] - 1) * 100.0
    if not np.isfinite(gap_pct) or gap_pct > RULES["autodca"]["gap_thresh"]:
        return False, {"reason": "no qualifying gap", "gap_pct": float(gap_pct)}
    gap_mid = (D1["High"] + D1["Low"]) / 2.0
    reclaim_mid = bool(D0["Close"] > gap_mid)
    above_ema21 = bool(D0["Close"] > D0["EMA21"])
    gap_size = max(D2["Close"] - D1["Open"], 0.0)
    fill_pct = float(
        0.0 if gap_size == 0 else (D0["Close"] - D1["Open"]) / gap_size * 100.0
    )
    filled50 = bool(fill_pct >= RULES["autodca"]["fill_req"])
    flag = reclaim_mid and above_ema21 and filled50
    return flag, {
        "gap_pct": float(gap_pct),
        "reclaim_mid": reclaim_mid,
        "above_ema21": above_ema21,
        "gap_fill_%": fill_pct,
    }


def _pivots(ind, window=PIVOT_WINDOW):
    v = ind.tail(PATTERN_LOOKBACK).reset_index(drop=True).copy()
    ph = v["High"] == v["High"].rolling(window * 2 + 1, center=True).max()
    pl = v["Low"]  == v["Low"].rolling(window * 2 + 1, center=True).min()
    v["PH"] = ph.fillna(False)
    v["PL"] = pl.fillna(False)
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
            if lj - li < 10:
                continue
            p1, p2 = float(v.loc[li, "Low"]), float(v.loc[lj, "Low"])
            if not _similar(p1, p2):
                continue
            neck = float(v.loc[li:lj, "High"].max())
            confirmed = bool(v["Close"].iloc[-1] > neck)
            conf = 0.6 + (0.2 if confirmed else 0.0)
            if (
                np.isfinite(v["Vol20"].iloc[-1])
                and confirmed
                and v["Volume"].iloc[-1] > 1.2 * v["Vol20"].iloc[-1]
            ):
                conf += 0.2
            lines = [
                ("h", v.loc[li, "Date"], v.loc[lj, "Date"], (p1 + p2) / 2.0),
                ("h", v.loc[li, "Date"], v["Date"].iloc[-1], neck),
            ]
            out.append(
                {
                    "name": "Double Bottom",
                    "status": "confirmed" if confirmed else "forming",
                    "confidence": round(min(conf, 1.0), 2),
                    "levels": {
                        "base": round((p1 + p2) / 2.0, 4),
                        "neckline": round(neck, 4),
                    },
                    "lines": lines,
                }
            )
            return out
    return out


def detect_double_top(ind):
    v = _pivots(ind)
    highs = v.index[v["PH"]].tolist()
    out = []
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            hi, hj = highs[i], highs[j]
            if hj - hi < 10:
                continue
            p1, p2 = float(v.loc[hi, "High"]), float(v.loc[hj, "High"])
            if not _similar(p1, p2):
                continue
            neck = float(v.loc[hi:hj, "Low"].min())
            confirmed = bool(v["Close"].iloc[-1] < neck)
            conf = 0.6 + (0.2 if confirmed else 0.0)
            if (
                np.isfinite(v["Vol20"].iloc[-1])
                and confirmed
                and v["Volume"].iloc[-1] > 1.2 * v["Vol20"].iloc[-1]
            ):
                conf += 0.2
            lines = [
                ("h", v.loc[hi, "Date"], v.loc[hj, "Date"], (p1 + p2) / 2.0),
                ("h", v.loc[hi, "Date"], v["Date"].iloc[-1], neck),
            ]
            out.append(
                {
                    "name": "Double Top",
                    "status": "confirmed" if confirmed else "forming",
                    "confidence": round(min(conf, 1.0), 2),
                    "levels": {
                        "ceiling": round((p1 + p2) / 2.0, 4),
                        "neckline": round(neck, 4),
                    },
                    "lines": lines,
                }
            )
            return out
    return out


def detect_inverse_hs(ind):
    v = _pivots(ind)
    lows = v.index[v["PL"]].tolist()
    out = []
    for i in range(len(lows) - 2):
        l1, h, l2 = lows[i], lows[i + 1], lows[i + 2]
        pL1, pH, pL2 = float(v.loc[l1, "Low"]), float(v.loc[h, "Low"]), float(
            v.loc[l2, "Low"]
        )
        if not (pH < pL1 * (1 - 0.04) and pH < pL2 * (1 - 0.04)):
            continue
        if not _similar(pL1, pL2):
            continue
        left_high = float(v.loc[l1:h, "High"].max())
        right_high = float(v.loc[h:l2, "High"].max())
        confirmed = bool(v["Close"].iloc[-1] > min(left_high, right_high))
        conf = 0.6 + (0.2 if confirmed else 0.0)
        lines = [
            ("seg", v.loc[l1, "Date"], left_high, v.loc[l2, "Date"], right_high),
        ]
        out.append(
            {
                "name": "Inverse H&S",
                "status": "confirmed" if confirmed else "forming",
                "confidence": round(min(conf, 1.0), 2),
                "levels": {
                    "neck_left": round(left_high, 4),
                    "neck_right": round(right_high, 4),
                },
                "lines": lines,
            }
        )
        return out
    return out


def detect_hs(ind):
    v = _pivots(ind)
    highs = v.index[v["PH"]].tolist()
    out = []
    for i in range(len(highs) - 2):
        l1, h, l2 = highs[i], highs[i + 1], highs[i + 2]
        pL1, pH, pL2 = float(v.loc[l1, "High"]), float(v.loc[h, "High"]), float(
            v.loc[l2, "High"]
        )
        if not (pH > pL1 * (1 + 0.04) and pH > pL2 * (1 + 0.04)):
            continue
        if not _similar(pL1, pL2):
            continue
        left_low = float(v.loc[l1:h, "Low"].min())
        right_low = float(v.loc[h:l2, "Low"].min())
        confirmed = bool(v["Close"].iloc[-1] < max(left_low, right_low))
        conf = 0.6 + (0.2 if confirmed else 0.0)
        lines = [
            ("seg", v.loc[l1, "Date"], left_low, v.loc[l2, "Date"], right_low),
        ]
        out.append(
            {
                "name": "Head & Shoulders",
                "status": "confirmed" if confirmed else "forming",
                "confidence": round(min(conf, 1.0), 2),
                "levels": {
                    "neck_left": round(left_low, 4),
                    "neck_right": round(right_low, 4),
                },
                "lines": lines,
            }
        )
        return out
    return out


def detect_triangles(ind):
    v = _pivots(ind)
    tail = v.tail(120).copy()
    phs = tail[tail["PH"]]
    pls = tail[tail["PL"]]
    out = []
    if len(phs) >= 2 and len(pls) >= 2:
        # Ascending
        ph_vals = phs["High"].values
        for i in range(len(ph_vals) - 1):
            if _similar(ph_vals[i], ph_vals[i + 1]):
                res = (ph_vals[i] + ph_vals[i + 1]) / 2.0
                slope = np.polyfit(np.arange(len(pls)), pls["Low"].values, 1)[0]
                if slope > 0:
                    confirmed = bool(tail["Close"].iloc[-1] > res)
                    conf = 0.55 + (0.25 if confirmed else 0.0)
                    lines = [
                        ("h", pls["Date"].iloc[0], tail["Date"].iloc[-1], res),
                        (
                            "seg",
                            pls["Date"].iloc[0],
                            pls["Low"].iloc[0],
                            pls["Date"].iloc[-1],
                            pls["Low"].iloc[-1],
                        ),
                    ]
                    out.append(
                        {
                            "name": "Ascending Triangle",
                            "status": "confirmed" if confirmed else "forming",
                            "confidence": round(min(conf, 1.0), 2),
                            "levels": {"resistance": round(res, 4)},
                            "lines": lines,
                        }
                    )
                    break
        # Descending
        pl_vals = pls["Low"].values
        for i in range(len(pl_vals) - 1):
            if _similar(pl_vals[i], pl_vals[i + 1]):
                sup = (pl_vals[i] + pl_vals[i + 1]) / 2.0
                slope = np.polyfit(np.arange(len(phs)), phs["High"].values, 1)[0]
                if slope < 0:
                    confirmed = bool(tail["Close"].iloc[-1] < sup)
                    conf = 0.55 + (0.25 if confirmed else 0.0)
                    lines = [
                        ("h", phs["Date"].iloc[0], tail["Date"].iloc[-1], sup),
                        (
                            "seg",
                            phs["Date"].iloc[0],
                            phs["High"].iloc[0],
                            phs["Date"].iloc[-1],
                            phs["High"].iloc[-1],
                        ),
                    ]
                    out.append(
                        {
                            "name": "Descending Triangle",
                            "status": "confirmed" if confirmed else "forming",
                            "confidence": round(min(conf, 1.0), 2),
                            "levels": {"support": round(sup, 4)},
                            "lines": lines,
                        }
                    )
                    break
    return out


def detect_flag(ind):
    if len(ind) < 60:
        return False, {}
    look = ind.tail(40)
    impulse = (look["Close"].max() / look["Close"].min() - 1) * 100
    if not np.isfinite(impulse) or impulse < 12:
        return False, {}
    win = 14
    tail = ind.tail(max(win, 8)).copy()
    x = np.arange(len(tail))
    hi = np.polyfit(x, tail["High"].values, 1)
    lo = np.polyfit(x, tail["Low"].values, 1)
    slope_pct = (hi[0] / tail["Close"].iloc[-1]) * 100
    ch = np.polyval(hi, x[-1]) - np.polyval(lo, x[-1])
    tight = ch <= max(
        0.4 * (look["Close"].max() - look["Close"].min()),
        0.02 * tail["Close"].iloc[-1],
    )
    gentle = (-0.006 <= slope_pct <= 0.002)
    return (tight and gentle), {"hi": hi.tolist(), "lo": lo.tolist(), "win": win}


def pattern_bias(name: str) -> str:
    if name in ("Double Bottom", "Inverse H&S", "Ascending Triangle", "Bull Flag"):
        return "bullish"
    if name in ("Double Top", "Head & Shoulders", "Descending Triangle"):
        return "bearish"
    return "neutral"


def breakout_ready_dt(ind: pd.DataFrame, pat: dict, rules: dict):
    """
    Breakout readiness for Double Top invalidation (short squeeze style).
    """
    if not pat or pat.get("name") != "Double Top":
        return False, {}
    last = ind.iloc[-1]
    atr = float(last.get("ATR14", np.nan))
    vol = float(last.get("Volume", np.nan))
    vol20 = float(last.get("Vol20", np.nan))
    ceiling = float(pat.get("levels", {}).get("ceiling", np.nan))
    if not (np.isfinite(atr) and np.isfinite(vol) and np.isfinite(vol20) and np.isfinite(ceiling)):
        return False, {}
    close = float(last["Close"])
    ok_price = (close >= ceiling * (1.0 + rules["buffer_pct"])) and (
        close >= ceiling + rules["atr_mult"] * atr
    )
    ok_vol = (vol20 > 0) and (vol >= rules["vol_mult"] * vol20)
    return bool(ok_price and ok_vol), {
        "ceiling": round(ceiling, 4),
        "atr": round(atr, 4),
        "stop": round(close - atr, 4),
    }

# ---------------- Commentary & Rendering Helpers ----------------

def is_euphoria(r: pd.Series) -> bool:
    return (
        (r["Dist_to_52W_High_%"] > -3.5)
        and (r["Dist_to_SMA200_%"] > 50.0)
        and (r["RSI14"] >= 70.0)
    )


def comment_for_row(r: pd.Series) -> str:
    """
    Core comment logic incorporating technical signal, fundamentals, and category.
    """
    d200 = r["Dist_to_SMA200_%"]
    d52 = r["Dist_to_52W_High_%"]
    rsi = r["RSI14"]
    sig = str(r.get("Signal", "")).upper()
    f_score = r.get("Fundy_Score", 0)
    cat = str(r.get("Category", "Core"))

    # Adjust nomenclature
    is_spec_bucket = (cat.lower() == "spec")

    # Matrix Logic (base)
    if sig == "BUY":
        if f_score >= 7:
            base = (
                "<b>CORE BUY (High Conviction):</b> Strong technical uptrend backed by "
                f"Fortress/Quality fundamentals ({f_score}/10)."
            )
        elif f_score <= 3:
            base = (
                "<b>SPECULATIVE BUY:</b> Uptrend in price, but weak fundamentals "
                f"({f_score}/10). Tight risk management only."
            )
        else:
            base = (
                "Standard Buy: Uptrend intact (close > 200DMA). "
                f"RSI {rsi:.0f} is constructive. Fundys OK ({f_score}/10)."
            )
    elif sig == "DCA":
        if f_score <= 3:
            base = (
                "<b>AVOID (Falling Knife):</b> Technicals suggest a dip-buy, but "
                f"fundamentals ({f_score}/10) are weak. High risk of value trap."
            )
        elif f_score >= 7:
            base = (
                "<b>QUALITY DIP (Accumulate):</b> Fortress balance sheet "
                f"({f_score}/10) near 200DMA. Attractive DCA zone."
            )
        else:
            base = (
                "DCA Zone: Trading near 200DMA (Δ "
                f"{d200:.1f}%) with cooling RSI. Decent risk/reward."
            )
    elif sig == "WATCH":
        if is_euphoria(r):
            if f_score <= 3:
                base = (
                    "<b>EXIT WARNING:</b> Weak fundamentals in euphoria zone "
                    f"({abs(d52):.1f}% off high). Consider de-risking."
                )
            else:
                base = (
                    "Euphoria Zone: Price extended. Quality hold, but trimming or "
                    "tighter trailing stops makes sense."
                )
        else:
            if f_score >= 7:
                base = (
                    "<b>GOLDEN WATCHLIST:</b> High quality "
                    f"({f_score}/10) near potential inflection. Wait for clean setup."
                )
            else:
                base = f"Watch: {abs(d52):.1f}% off highs. Momentum mixed."
    elif sig == "AVOID":
        if f_score >= 7:
            base = (
                "<b>VALUE WATCH:</b> Great business "
                f"({f_score}/10) in a downtrend. Do not catch falling knives; "
                "wait for basing and trend repair."
            )
        else:
            base = (
                f"Avoid: Weak trend (Δ 200DMA {d200:.1f}%) with weak fundamentals "
                f"({f_score}/10). Better opportunities elsewhere."
            )
    else:
        base = "Neutral."

    # Overlay: bucket = Spec -> explicitly label as Degenerate Radar candidate
    if is_spec_bucket:
        base += " <b>[Degenerate Radar / High-Risk Speculative]</b>"

    return base


def mini_candle(ind: pd.DataFrame, flag_info=None, pattern_lines=None) -> str:
    v = ind.tail(MINI_BARS).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=v["Date"],
            open=v["Open"],
            high=v["High"],
            low=v["Low"],
            close=v["Close"],
            hoverinfo="skip",
            showlegend=False,
            increasing_line_color="#4ade80",
            decreasing_line_color="#f87171",
        )
    )
    if "SMA20" in v.columns:
        fig.add_trace(
            go.Scatter(
                x=v["Date"],
                y=v["SMA20"],
                mode="lines",
                line=dict(width=1.4, color="rgba(56,189,248,0.8)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if flag_info:
        t2 = ind.tail(max(flag_info.get("win", 14), 8)).copy()
        x = np.arange(len(t2))
        hi = np.poly1d(flag_info["hi"])
        lo = np.poly1d(flag_info["lo"])
        for line_data in (hi(x), lo(x)):
            fig.add_trace(
                go.Scatter(
                    x=t2["Date"],
                    y=line_data,
                    mode="lines",
                    line=dict(width=2, dash="dash", color="rgba(167,139,250,0.95)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    def _expand(lines):
        out = []
        if not lines:
            return out
        for ln in lines:
            if ln[0] == "h":
                _, d_left, d_right, y = ln
                out.append(("h", d_left, y, d_right, y))
            else:
                _, d1, y1, d2, y2 = ln
                out.append(("seg", d1, y1, d2, y2))
        return out

    if pattern_lines:
        for (kind, x1, y1, x2, y2) in _expand(pattern_lines):
            c, d = (
                ("rgba(34,197,94,0.95)", "dot") if kind == "h" else ("rgba(234,179,8,0.95)", "solid")
            )
            fig.add_trace(
                go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode="lines",
                    line=dict(width=2, color=c, dash=d),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=130,
        width=280,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "staticPlot": True})


def mini_spark(ind: pd.DataFrame) -> str:
    spark_ind = ind.tail(SPARK_DAYS)
    fig = go.Figure(
        go.Scatter(
            x=spark_ind["Date"],
            y=spark_ind["Close"],
            mode="lines",
            line=dict(width=1, color="#94a3b8"),
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=50,
        width=120,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "staticPlot": True})

# ---------------- ASX Announcements Parsing ----------------

def parse_announcements(market_code: str) -> pd.DataFrame:
    """
    Only parse PDFs for ASX announcements (Appendix 3Y etc.).
    Other markets return an empty DataFrame.
    """
    if market_code != "AUS":
        return pd.DataFrame(columns=["Date", "Ticker", "Type", "Tag", "Headline", "Details", "Path", "Recent"])

    NEWS_TYPES_REGEX = [
        ("Appendix 3Y", r"Appendix\s*3Y|Change of Director.?s? Interest Notice", "director"),
        ("Appendix 2A", r"Appendix\s*2A|Application for quotation of securities", "issue"),
        ("Cleansing Notice", r"Cleansing Notice", "issue"),
        ("Price Query", r"Price Query|Aware Letter|Response to ASX Price Query", "reg"),
        ("Share Price Movement", r"Share Price Movement", "reg"),
        ("Trading Halt", r"Trading Halt", "reg"),
        ("Withdrawal", r"withdrawn", "reg"),
        ("Orders / Contracts", r"order|contract|sale|revenue|cash receipts", "ops"),
    ]

    def parse_3y_stats(text: str):
        act = "Disposed" if re.search(r"\bDisposed\b", text, re.I) else ("Acquired" if re.search(r"\bAcquired\b", text, re.I) else None)
        shares, value = None, None
        m = re.search(r"(\d{1,3}(?:,\d{3}){1,3})\s+(?:ordinary|fully\s+paid|shares)", text, re.I)
        if m:
            shares = m.group(1)
        v = re.search(r"\$ ?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)", text)
        if v:
            value = v.group(1)
        parts = [p for p in [act, f"{shares} shares" if shares else None, f"A${value}" if value else None] if p]
        return " • ".join(parts) if parts else None

    def read_pdf_first_text(path: str) -> str:
        if not HAVE_PYPDF:
            return ""
        try:
            txt = PdfReader(path).pages[0].extract_text() or ""
            return re.sub(r"[ \t]+", " ", txt)
        except Exception:
            return ""

    rows = []
    today_syd = datetime.now(zoneinfo.ZoneInfo("Australia/Sydney")).date()
    if not os.path.isdir(ANN_DIR):
        return pd.DataFrame(columns=["Date", "Ticker", "Type", "Tag", "Headline", "Details", "Path", "Recent"])

    for fp in sorted(glob.glob(os.path.join(ANN_DIR, "*.pdf"))):
        fname = os.path.basename(fp)
        m = re.match(r"([A-Z]{2,4})[_-]", fname)
        ticker = m.group(1) if m else None
        text = read_pdf_first_text(fp)
        combined = fname + " " + text
        _type, tag = next(
            ((label, t) for (label, patt, t) in NEWS_TYPES_REGEX if re.search(patt, combined, re.I)),
            ("Announcement", "gen"),
        )
        d = None
        md = re.search(r"(\d{1,2}\s+[A-Za-z]{3,9}\s+20\d{2})", text)
        if md:
            for fmt in ("%d %B %Y", "%d %b %Y"):
                try:
                    d = datetime.strptime(md.group(1), fmt)
                    break
                except Exception:
                    continue
        if d is None:
            d = datetime.fromtimestamp(os.path.getmtime(fp)).replace(tzinfo=None)
        details = parse_3y_stats(text) if _type == "Appendix 3Y" else None
        is_recent = (today_syd - d.date()).days <= NEWS_WINDOW_DAYS
        rows.append(
            {
                "Date": d.date().isoformat(),
                "Ticker": ticker or "",
                "Type": _type,
                "Tag": tag,
                "Headline": _type,
                "Details": details or "",
                "Path": fp,
                "Recent": is_recent,
            }
        )
    return pd.DataFrame(rows)

# ---------------- Analysis Pipeline ----------------

def process_market(market_code: str, market_conf: dict):
    print(f"--> Analyzing {market_conf['name']}...")
    snaps = []
    frames = []

    # 1. News
    news_df = parse_announcements(market_code)

    # 2. Main loop per ticker
    for t_key, t_meta in market_conf["tickers"].items():
        # Unpack category triple
        if len(t_meta) == 3:
            t_name, t_desc, t_cat = t_meta
        else:
            t_name, t_desc = t_meta
            t_cat = "Core"

        full_sym = f"{t_key}{market_conf['suffix']}"

        df = fetch_prices(full_sym, market_conf["tz"])
        if df.empty:
            continue
        df["Ticker"] = t_key
        frames.append(df)

        fundy = fetch_deep_fundamentals(full_sym)

    # REMOVED 'SMA200' and 'SMA50' from the mandatory list
        ind = indicators(df).dropna(subset=['High20', 'RSI14', 'EMA21', 'Vol20', 'ATR14'])
	
        if ind.empty:
            continue
        last = ind.iloc[-1]
        sig = label_row(last)

        # Patterns
        flag_flag, flag_det = detect_flag(ind)
        pats = (
            detect_double_bottom(ind)
            + detect_double_top(ind)
            + detect_inverse_hs(ind)
            + detect_hs(ind)
            + detect_triangles(ind)
        )
        if PATTERNS_CONFIRMED_ONLY:
            pats = [p for p in pats if p.get("status") == "confirmed"]

        breakout_ready, breakout_info = (False, {})
        dt_pats = [p for p in pats if p.get("name") == "Double Top"]
        if dt_pats:
            breakout_ready, breakout_info = breakout_ready_dt(ind, dt_pats[0], BREAKOUT_RULES)

        signal_auto = False
        if AUTO_UPGRADE_BREAKOUT and breakout_ready:
            sig = "BUY"
            signal_auto = True

        gate_flag, gate_det = auto_dca_gate(ind)

        pname = pats[0]["name"] if pats else ""
        palign = "ALIGNED"
        pbias = pattern_bias(pname)
        sig_str = str(sig).lower()
        if pbias == "bullish" and sig_str not in ("buy", "dca", "watch"):
            palign = "CONFLICT"
        elif pbias == "bearish" and sig_str not in ("avoid", "watch"):
            palign = "CONFLICT"

        snaps.append(
            {
                "Ticker": t_key,
                "Name": t_name,
                "Desc": t_desc,
                "Category": t_cat,
                "LastDate": pd.to_datetime(last["Date"]).strftime("%Y-%m-%d"),
                "LastClose": float(last["Close"]),
                "SMA20": float(last["SMA20"]),
                "SMA50": float(last["SMA50"]),
                "SMA200": float(last["SMA200"]),
                "RSI14": float(last["RSI14"]),
                "High52W": float(last["High52W"]),
                "Dist_to_52W_High_%": float(last["Dist_to_52W_High_%"]),
                "Dist_to_SMA200_%": float(last["Dist_to_SMA200_%"]),
                "Signal": sig,
                "SignalAuto": bool(signal_auto),
                "Comment": None,
                "Flag": bool(flag_flag),
                "_flag_info": flag_det,
                "_pattern_lines": pats[0]["lines"] if pats else None,
                "_pattern_name": pname,
                "_pattern_status": pats[0]["status"] if pats else "",
                "_pattern_conf": pats[0]["confidence"] if pats else np.nan,
                "_pattern_align": palign,
                "AutoDCA_Flag": bool(gate_flag),
                "AutoDCA_Gap_%": float(gate_det.get("gap_pct", np.nan)),
                "AutoDCA_ReclaimMid": bool(gate_det.get("reclaim_mid", False)),
                "AutoDCA_AboveEMA21": bool(gate_det.get("above_ema21", False)),
                "AutoDCA_Fill_%": float(gate_det.get("gap_fill_%", np.nan)),
                "BreakoutReady": bool(breakout_ready),
                "Breakout_Level": float(breakout_info.get("ceiling", np.nan)),
                "Breakout_Stop": float(breakout_info.get("stop", np.nan)),
                # Fundamentals
                "Fundy_Score": fundy["score"],
                "Fundy_Tier": fundy["tier"],
                "Fundy_ROE": fundy["roe_3y"],
                "Fundy_Margin": fundy["margins"],
                "Fundy_PE": fundy["pe"],
                "Fundy_RevCAGR": fundy["rev_cagr"],
                "Fundy_Growth": fundy["rev_cagr"],
                "Fundy_Cash": fundy["cash"],
                "Fundy_Debt": 0 if fundy["debt_eq"] == 999 else fundy["debt_eq"],
            }
        )

    prices_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    snaps_df = pd.DataFrame(snaps)

    rows = []
    if not snaps_df.empty:
        for _, r in snaps_df.iterrows():
            r = r.copy()
            t = r["Ticker"]
            df = prices_all[prices_all["Ticker"] == t].copy()
            ind = indicators(df)

            # Comment
            r["Comment"] = comment_for_row(r)
            if r.get("BreakoutReady", False):
                r["Comment"] += f" • BreakoutReady: cleared {r['Breakout_Level']:.2f}."
            if r.get("SignalAuto", False):
                r["Comment"] += " • Auto-upgraded (DT invalidation)."

            # News
            if not news_df.empty:
                nd = news_df[(news_df["Ticker"] == t) & (news_df["Recent"])]
                if not nd.empty:
                    top = nd.sort_values("Date").iloc[-1]
                    badge = (
                        "Director sale"
                        if top["Type"] == "Appendix 3Y"
                        and "Disposed" in (top["Details"] or "")
                        else top["Type"]
                    )
                    r["Comment"] += f" • News: {badge} ({top['Date']})"

            # Charts
            r["_mini_spark"] = mini_spark(ind)
            r["_mini_candle"] = mini_candle(
                ind,
                r["_flag_info"] if r["Flag"] else None,
                r["_pattern_lines"],
            )
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
    background: var(--bg);
    background-image:
        radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.1) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.1) 0px, transparent 50%);
    background-attachment: fixed;
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding-bottom: 60px;
    font-size: 14px;
}
.mono { font-family: 'JetBrains Mono', monospace; }
.text-green { color: var(--accent-green); }
.text-red { color: var(--accent-red); }
.text-amber { color: var(--accent-amber); }
.text-purple { color: var(--accent-purple); }
.text-primary { color: var(--primary); }
.hidden { display: none !important; }

/* Market Tabs */
.market-tabs {
    position: sticky;
    top: 0;
    z-index: 200;
    background: #020617;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: center;
    gap: 10px;
    padding: 10px;
}
.market-tab {
    background: transparent;
    border: 1px solid var(--text-muted);
    color: var(--text-muted);
    padding: 8px 20px;
    border-radius: 999px;
    cursor: pointer;
    font-weight: 600;
    transition: 0.2s;
}
.market-tab.active {
    background: var(--primary);
    border-color: var(--primary);
    color: white;
}

/* Internal Nav */
.nav-wrapper {
    position: sticky;
    top: 53px;
    z-index: 100;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 10px 16px;
}
.nav-inner {
    display: flex;
    align-items: center;
    gap: 12px;
    max-width: 1200px;
    margin: 0 auto;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
}
.nav-inner::-webkit-scrollbar { display: none; }
.nav-link {
    white-space: nowrap;
    color: var(--text-muted);
    text-decoration: none;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 500;
    background: rgba(255,255,255,0.03);
    border: 1px solid transparent;
    transition: all 0.2s;
}
.nav-link:hover, .nav-link.active {
    background: rgba(255,255,255,0.1);
    color: white;
    border-color: rgba(255,255,255,0.1);
}

.market-container { display: none; }
.market-container.active { display: block; animation: fadein 0.3s; }
@keyframes fadein { from { opacity: 0; } to { opacity: 1; } }

.search-container {
    max-width: 1200px;
    margin: 16px auto 0;
    padding: 0 16px;
}
.search-input {
    width: 100%;
    background: var(--glass);
    border: 1px solid var(--border);
    padding: 12px 16px;
    border-radius: 12px;
    color: white;
    font-family: 'Inter';
    font-size: 15px;
    outline: none;
    transition: border-color 0.2s;
}
.search-input:focus { border-color: var(--primary); }

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px 16px;
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
}
@media(max-width: 600px) {
    .grid { grid-template-columns: 1fr; }
}

.card {
    background: var(--glass);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
    overflow: hidden;
    position: relative;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 10px;
}
.ticker-badge {
    background: rgba(255,255,255,0.05);
    padding: 4px 8px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.5px;
    color: white;
    text-decoration: none;
    display: inline-block;
}
.price-block { text-align: right; }
.price-main { font-size: 18px; font-weight: 600; }
.price-sub { font-size: 11px; color: var(--text-muted); }

.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 12px;
    background: rgba(0,0,0,0.2);
    padding: 8px;
    border-radius: 8px;
}
.metric { display: flex; flex-direction: column; }
.metric label {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
}
.metric span { font-size: 13px; font-weight: 500; }

.comment-box {
    font-size: 13px;
    line-height: 1.5;
    color: #cbd5e1;
    margin-bottom: 12px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
}

.playbook {
    background: rgba(0,0,0,0.2);
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 13px;
    color: #e2e8f0;
    line-height: 1.6;
}
.playbook b { color: white; }

.badge {
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    display: inline-block;
}
.badge.buy {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-green);
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.badge.dca {
    background: rgba(245, 158, 11, 0.15);
    color: var(--accent-amber);
    border: 1px solid rgba(245, 158, 11, 0.2);
}
.badge.watch {
    background: rgba(59, 130, 246, 0.15);
    color: var(--primary);
    border: 1px solid rgba(59, 130, 246, 0.2);
}
.badge.avoid {
    background: rgba(239, 68, 68, 0.15);
    color: var(--accent-red);
    border: 1px solid rgba(239, 68, 68, 0.2);
}
.badge.news {
    background: rgba(168, 85, 247, 0.15);
    color: var(--accent-purple);
}

.badge.shield-high {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-green);
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.badge.shield-low {
    background: rgba(239, 68, 68, 0.15);
    color: var(--accent-red);
    border: 1px solid rgba(239, 68, 68, 0.2);
}

.euphoria-glow {
    box-shadow: 0 0 15px rgba(245, 158, 11, 0.25);
    border-color: rgba(245, 158, 11, 0.4);
}

.kpi-scroll {
    display: flex;
    gap: 12px;
    overflow-x: auto;
    padding-bottom: 8px;
    margin-bottom: 24px;
    scrollbar-width: none;
}
.kpi-scroll::-webkit-scrollbar { display: none; }
.kpi-card {
    min-width: 140px;
    background: var(--surface-1);
    border-radius: 12px;
    padding: 12px;
    border: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-val {
    font-size: 24px;
    font-weight: 700;
    line-height: 1;
    margin-top: 4px;
}
.kpi-lbl {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.table-responsive {
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid var(--border);
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
th {
    text-align: left;
    padding: 12px 16px;
    color: var(--text-muted);
    font-weight: 500;
    border-bottom: 1px solid var(--border);
    background: rgba(15, 23, 42, 0.5);
    white-space: nowrap;
    cursor: pointer;
}
td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(255,255,255,0.02); }

.chart-container {
    margin-top: 10px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}
"""

JS = """
function switchMarket(code) {
    document.querySelectorAll('.market-container').forEach(el => el.classList.remove('active'));
    document.getElementById('cont-' + code).classList.add('active');
    document.querySelectorAll('.market-tab').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + code).classList.add('active');
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
            th.dataset.asc = (!asc).toString();
        });
    });
}
window.addEventListener('DOMContentLoaded', init);
"""

def render_card(r: pd.Series, badge_type: str, curr: str) -> str:
    euphoria_cls = "euphoria-glow" if is_euphoria(r) else ""
    euphoria_tag = (
        '<span class="badge" style="background:rgba(245,158,11,0.2);color:#fbbf24;margin-left:6px">Euphoria</span>'
        if is_euphoria(r) else ""
    )
    news_tag = (
        '<span class="badge news" style="margin-left:6px">News</span>'
        if "News:" in (r["Comment"] or "") else ""
    )

    # Fundamental badge
    score = r["Fundy_Score"]
    if score >= 7:
        s_badge = "shield-high"
    elif score <= 3:
        s_badge = "shield-low"
    else:
        s_badge = "buy"
    if score == 10:
        s_icon = "💎"
    elif score >= 7:
        s_icon = "🛡️"
    elif score <= 3:
        s_icon = "⚠️"
    else:
        s_icon = "⚖️"
    fundy_html = (
        f'<span class="badge {s_badge}" style="margin-left:6px">'
        f'{s_icon} {score}/10 {r["Fundy_Tier"]}</span>'
    )

    # Category badge
    cat = str(r.get("Category", "Core"))
    cat_lower = cat.lower()
    if cat_lower == "core":
        bucket_html = (
            '<span class="badge" style="margin-left:6px;'
            'background:rgba(56,189,248,0.14);color:#38bdf8;'
            'border:1px solid rgba(56,189,248,0.4)">Core</span>'
        )
    elif cat_lower == "growth":
        bucket_html = (
            '<span class="badge" style="margin-left:6px;'
            'background:rgba(16,185,129,0.12);color:#22c55e;'
            'border:1px solid rgba(34,197,94,0.5)">Growth</span>'
        )
    elif cat_lower == "spec":
        bucket_html = (
            '<span class="badge" style="margin-left:6px;'
            'background:rgba(239,68,68,0.12);color:#f97373;'
            'border:1px solid rgba(248,113,113,0.6)">Degenerate</span>'
        )
    else:
        bucket_html = ""

    rsi_color = (
        "#ef4444" if r["RSI14"] > 70 else ("#10b981" if r["RSI14"] > 45 else "#f59e0b")
    )

    return f"""
    <div class="card searchable-item {euphoria_cls}">
        <div class="card-header">
            <div>
                <a href="#" class="ticker-badge mono">{r['Ticker']}</a>
                <span class="badge {badge_type}" style="margin-left:8px">{r['Signal']}</span>
                {fundy_html} {bucket_html} {euphoria_tag} {news_tag}
                <div style="font-size:12px; color:var(--text-muted); margin-top:4px">
                    {r['Name']}
                </div>
            </div>
            <div class="price-block">
                <div class="price-main mono">{curr}{r['LastClose']:.2f}</div>
                <div class="price-sub">{r['LastDate']}</div>
            </div>
        </div>
        <div class="metrics-row">
            <div class="metric">
                <label>RSI(14)</label>
                <span class="mono" style="color:{rsi_color}">{r['RSI14']:.0f}</span>
            </div>
            <div class="metric">
                <label>vs 200DMA</label>
                <span class="mono">{r['Dist_to_SMA200_%']:+.1f}%</span>
            </div>
            <div class="metric">
                <label>vs 52W High</label>
                <span class="mono">{r['Dist_to_52W_High_%']:+.1f}%</span>
            </div>
        </div>
        <div class="comment-box">{r['Comment']}</div>
        <div class="chart-container">{r['_mini_candle']}</div>
    </div>
    """

def render_kpi(label: str, val, color_cls: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-lbl">{label}</div>
        <div class="kpi-val {color_cls}">{val}</div>
    </div>
    """

def render_market_html(m_code: str, m_conf: dict, snaps_df: pd.DataFrame, news_df: pd.DataFrame) -> str:
    if snaps_df.empty:
        return f"<div id='cont-{m_code}' class='market-container'><div style='padding:50px;text-align:center'>No data for {m_code}</div></div>"

    # Split Core+Growth vs Spec
    mask_spec = snaps_df["Category"].str.lower() == "spec"
    CORE = snaps_df[~mask_spec].copy()
    DEGEN = snaps_df[mask_spec].copy()

    BUY = CORE[CORE.Signal == "BUY"].sort_values(
        ["Fundy_Score", "Dist_to_52W_High_%"], ascending=[False, False]
    )
    DCA = CORE[CORE.Signal == "DCA"].sort_values(
        ["Fundy_Score", "Dist_to_SMA200_%"], ascending=[False, True]
    )
    WATCH = CORE[CORE.Signal == "WATCH"].sort_values(
        ["Fundy_Score", "Dist_to_52W_High_%"], ascending=[False, False]
    )
    AVOID = CORE[CORE.Signal == "AVOID"].sort_values(
        "Fundy_Score", ascending=True
    )

    GATE = CORE[CORE["AutoDCA_Flag"] == True].sort_values(
        "AutoDCA_Fill_%", ascending=False
    )
    PATS = CORE[CORE["_pattern_name"] != ""].sort_values(
        ["_pattern_conf", "Ticker"], ascending=[False, True]
    )

    BRKCOUNT = int(CORE["BreakoutReady"].sum())
    NEWSCOUNT = len(news_df)

    curr = m_conf["currency"]

    # Cards
    html_cards = []
    for section, df, badge in [
        ("BUY — Actionable", BUY, "buy"),
        ("DCA — Accumulate", DCA, "dca"),
        ("WATCH — Monitoring", WATCH, "watch"),
        ("AVOID — Sidelines", AVOID, "avoid"),
    ]:
        if not df.empty:
            grid_items = "".join(
                [render_card(r, badge, curr) for _, r in df.iterrows()]
            )
            html_cards.append(
                f"<h2 id='{m_code}-{badge}' style='margin-top:30px; font-size:18px; color:var(--text-muted)'>{section}</h2><div class='grid'>{grid_items}</div>"
            )

    # KPI ribbon
    counts = CORE["Signal"].value_counts()
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

    # Auto-DCA table
    dca_rows = "".join(
        f"<tr class='searchable-item'><td><span class='ticker-badge mono'>{r['Ticker']}</span></td>"
        f"<td class='mono text-red'>{r['AutoDCA_Gap_%']:.1f}%</td>"
        f"<td class='mono'>{'Yes' if r['AutoDCA_ReclaimMid'] else 'No'}</td>"
        f"<td class='mono'>{'Yes' if r['AutoDCA_AboveEMA21'] else 'No'}</td>"
        f"<td class='mono'>{r['AutoDCA_Fill_%']:.1f}%</td>"
        f"<td>{r['_mini_spark']}</td></tr>"
        for _, r in GATE.iterrows()
    )

    # Patterns table
    pat_rows = "".join(
        f"<tr class='searchable-item'>"
        f"<td>{r['_pattern_name']}</td>"
        f"<td><span class='ticker-badge mono'>{r['Ticker']}</span></td>"
        f"<td class='mono'>{r['_pattern_status']}</td>"
        f"<td class='mono'>{r['_pattern_conf']:.2f}</td>"
        f"<td class='mono'>{r['_pattern_align']}</td>"
        f"<td>{r['_mini_candle']}</td></tr>"
        for _, r in PATS.iterrows()
    )

    # News table
    if not news_df.empty:
        news_rows = "".join(
            f"<tr class='searchable-item'>"
            f"<td class='mono' style='color:var(--text-muted)'>{r['Date']}</td>"
            f"<td><b>{r['Ticker']}</b></td>"
            f"<td><span class='badge news'>{r['Type']}</span></td>"
            f"<td>{r['Headline']}</td></tr>"
            for _, r in news_df.sort_values("Date", ascending=False).iterrows()
        )
    else:
        news_rows = "<tr><td colspan='4' style='text-align:center; color:gray'>No news data (PDFs) available for this region.</td></tr>"

    # Fundamentals table
    def fmt_pe(x):
        return f"{x:.1f}" if x and x > 0 else "-"

    def fmt_pct(x):
        return f"{x*100:.1f}%" if x else "-"

    fundy_rows = "".join(
        f"<tr class='searchable-item'>"
        f"<td><span class='ticker-badge mono'>{r['Ticker']}</span></td>"
        f"<td><span class='badge {'shield-high' if r['Fundy_Score']>=7 else ('shield-low' if r['Fundy_Score']<=3 else 'watch')}'>{r['Fundy_Score']}/10 {r['Fundy_Tier']}</span></td>"
        f"<td class='mono'>{fmt_pct(r['Fundy_ROE'])}</td>"
        f"<td class='mono'>{fmt_pct(r['Fundy_Margin'])}</td>"
        f"<td class='mono'>{fmt_pe(r['Fundy_PE'])}</td>"
        f"<td class='mono'>{fmt_pct(r['Fundy_RevCAGR'])}</td>"
        f"<td class='mono'>{r['Fundy_Debt']:.2f}</td></tr>"
        for _, r in CORE.sort_values("Fundy_Score", ascending=False).iterrows()
    )

    # Degenerate Radar (Spec only)
    degen_rows = "".join(
        f"<tr class='searchable-item'>"
        f"<td><span class='ticker-badge mono'>{r['Ticker']}</span></td>"
        f"<td>{r['Name']}</td>"
        f"<td class='mono'>{r['Fundy_Score']}/10</td>"
        f"<td class='mono'>{r['Signal']}</td>"
        f"<td>{r['_mini_spark']}</td></tr>"
        for _, r in DEGEN.sort_values("Fundy_Score", ascending=True).iterrows()
    )
    if not degen_rows:
        degen_rows = "<tr><td colspan='5' style='text-align:center'>No Spec/degenerate names defined for this market.</td></tr>"

    nav_html = f"""
    <div class="nav-wrapper">
        <div class="nav-inner">
            <a href="#" class="nav-link active" style="font-weight:700; color:white">Overview</a>
            <a href="#{m_code}-buy" class="nav-link">Buy</a>
            <a href="#{m_code}-dca" class="nav-link">DCA</a>
            <a href="#{m_code}-watch" class="nav-link">Watch</a>
            <a href="#{m_code}-fundy" class="nav-link">Fundamentals</a>
            <a href="#{m_code}-gate" class="nav-link">Auto-DCA</a>
            <a href="#{m_code}-patterns" class="nav-link">Patterns</a>
            <a href="#{m_code}-degen" class="nav-link">Degenerate Radar</a>
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
                <div style="color:var(--text-muted); font-size:13px">
                    Updated {datetime.now(zoneinfo.ZoneInfo(m_conf['tz'])).strftime('%I:%M %p %Z')}
                </div>
            </div>
            {kpi_html}
            {"".join(html_cards)}

            <h2 id="{m_code}-fundy" style="margin-top:40px">Fundamental Health Check</h2>
            <div class="card">
                <div class="playbook">
                    <b>The TraderBruh Shield:</b> 💎 7-10 (Fortress), ⚖️ 4-6 (Quality),
                    ⚠️ 0-3 (High-Risk). Category tag shows Core / Growth / Degenerate bucket.
                </div>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Ticker</th><th>Score</th><th>ROE</th><th>Margin</th>
                                <th>P/E</th><th>Rev Growth</th><th>Debt/Eq</th>
                            </tr>
                        </thead>
                        <tbody>{fundy_rows}</tbody>
                    </table>
                </div>
            </div>

            <h2 id="{m_code}-gate" style="margin-top:40px">Auto-DCA Candidates</h2>
            <div class="card">
                <div class="playbook">
                    <b>Playbook:</b> Gap-down &lt; {RULES['autodca']['gap_thresh']}%, reclaim mid-gap,
                    close &gt; EMA21 and &gt;= {RULES['autodca']['fill_req']}% gap-fill.
                </div>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Ticker</th><th>Gap %</th><th>Reclaim?</th>
                                <th>&gt; EMA21?</th><th>Gap-fill %</th><th>Trend</th>
                            </tr>
                        </thead>
                        <tbody>{dca_rows if dca_rows else "<tr><td colspan='6' style='text-align:center'>No setups.</td></tr>"}</tbody>
                    </table>
                </div>
            </div>

            <h2 id="{m_code}-patterns" style="margin-top:40px">Patterns &amp; Structures</h2>
            <div class="card">
                <div class="playbook">
                    <b>Playbook:</b> Confirmed Double Tops/Bottoms, Triangles and H&amp;S in the last {PATTERN_LOOKBACK} days.
                </div>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Pattern</th><th>Ticker</th><th>Status</th><th>Conf</th><th>Align</th><th>Mini</th>
                            </tr>
                        </thead>
                        <tbody>{pat_rows if pat_rows else "<tr><td colspan='6' style='text-align:center'>No patterns.</td></tr>"}</tbody>
                    </table>
                </div>
            </div>

            <h2 id="{m_code}-degen" style="margin-top:40px">Degenerate Radar (Spec Only)</h2>
            <div class="card">
                <div class="playbook">
                    <b>High-Risk Speculative Bucket:</b> Names explicitly tagged as Spec.
                    Treat as lotto tickets, not retirement plans.
                </div>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Ticker</th><th>Name</th><th>Shield</th><th>Signal</th><th>Trend</th>
                            </tr>
                        </thead>
                        <tbody>{degen_rows}</tbody>
                    </table>
                </div>
            </div>

            <h2 id="{m_code}-news" style="margin-top:40px">News</h2>
            <div class="card" style="padding:0">
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr><th>Date</th><th>Ticker</th><th>Type</th><th>Headline</th></tr>
                        </thead>
                        <tbody>{news_rows}</tbody>
                    </table>
                </div>
            </div>

            <div style="height:50px"></div>
        </div>
    </div>
    """

# ---------------- Main ----------------

if __name__ == "__main__":
    print("Starting TraderBruh Global Hybrid...")
    market_htmls = []
    tab_buttons = []

    for m_code, m_conf in MARKETS.items():
        df, news = process_market(m_code, m_conf)
        html_part = render_market_html(m_code, m_conf, df, news)
        market_htmls.append(html_part)
        active = "active" if m_code == "AUS" else ""
        tab_buttons.append(
            f"<button id='tab-{m_code}' class='market-tab {active}' onclick=\"switchMarket('{m_code}')\">{m_conf['name']}</button>"
        )

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
        Generated by TraderBruh Global v4.2-hybrid • Not Financial Advice
    </div>
</body>
</html>
"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(full_html)
    print("Done:", OUTPUT_HTML)
