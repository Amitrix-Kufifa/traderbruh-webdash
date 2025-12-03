# traderbruh_backtest.py
# "The Arena" V2 — Validating ATR Trailing Stops

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

UNIVERSE = ['BHP', 'CSL', 'CBA', 'XRO', 'FMG', 'WTC', 'TLS', 'WOW', 'STO', 'PLS']
SUFFIX = '.AX'
START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

def calculate_indicators(df):
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # ATR Calculation (14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

print("--- TRADERBRUH ARENA V2 (ATR TRAILING STOP) ---")
print(f"Testing with 3x ATR Trailing Stop on {len(UNIVERSE)} stocks...\n")

total_trades = 0
winning_trades = 0
total_pnl_pct = 0
results = []

for ticker in UNIVERSE:
    symbol = ticker + SUFFIX
    
    df = yf.download(symbol, start=START_DATE, progress=False, auto_adjust=False)
    if df.empty: continue
    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.xs(symbol, axis=1, level=-1)
        except: df.columns = df.columns.get_level_values(0)

    df = calculate_indicators(df).dropna()
    
    in_position = False
    entry_price = 0
    highest_price = 0  # Track highest price since entry for trailing stop
    ticker_pnl = 0
    trades_count = 0
    
    for i in range(1, len(df)):
        today = df.iloc[i]
        
        # 1. ENTRY (Same as before)
        # Price > SMA200, SMA50 > SMA200, RSI Healthy
        buy_signal = (today['Close'] > today['SMA200']) and \
                     (today['SMA50'] > today['SMA200']) and \
                     (45 <= today['RSI'] <= 75)
                     
        if buy_signal and not in_position:
            in_position = True
            entry_price = float(today['Close'])
            highest_price = entry_price
            trades_count += 1
            
        # 2. MANAGEMENT (While holding)
        elif in_position:
            current_price = float(today['Close'])
            if current_price > highest_price:
                highest_price = current_price
            
            # --- THE FIX: ATR TRAILING STOP ---
            # Stop is 3x ATR below the Highest Price reached during the trade
            atr_stop = highest_price - (3.0 * today['ATR'])
            
            # Hard Trend Stop (Backstop)
            sma_stop = today['SMA200'] * 0.97
            
            # EXIT TRIGGER
            if current_price < atr_stop or current_price < sma_stop:
                exit_price = current_price
                pnl = (exit_price - entry_price) / entry_price
                ticker_pnl += pnl
                in_position = False
                if pnl > 0: winning_trades += 1

    # Mark to market
    if in_position:
        exit_price = float(df.iloc[-1]['Close'])
        pnl = (exit_price - entry_price) / entry_price
        ticker_pnl += pnl
        if pnl > 0: winning_trades += 1

    print(f"{ticker}: {ticker_pnl*100:.1f}% ({trades_count} trades)")
    total_trades += trades_count
    total_pnl_pct += ticker_pnl
    results.append({'Ticker': ticker, 'Return': ticker_pnl})

print("\n" + "="*30)
print(f"Avg Return V2: {(total_pnl_pct/len(UNIVERSE))*100:.2f}%")
print(f"Win Rate V2:   {(winning_trades/total_trades*100) if total_trades else 0:.1f}%")
print("="*30)