import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_rsi(series, periods=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

    print("Stock index preview:", self.stock_data.index[:5])
    print("Index type:", type(self.stock_data.index))

def add_technical_indicators(df, symbols):
    for symbol in symbols:
        price = df[symbol]

        # Moving Averages
        df[f'{symbol}_MA10'] = price.rolling(window=10).mean()
        df[f'{symbol}_EMA20'] = price.ewm(span=20, adjust=False).mean()

        # RSI
        df[f'{symbol}_RSI'] = compute_rsi(price, 14)

        # MACD
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        df[f'{symbol}_MACD'] = ema12 - ema26

        # Bollinger Bands
        df[f'{symbol}_Bollinger_Upper'] = price.rolling(20).mean() + price.rolling(20).std() * 2
        df[f'{symbol}_Bollinger_Lower'] = price.rolling(20).mean() - price.rolling(20).std() * 2

        # ✅ New Features:
        df[f'{symbol}_Momentum'] = price - price.shift(10)
        df[f'{symbol}_Volatility'] = price.rolling(10).std()

        if 'Volume' in df.columns:
            df[f'{symbol}_Volume_MA'] = df['Volume'].rolling(10).mean()

    return df

def calculate_risk(symbol, stock_data, results):
    data = stock_data[symbol].dropna()
    if len(data) < 10:
        print(f"Warning: Insufficient data for risk calculation for {symbol}. Defaulting to 50.")
        return 50
    returns = data.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100
    rsi = stock_data[f'{symbol}_RSI'].iloc[-1]
    rsi_risk = max(0, abs(rsi - 50) - 20) / 0.7
    pred_change = (results[symbol]['predicted'].iloc[-1] - results[symbol]['actual'].iloc[-1]) / results[symbol]['actual'].iloc[-1] * 100
    pred_risk = abs(pred_change) / 2
    risk_score = min(100, (volatility + rsi_risk + pred_risk) / 3)
    print(f"Calculated risk for {symbol}: {risk_score:.2f}")
    return risk_score

def get_advice(series):
    recent = series.dropna().iloc[-10:]
    if len(recent) < 2:
        return "Insufficient data to generate advice."

    trend = "Uptrend" if recent.iloc[-1] > recent.iloc[0] else "Downtrend"
    if trend == "Uptrend":
        return "Uptrend predicted. Consider holding or buying more."
    else:
        return "Downtrend predicted. You might want to wait."

def get_metrics(symbol, data):
    return f"{symbol} Metrics: MA10={data[f'{symbol}_MA10'].iloc[-1]:.2f}, RSI={data[f'{symbol}_RSI'].iloc[-1]:.2f}"