import logging
from services.data_fetcher import ALTERNATIVE_STOCKS

logger = logging.getLogger(__name__)

ALERTS = {}
PORTFOLIO = {}

def get_strategy(advice, risk_score):
    if risk_score > 70:
        return "High-risk strategy: Avoid or use options hedging."
    elif "invest" in advice:
        return "Dollar-cost averaging"
    else:
        return "Wait and monitor"

def check_alerts(results):
    for symbol, threshold in ALERTS.items():
        if symbol in results:
            pred_change = (results[symbol]['predicted'].iloc[-1] - results[symbol]['actual'].iloc[-1]) / results[symbol]['actual'].iloc[-1]
            if abs(pred_change) > threshold:
                return f"Alert: {symbol} changed by {pred_change*100:.2f}%!"
    return "No alerts triggered."


def get_alternative_options(symbol, results, stock_data):
    alternatives = {}
    for alt_symbol in ALTERNATIVE_STOCKS:
        if alt_symbol not in results:
            alt_data = fetch_stock_data(alt_symbol)
            if alt_data is not None:
                alt_rsi = compute_rsi(alt_data[alt_symbol]).iloc[-1]
                alt_trend = "Uptrend" if alt_data[alt_symbol].iloc[-1] > alt_data[alt_symbol].iloc[-10] else "Downtrend"
                alternatives[alt_symbol] = {'RSI': alt_rsi, 'Trend': alt_trend}
    if not alternatives:
        return "No viable alternatives available."
    best_alt = min(alternatives.items(), key=lambda x: abs(x[1]['RSI'] - 50))
    return f"Consider {best_alt[0]}: RSI={best_alt[1]['RSI']:.2f}, Trend={best_alt[1]['Trend']}"

