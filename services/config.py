# Configure logging
logging.basicConfig(level=logging.INFO)

START_DATE = '2015-01-01'
LOOKBACK = 30  # Reduced to ensure sufficient data
TRAIN_SPLIT = 0.9
ALTERNATIVE_STOCKS = ['MSFT', 'GOOGL', 'JPM']