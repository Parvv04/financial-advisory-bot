import time
from services.data_fetcher import fetch_stock_data

def manual_verify_stock_fetching():
    """Manual verification of stock data fetching"""
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    print("Testing stock data fetching...")
    
    for symbol in symbols:
        print(f"\nFetching data for {symbol}...")
        start_time = time.time()
        data = fetch_stock_data(symbol)
        elapsed = time.time() - start_time
        
        if data is not None and not data.empty:
            print(f"✅ Successfully fetched {symbol} in {elapsed:.2f} seconds")
            print(f"Data shape: {data.shape}")
            print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"Latest price: {data.iloc[-1, 0]:.2f}")
        else:
            print(f"❌ Failed to fetch data for {symbol}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    manual_verify_stock_fetching()