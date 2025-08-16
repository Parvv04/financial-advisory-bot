import pandas as pd
import numpy as np
import time
from typing import Optional, List
import datetime
import yfinance as yf
import concurrent.futures
from functools import lru_cache
from typing import List, Optional, Dict, Any, Union
import logging
import os  # Import os at the top level

LOOKBACK = 30
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    
    # Load .env file from the same directory as this script
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded .env file from: {env_path}")
    else:
        logger.warning(f"No .env file found at: {env_path}")
    
    # Get API keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    
    # Debug: Check if API keys are loaded
    logger.info(f"GEMINI_API_KEY loaded: {'Yes' if GEMINI_API_KEY else 'No'}")
    
    # Configure genai if the API key is available
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("Successfully configured Google Generative AI")
        except ImportError:
            logger.warning("google-generativeai package not found. Some features may be limited.")
            genai = None
        except Exception as e:
            logger.error(f"Error configuring Google Generative AI: {str(e)}")
            genai = None
    else:
        logger.warning("GEMINI_API_KEY not found in environment variables")
        genai = None
        
except ImportError:
    logger.warning("python-dotenv package not found. Using system environment variables.")
    # Still try to get the API keys from system environment
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
    genai = None
    # dotenv not available, use system environment variables
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
    genai = None

def fetch_stock_data(symbol, period='1y', interval='1d', max_retries=2):
    """
    Fetch stock data with retries and improved error handling.
    Optimized for speed with reduced timeouts and parallel processing.
    For multiple symbols, it returns a combined DataFrame.

    Args:
        symbol: Stock symbol to fetch data for
        period: Data period to fetch (default: '1y' for 1 year)
        interval: Data interval (default: '1d' for daily)
        max_retries: Maximum number of retry attempts (reduced to 2 for speed)
        
    Returns:
        DataFrame with stock data or None if fetch failed
    """

     # If it's a list of symbols, fetch them individually and combine
    if isinstance(symbol, list):
        all_data = []
        for sym in symbol:
            data = _fetch_single_stock(sym, period, interval)
            if data is not None:
                all_data.append(data)
        if not all_data:
            return None
        return pd.concat(all_data, axis=1).dropna(how='all')
    
    # For single symbol, use the existing logic
    return _fetch_single_stock(symbol, period, interval, max_retries)

    # Cache key for memoization
    cache_key = (symbol, period, interval)
    if hasattr(fetch_stock_data, '_cache') and cache_key in fetch_stock_data._cache:
        cached_time, cached_data = fetch_stock_data._cache[cache_key]
        if time.time() - cached_time < 300:  # 5 minute cache
            return cached_data.copy()
    
    for attempt in range(max_retries):
        try:
            # Initialize Ticker without session
            ticker = yf.Ticker(symbol)
            
            # Fetch data with only supported parameters
            data = ticker.history(
                period=period,
                interval=interval
            )
            
            if data is None or data.empty:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: No data for {symbol}")
                continue
                
            # Process data more efficiently
            data = data[['Close']].rename(columns={'Close': symbol})
            data.index = data.index.tz_localize(None, ambiguous='infer')
            
            # Verify data sufficiency
            if len(data) < LOOKBACK:
                logger.warning(f"Insufficient data for {symbol}. Have {len(data)} points.")
                if period != 'max':
                    return fetch_stock_data(symbol, period='max', interval=interval, max_retries=1)
                continue
                
            # Cache the result
            if not hasattr(fetch_stock_data, '_cache'):
                fetch_stock_data._cache = {}
            fetch_stock_data._cache[cache_key] = (time.time(), data.copy())
            
            logger.info(f"Fetched {symbol} with {len(data)} data points")
            return data
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {symbol}: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch {symbol} after {max_retries} attempts")
                return None
            
            # Shorter backoff
            time.sleep(1 * (attempt + 1))  # 1s, 2s
    
    return None

@lru_cache(maxsize=32)
def _fetch_single_stock(symbol: str, period: str = '2y', interval: str = '1d', timeout: int = 10) -> Optional[pd.DataFrame]:
    """Fetch data for a single stock symbol with error handling and retries."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            period=period,
            interval=interval,
            timeout=timeout
        )
        
        if data.empty:
            logger.warning(f"No data returned for symbol: {symbol}")
            return None
            
        data = data[['Close']].rename(columns={'Close': symbol})
        data.index = data.index.tz_localize(None)
        return data
        
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {str(e)}")
        return None

def get_stock_data(symbols: List[str], lookback: int = 30, max_workers: int = 10) -> Optional[pd.DataFrame]:
    """
    Fetch stock data for multiple symbols in parallel with improved performance.
    
    Args:
        symbols: List of stock symbols to fetch
        lookback: Minimum number of data points required
        max_workers: Maximum number of parallel workers (increased to 10)
        
    Returns:
        DataFrame with stock data or None if no data could be fetched
    """

    if not isinstance(symbols, (list, tuple)):
        symbols = [symbols]
    
    # Clean and deduplicate symbols
    symbols = list({s.upper().strip() for s in symbols if s and isinstance(s, str)})
    
    if not symbols:
        logger.warning("No valid symbols provided")
        return None
    
    logger.info(f"Fetching data for {len(symbols)} symbols...")
    start_time = time.time()
    
    # Use ProcessPoolExecutor for CPU-bound work (better than ThreadPool for yfinance)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
        # Submit all fetch tasks
        future_to_symbol = {}
        for symbol in symbols:
            future = executor.submit(
                fetch_stock_data, 
                symbol=symbol,
                period='1y',  # Start with 1 year
                interval='1d',
                max_retries=1  # Reduced retries for speed
            )
            future_to_symbol[future] = symbol
        
        # Process results as they complete
        all_data = []
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if data is not None and len(data) >= lookback:
                    all_data.append(data)
                    logger.debug(f"Fetched {symbol} with {len(data)} rows")
                else:
                    logger.warning(f"Insufficient data for {symbol}")
            except Exception as e:
                logger.warning(f"Error processing {symbol}: {str(e)}")
    
    if not all_data:
        logger.error("No valid data could be fetched for any symbol")
        return None
    
    # Combine data more efficiently
    try:
        combined = pd.concat(all_data, axis=1).dropna(how='all')
        if combined.empty:
            logger.error("No overlapping data points found across symbols")
            return None
            
        logger.info(f"Fetched data for {len(combined.columns)}/{len(symbols)} symbols in "
                   f"{time.time() - start_time:.2f} seconds")
        return combined
        
    except Exception as e:
        logger.error(f"Error combining data: {str(e)}")
        return None

