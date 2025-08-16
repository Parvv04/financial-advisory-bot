from typing import List, Optional, Dict, Any
import time
import datetime
from functools import lru_cache
import logging
from services.data_fetcher import NEWS_API_KEY
import requests
import os

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
logger = logging.getLogger(__name__)

# Cache for storing news responses to avoid repeated API calls
NEWS_CACHE = {}
NEWS_CACHE_TTL = 300  # 5 minutes cache TTL

@lru_cache(maxsize=32)
def _fetch_news_from_api(symbol: str, max_articles: int = 3) -> Optional[list]:
    """
    Internal function to fetch news from API with caching.
    Returns None if there was an error.
    """
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not configured. News feature will be disabled.")
        return None
    
    cache_key = f"{symbol}:{max_articles}"
    current_time = time.time()
    
    # Return cached result if it exists and is still valid
    if cache_key in NEWS_CACHE:
        cached_time, cached_data = NEWS_CACHE[cache_key]
        if current_time - cached_time < NEWS_CACHE_TTL:
            return cached_data
    
    try:
        # Use a shorter timeout for the request
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}&language=en&pageSize={max_articles}"
        response = requests.get(url, timeout=5)  # Reduced timeout to 5 seconds
        response.raise_for_status()
        
        articles = response.json().get("articles", [])[:max_articles]
        
        # Cache the result
        NEWS_CACHE[cache_key] = (current_time, articles)
        return articles
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in _fetch_news_from_api: {str(e)}")
        return None

def fetch_stock_news(symbol, max_articles=3):
    """
    Fetch news articles for a given stock symbol using NewsAPI with caching.
    
    Args:
        symbol: Stock symbol to fetch news for
        max_articles: Maximum number of articles to return (max 10)
        
    Returns:
        Formatted string with news articles or error message
    """
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not configured.")
        return "News feature is disabled. Please configure NEWS_API_KEY."

    try:
        # Limit max_articles to prevent excessive API usage
        max_articles = min(int(max_articles), 10)
        
        # Get articles from cache or API
        articles = _fetch_news_from_api(symbol, max_articles)
        
        if not articles:
            return f"Could not fetch news for {symbol}. Please try again later."
            
        if not articles:
            return f"No recent news found for {symbol}."

        # Format the news string
        news_str = f"📰 **Latest News for {symbol}**:\n"
        for article in articles:
            title = article.get('title', 'No title')
            link = article.get('url', '#')
            source = article.get('source', {}).get('name', 'Unknown source')
            date = article.get('publishedAt', '')[:10] if article.get('publishedAt') else 'Unknown date'
            news_str += f"- **{title}** ({source}, {date})\n  [Read more]({link})\n"
        return news_str
        
    except Exception as e:
        logger.error(f"Error in fetch_stock_news: {str(e)}")
        return f"An error occurred while fetching news for {symbol}."