from services.news_fetcher import fetch_stock_news

def test_fetch_valid_news():
    """Test news fetching for valid symbol"""
    news = fetch_stock_news("AAPL")
    assert isinstance(news, str)
    assert "AAPL" in news or "Apple" in news
    assert len(news) > 50  # Reasonable news length

def test_fetch_invalid_news():
    """Test news fetching for invalid symbol"""
    news = fetch_stock_news("INVALID_SYMBOL_123")
    assert "No recent news" in news or "Error" in news

# tests/test_news_fetcher.py
def test_fetch_invalid_news():
    news = fetch_stock_news("INVALID_SYMBOL_123")
    assert "Could not fetch news" in news or "No recent news" in news or "disabled" in news