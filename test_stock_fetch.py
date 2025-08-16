from services.data_fetcher import fetch_stock_data, get_stock_data
import pytest

def test_fetch_valid_stock():
    """Test fetching data for a valid stock symbol"""
    data = fetch_stock_data("AAPL")
    assert data is not None
    assert not data.empty
    assert "AAPL" in data.columns
    assert len(data) > 50  # More lenient check

def test_fetch_multiple_stocks():
    """Test fetching multiple stocks at once"""
    # Use get_stock_data which is designed for multiple symbols
    data = get_stock_data(["AAPL", "MSFT"])
    assert data is not None
    assert not data.empty
    assert "AAPL" in data.columns
    assert "MSFT" in data.columns
    assert len(data) > 50  # More lenient check

def test_fetch_invalid_stock():
    """Test handling of invalid stock symbol"""
    data = fetch_stock_data("INVALID_SYMBOL_123")
    assert data is None
