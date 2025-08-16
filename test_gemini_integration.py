# tests/test_gemini_integration.py
from unittest.mock import patch
from services.gemini_chatbot import get_general_financial_advice

def test_gemini_response():
    """Test basic Gemini response with mocked API"""
    with patch('services.gemini_chatbot.genai.GenerativeModel') as mock_model:
        # Set up mock response
        mock_response = type('obj', (object,), {'text': 'Compound interest is the interest calculated on the initial principal and also on the accumulated interest of previous periods.'})
        mock_model.return_value.generate_content.return_value = mock_response
        
        response = get_general_financial_advice("What is compound interest?")
        
        assert isinstance(response, str)
        assert len(response) > 50
        assert "compound" in response.lower() or "interest" in response.lower()