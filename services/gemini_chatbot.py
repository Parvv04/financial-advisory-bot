from deep_translator import GoogleTranslator
import logging
import google.generativeai as genai
from services.data_fetcher import GEMINI_API_KEY

TRANSLATOR = GoogleTranslator(source='auto', target='en')

EDUCATION = {
    "rsi": "RSI (Relative Strength Index) measures momentum on a scale of 0-100, indicating overbought (>70) or oversold (<30) conditions.",
    "p/e": "P/E (Price-to-Earnings) ratio compares a company's stock price to its earnings per share, useful for valuation."
}

GENERAL_EDUCATION = {
    "emergency fund": "An emergency fund is savings for unexpected expenses like job loss or medical emergencies. Aim for 3-6 months of living expenses.",
    "budgeting": "Budgeting involves tracking income and expenses to control spending. The 50/30/20 rule is popular: 50% needs, 30% wants, 20% savings.",
    "debt snowball": "The debt snowball method means paying off your smallest debts first while making minimum payments on larger debts, gaining momentum.",
    "compound interest": "Compound interest means your money earns interest on interest. Early investing benefits from exponential growth.",
    "etf": "ETFs (Exchange Traded Funds) are collections of stocks or bonds you can buy in a single fund, offering diversification at low cost.",
    "diversification": "Diversification reduces risk by spreading investments across different assets like stocks, bonds, real estate, etc.",
    "roth ira": "A Roth IRA allows post-tax contributions and tax-free withdrawals in retirement, ideal if you expect to be in a higher tax bracket later.",
    "401k match": "A 401(k) employer match is free money. Contribute at least enough to get the full match—it’s an instant 100% return.",
    "index fund": "Index funds track a market index (like the S&P 500). They offer broad diversification and low fees, great for long-term growth."
}

# Financial Education Section
finance_questions = [
            "What is an emergency fund?",
            "How does budgeting work?",
            "Explain the debt snowball method",
            "What is compound interest?",
            "What are ETFs?",
            "What is diversification?",
            "How does a Roth IRA work?",
            "What is a 401k match?",
            "What is an index fund?"
        ]

def get_general_financial_advice(query):
    try:
        # Use the latest supported Gemini model
        model = genai.GenerativeModel("gemini-1.5-pro-latest")

        # Call Gemini with the full question
        response = model.generate_content(query)

        return response.text
    except Exception as e:
        return f"Error getting response: {e}"

def get_general_financial_advice(query, symbols=None, stock_data=None, results=None):
    import google.generativeai as genai
    import os
    from dotenv import load_dotenv
    load_dotenv()

    try:
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
        
        # Configure and use Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        return f"Error getting response: {e}"

    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    context = ""

    if symbols and stock_data is not None and results is not None:
        for symbol in symbols:
            try:
                rsi = stock_data[f"{symbol}_RSI"].iloc[-1]
                actual = results[symbol]["actual"].iloc[-1]
                predicted = results[symbol]["predicted"].iloc[-1]
                trend = "rising" if predicted > actual else "falling"
                context += f"{symbol}: RSI = {rsi:.2f}, Trend = {trend}, Predicted Price = ₹{predicted:.2f}, Actual Price = ₹{actual:.2f}\n"
            except:
                continue

    prompt = f"""You are a financial assistant. Here's the current market context:\n{context}\n\nUser query: {query}
Provide smart, actionable, and responsible investment guidance. Avoid financial guarantees. Be brief and specific when possible."""

    try:
        result = model.generate_content(prompt)
        return result.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

def calculate_savings_goal(target_amount, years, annual_return_percent):
    """
    Calculate monthly saving needed to reach a financial goal.
    Args:
        target_amount (float): Desired final amount
        years (float): Years to save
        annual_return_percent (float): Expected annual return (e.g., 7%)

    Returns:
        dict with required monthly savings and projected value
    """
    r = annual_return_percent / 100 / 12  # monthly rate
    n = years * 12  # total months
    if r == 0:
        monthly = target_amount / n
    else:
        monthly = target_amount * r / ((1 + r) ** n - 1)
    monthly = abs(monthly)

    return {
        "monthly_saving": monthly,
        "years": years,
        "target_amount": target_amount,
        "annual_return": annual_return_percent
    }

def translate_response(text, lang='en'):
    return TRANSLATOR.translate(text, dest=lang).text

