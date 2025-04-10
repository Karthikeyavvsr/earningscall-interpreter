import yfinance as yf
import spacy
import re

# Load spaCy model for NER (Named Entity Recognition)
nlp = spacy.load("en_core_web_sm")

# Predefined Company to Ticker mapping (dictionary)
company_to_ticker = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    # Add more companies here as needed
}

def extract_company_name(text):
    """Extract company names from the transcript using spaCy NER."""
    doc = nlp(text)
    companies = [ent.text for ent in doc.ents if ent.label_ == "ORG"]  # Only extracting organizations
    return companies

def map_to_ticker(company_name):
    """Search for ticker symbol using yfinance or predefined mapping."""
    
    # First check if the company is in our predefined dictionary
    ticker = company_to_ticker.get(company_name, None)
    if ticker:
        return ticker
    
    # If not found in dictionary, try to search dynamically using Yahoo Finance
    try:
        # Clean the company name to avoid common suffixes like "Inc.", "Ltd.", "Corp."
        company_name_clean = re.sub(r"(Inc|Ltd|Corp|Corporation|LLC)$", "", company_name).strip()
        
        # Use Yahoo Finance to get the company ticker
        company = yf.Ticker(company_name_clean)
        
        # Try to fetch the ticker symbol
        if 'symbol' in company.info:
            return company.info['symbol']
        else:
            return None
    except Exception as e:
        return None  # Return None if ticker is not found or an error occurs

def get_stock_info(ticker):
    """Fetch detailed stock data using yfinance."""
    try:
        stock_data = yf.Ticker(ticker)
        stock_info = {
            "company_name": stock_data.info.get('longName', 'N/A'),
            "current_price": stock_data.info.get('currentPrice', 'N/A'),
            "market_cap": stock_data.info.get('marketCap', 'N/A'),
            "pe_ratio": stock_data.info.get('trailingPE', 'N/A'),
            "fifty_two_week_high": stock_data.info.get('fiftyTwoWeekHigh', 'N/A'),
            "fifty_two_week_low": stock_data.info.get('fiftyTwoWeekLow', 'N/A'),
            "dividend_yield": stock_data.info.get('dividendYield', 'N/A'),
            "beta": stock_data.info.get('beta', 'N/A'),
            "revenue": stock_data.financials.loc['Total Revenue'] if 'Total Revenue' in stock_data.financials.index else 'N/A',
            "net_income": stock_data.financials.loc['Net Income'] if 'Net Income' in stock_data.financials.index else 'N/A',
            "guidance": stock_data.guidance if 'guidance' in stock_data.info else 'N/A'
        }
        return stock_info
    except Exception as e:
        return None  # Return None if there is any issue fetching stock data
