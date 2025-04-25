# utils/stock_info.py
import os
import re
import json
import yfinance as yf
import streamlit as st
import requests

# Local cache directory
CACHE_DIR = os.path.join("data", "cache") # Ensure data directory exists at root or adjust path
os.makedirs(CACHE_DIR, exist_ok=True)

# Manual mapping for known names
MANUAL_MAPPING = {
    "Westport Fuel Systems": "WPRT",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Alphabet": "GOOGL", # Alphabet Inc. maps to GOOGL (Class A)
    "Meta": "META", # Meta Platforms, Inc.
    "Ally Financial": "ALLY",
    "Lululemon Athletica": "LULU",
    # Add more common mappings if needed
}

# --- Finnhub config (ensure key is ideally from secrets/env vars) ---
FINNHUB_API_KEY = None
FINNHUB_SECRET_KEY = "FINNHUB_API_KEY" # Key name in secrets file
FINNHUB_FALLBACK_KEY = "d018oq9r01qile5u5a50d018oq9r01qile5u5a5g" # Your potentially hardcoded key (replace/remove for production)

try:
    # Try accessing secrets only if running within Streamlit context
    if hasattr(st, 'secrets'):
        FINNHUB_API_KEY = st.secrets.get(FINNHUB_SECRET_KEY)
    # Fallback to environment variable if secrets unavailable or key not found
    if not FINNHUB_API_KEY:
        FINNHUB_API_KEY = os.environ.get(FINNHUB_SECRET_KEY)
    # Final fallback to hardcoded key ONLY if others fail (NOT RECOMMENDED)
    if not FINNHUB_API_KEY:
        FINNHUB_API_KEY = FINNHUB_FALLBACK_KEY
        if FINNHUB_API_KEY == FINNHUB_FALLBACK_KEY: # Add warning if using fallback
             print("Warning: Using hardcoded Finnhub API key as fallback.")

except Exception as e:
     print(f"Error accessing secrets/env vars for Finnhub key, using fallback: {e}")
     FINNHUB_API_KEY = FINNHUB_FALLBACK_KEY # Use fallback on error

FINNHUB_URL = "https://finnhub.io/api/v1"
# --- End Finnhub config ---


def extract_company_name(transcript):
    """Extracts company name and ticker symbol from the beginning of a transcript."""
    if not transcript or not isinstance(transcript, str):
         return None, None

    lines = transcript.strip().splitlines()
    for line in lines[:15]: # Check first 15 lines
        line_lower = line.lower()
        # Pattern 1: Explicit "Company Name (TICKER)"
        # Allows more chars in name, limits ticker length
        match = re.search(r"([A-Za-z0-9\.\s,&'-\/]+?)\s+\(?([A-Z\.]{1,6})\)?", line)
        if match:
            company_name = match.group(1).strip()
            ticker = match.group(2).strip().upper().replace(".", "") # Remove potential dots in ticker
            # Basic validation
            if 1 <= len(ticker) <= 6 and len(company_name) > 2 and not company_name.isdigit():
                print(f"Extracted (Pattern 1): Name='{company_name}', Ticker='{ticker}'")
                return company_name, ticker

        # Pattern 2: Look for common phrases identifying the company near start
        # Handles cases where ticker might be missing initially
        common_phrases = ["earnings call", "conference call", "prepared remarks"]
        if any(phrase in line_lower for phrase in common_phrases):
             # Try to extract a plausible company name before the phrase
             match_name = re.search(r"^([A-Z][A-Za-z0-9\.\s,&'-]+(?:Inc\.?|Incorporated|Corporation|Corp\.?|Ltd\.?|LLC|plc|Group|Technologies|Holdings))", line)
             if match_name:
                  company_name = match_name.group(1).strip()
                  if len(company_name) > 4: # Avoid very short spurious matches
                       print(f"Extracted (Pattern 2): Potential Name='{company_name}', Ticker=None")
                       return company_name, None # Return name, let mapping handle ticker

    print("Could not extract company name or ticker from first 15 lines.")
    return None, None


def map_to_ticker(company_name):
    """Maps a company name to a ticker symbol using yfinance and Finnhub fallback."""
    if not company_name or not isinstance(company_name, str) or len(company_name) < 3:
        return None

    query_name = company_name.strip()
    print(f"Attempting to map Company Name: '{query_name}'")

    # 1. Check Manual Mapping (case-insensitive keys)
    for manual_name, ticker in MANUAL_MAPPING.items():
         if query_name.lower() == manual_name.lower():
              print(f"Found '{query_name}' in manual map: {ticker}")
              return ticker

    # 2. Try yfinance lookup (less reliable for names, but worth trying)
    try:
        print(f"Attempting yfinance Ticker lookup for: {query_name}")
        stock = yf.Ticker(query_name)
        info = stock.info # Fetch info dict

        # Check if info was retrieved and contains a symbol
        if info and isinstance(info, dict) and info.get("symbol"):
            retrieved_symbol = info['symbol']
            retrieved_name = info.get('shortName', info.get('longName', ''))
            # Basic check: Is the name reasonably similar? Or is symbol common?
            if retrieved_symbol and retrieved_name and query_name.lower().split()[0] in retrieved_name.lower():
                 print(f"yfinance Ticker found symbol: {retrieved_symbol} for name '{query_name}' (matched name: '{retrieved_name}')")
                 return retrieved_symbol.strip().upper()
            else:
                 print(f"yfinance Ticker found symbol {retrieved_symbol}, but name mismatch? ('{retrieved_name}')")
                 # Proceed to Finnhub search
        else:
             print(f"yfinance Ticker info was empty or missing symbol for '{query_name}'.")

    except Exception as e:
        # yfinance often raises errors for non-ticker strings
        print(f"yfinance Ticker lookup failed for '{query_name}': {type(e).__name__}")
        # Proceed to Finnhub search

    # 3. Try Finnhub Symbol Search Fallback (if API key exists)
    if FINNHUB_API_KEY and FINNHUB_API_KEY != FINNHUB_FALLBACK_KEY and FINNHUB_API_KEY != "YOUR_API_KEY_HERE":
        try:
            search_url = f"{FINNHUB_URL}/search"
            params = {'q': query_name, 'token': FINNHUB_API_KEY}
            print(f"Attempting Finnhub search for: {query_name}")
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()

            if results and results.get('count', 0) > 0 and results.get('result'):
                # Select the best match based on description similarity & symbol format
                best_match = None
                highest_score = -1
                query_lower = query_name.lower()

                for item in results['result']:
                    desc = item.get('description', '').lower()
                    symbol = item.get('symbol', '')
                    # Basic filtering: Skip symbols with '.', likely non-US or non-common stock
                    if not symbol or '.' in symbol or ':' in symbol:
                         continue

                    score = 0
                    if query_lower == desc: score = 3
                    elif query_lower in desc: score = 2
                    # Check if first word matches
                    elif query_lower.split() and desc.split() and query_lower.split()[0] == desc.split()[0]: score = 1

                    # Boost score if symbol type seems like common stock (heuristic)
                    if item.get('type', '').upper() in ['COMMON STOCK', 'EQUITY', 'ADR']:
                         score += 1

                    if score > highest_score:
                         highest_score = score
                         best_match = item

                if best_match and best_match.get('symbol'):
                     ticker = best_match['symbol'].strip().upper()
                     # Final sanity check on ticker format
                     if 1 <= len(ticker) <= 6 and ticker.isalnum():
                         print(f"Finnhub search mapped '{query_name}' to ticker: {ticker} (Description: {best_match.get('description')}, Score: {highest_score})")
                         return ticker
                     else:
                         print(f"Finnhub best match '{ticker}' has invalid format, discarding.")
                else:
                     print(f"Finnhub search found results, but no confident match for '{query_name}'.")
            else:
                print(f"Finnhub search yielded no results for '{query_name}'.")

        except requests.exceptions.RequestException as e:
            print(f"Finnhub search request failed for '{query_name}': {e}")
        except Exception as e:
            print(f"Error during Finnhub search processing for '{query_name}': {e}")
    else:
         print("Skipping Finnhub search: API key not configured properly.")

    print(f"Could not map company name '{query_name}' to a ticker using available methods.")
    return None # Failed to map


# --- Caching Functions ---
def load_cached_stock_info(ticker):
    """Loads stock info from local JSON cache."""
    path = os.path.join(CACHE_DIR, f"{ticker}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache file {path}: {e}")
            return None
    return None

def save_stock_info_cache(ticker, data):
    """Saves stock info to local JSON cache."""
    path = os.path.join(CACHE_DIR, f"{ticker}.json")
    try:
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4) # Use indent for readability
    except Exception as e:
        print(f"Error saving cache file {path}: {e}")

# --- Fallback using Finnhub Quote (if yfinance fails) ---
def get_finnhub_backup(ticker):
    """Fetches basic quote data from Finnhub as a fallback."""
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == FINNHUB_FALLBACK_KEY or FINNHUB_API_KEY == "YOUR_API_KEY_HERE":
        return {"error": "Finnhub API key not configured for backup quote."}
    try:
        url = f"{FINNHUB_URL}/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Basic check if Finnhub returned meaningful data (current price 'c' is often key)
        if data.get('c') is None or data.get('c') == 0:
             print(f"Finnhub quote data for {ticker} seems empty or invalid.")
             return {"error": f"Finnhub returned no valid quote data for {ticker}."}

        # Return a structure similar to get_stock_info, filling what we can
        return {
            "company": ticker, # Can't get company name easily from /quote
            "sector": "N/A",
            "market_cap": "N/A",
            "dividend_yield": "N/A",
            "current_price": data.get("c"), # Current price
            "pe_ratio": "N/A",
            "revenue": "N/A",
            "net_income": "N/A",
            "guidance": "N/A",
            "fifty_two_week_high": data.get("h"), # High price of the day (Not 52wk) - Might need separate API call for that
            "fifty_two_week_low": data.get("l"),  # Low price of the day (Not 52wk)
            "beta": "N/A",
            "_source": "Finnhub Quote (Fallback)" # Indicate data source
        }
    except requests.exceptions.RequestException as e:
        print(f"Finnhub quote request failed for {ticker}: {e}")
        return {"error": f"Finnhub quote failed: {str(e)}"}
    except Exception as e:
        print(f"Error processing Finnhub quote for {ticker}: {e}")
        return {"error": f"Finnhub quote processing error: {str(e)}"}

# --- Main Stock Info Function ---
@st.cache_data(ttl=3600, show_spinner=False) # Cache for 1 hour
def get_stock_info(ticker):
    """Gets stock information using yfinance with caching and Finnhub fallback for basic quote."""
    if not ticker or not isinstance(ticker, str):
        return {"error": "Invalid ticker provided."}

    ticker = ticker.strip().upper()
    print(f"Getting stock info for: {ticker}")

    # 1. Try cache
    cached = load_cached_stock_info(ticker)
    if cached:
        print(f"Returning cached info for {ticker}")
        return cached

    # 2. Try yfinance
    info = None
    try:
        stock = yf.Ticker(ticker)
        # Fetch info. May be slow or fail for invalid tickers.
        info = stock.info
        # Check if info dict is reasonably populated
        if not info or len(info) < 5: # Arbitrary check for minimal data
             print(f"yfinance info for {ticker} seems sparse or empty. Trying fallback.")
             info = None # Treat as failure if too little info
        else:
             print(f"Successfully fetched yfinance info for {ticker}")

    except Exception as e:
        print(f"yfinance info fetch failed for {ticker}: {type(e).__name__}. Trying fallback.")
        info = None # Ensure info is None on exception

    # 3. Prepare result or use fallback
    result = None
    if info:
        # Construct result from yfinance info
        result = {
            "company": info.get("shortName", info.get("longName", ticker)), # Prefer shortName
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"), # Add industry
            "market_cap": info.get("marketCap"), # Keep as number or None
            "dividend_yield": info.get("dividendYield"), # Keep as number or None
            "current_price": info.get("currentPrice", info.get("previousClose")), # Fallback to previous close
            "pe_ratio": info.get("trailingPE"), # Keep as number or None
            "forward_pe": info.get("forwardPE"), # Add forward PE
            "revenue": info.get("totalRevenue"), # Keep as number or None
            "net_income": info.get("netIncomeToCommon"), # Keep as number or None
            "guidance": info.get("forwardPE", "N/A"), # Use forwardPE as proxy? Or N/A
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"), # Keep as number or None
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"), # Keep as number or None
            "beta": info.get("beta"), # Keep as number or None
            "_source": "yfinance"
        }
    else:
        # If yfinance failed, try Finnhub backup for basic quote
        print(f"Attempting Finnhub quote fallback for {ticker}")
        result = get_finnhub_backup(ticker) # This returns dict with "error" key on failure

    # 4. Replace None values with "N/A" for consistent display AFTER caching attempt
    final_result = {}
    if result and "error" not in result:
         for key, value in result.items():
              final_result[key] = value if value is not None else "N/A"
         # Cache the successful result (before adding N/A strings if possible)
         save_stock_info_cache(ticker, result) # Cache the data with potential None values
         print(f"Successfully retrieved and cached data for {ticker} from {final_result.get('_source', 'Unknown')}")
         return final_result # Return the version with N/A strings for display
    elif result and "error" in result:
         print(f"Failed to get stock info for {ticker}: {result['error']}")
         return result # Return the dictionary containing the error message
    else:
         # Should not happen, but catch all
         err_msg = f"Unknown error retrieving stock info for {ticker}"
         print(err_msg)
         return {"error": err_msg}