# utils/finnhub_utils.py

import requests

FINNHUB_API_KEY = "YOur key goes here"
BASE_URL = "https://finnhub.io/api/v1"

def get_finnhub_quote(symbol):
    try:
        url = f"{BASE_URL}/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
        response = requests.get(url)
        data = response.json()
        return {
            "current_price": data.get("c"),
            "high": data.get("h"),
            "low": data.get("l"),
            "open": data.get("o"),
            "previous_close": data.get("pc")
        }
    except Exception as e:
        return {"error": str(e)}
