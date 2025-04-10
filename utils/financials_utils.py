import spacy
import re

# Load spaCy model for NER (Named Entity Recognition)
nlp = spacy.load("en_core_web_sm")

def extract_financials(text):
    """Extract financial highlights (revenue, net income, guidance) from the transcript."""
    doc = nlp(text)
    
    # Initialize variables to hold extracted values
    revenue = None
    net_income = None
    guidance = None
    
    # Enhanced regex patterns for capturing numbers with or without commas, dollars, etc.
    revenue_pattern = r"(revenue|sales|turnover)[^a-zA-Z0-9]?\$?([0-9,\.]+(?:\s*(million|billion))?)"
    net_income_pattern = r"(net income|net profit|earnings)[^a-zA-Z0-9]?\$?([0-9,\.]+(?:\s*(million|billion))?)"
    guidance_pattern = r"(guidance|outlook)[^a-zA-Z0-9]?\$?([0-9,\.]+(?:\s*(million|billion))?)"
    
    # Search for the patterns in the transcript
    revenue_match = re.search(revenue_pattern, text, re.IGNORECASE)
    net_income_match = re.search(net_income_pattern, text, re.IGNORECASE)
    guidance_match = re.search(guidance_pattern, text, re.IGNORECASE)
    
    if revenue_match:
        revenue = revenue_match.group(2).replace(",", "")
        if revenue_match.group(3):  # Check for million or billion
            revenue = convert_to_numeric(revenue, revenue_match.group(3))
    
    if net_income_match:
        net_income = net_income_match.group(2).replace(",", "")
        if net_income_match.group(3):  # Check for million or billion
            net_income = convert_to_numeric(net_income, net_income_match.group(3))
    
    if guidance_match:
        guidance = guidance_match.group(2).replace(",", "")
        if guidance_match.group(3):  # Check for million or billion
            guidance = convert_to_numeric(guidance, guidance_match.group(3))
    
    return {
        "revenue": revenue,
        "net_income": net_income,
        "guidance": guidance
    }

def convert_to_numeric(value, scale):
    """Convert values like 'million' or 'billion' to numeric equivalents."""
    value = float(value)
    if scale.lower() == "million":
        value *= 1_000_000
    elif scale.lower() == "billion":
        value *= 1_000_000_000
    return value
