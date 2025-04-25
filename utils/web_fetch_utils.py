# utils/web_fetch_utils.py
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st # For caching
import pandas as pd # For year checking
import time
from urllib.parse import urlparse, urljoin

# --- NEW IMPORT for Fallback ---
try:
    from duckduckgo_search import DDGS
except ImportError:
    # Provide guidance if library is missing
    st.error("DuckDuckGo Search library not found. Please install it: pip install duckduckgo-search")
    DDGS = None # Set DDGS to None so fallback check fails gracefully


# --- Constants & Helpers ---

# Headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/' # Add a referer
}

# Common patterns for Investor Relations links
IR_PATTERNS = [
    r'investor relations', r'ir\.', r'/investors?', r'shareholders', # Added '/' prefix for path check
    r'financial information', r'company/investors', r'sec filings'
]

# Common patterns for earnings documents/pages
# Added more specific terms and year/quarter patterns
EARNINGS_PATTERNS = [
    r'earnings call transcript', r'earnings transcript', r'call transcript',
    r'earnings release', r'earnings report', r'financial results',
    r'quarterly results', r'earnings webcast', r'earnings presentation',
    r'q\d{1,2}\s?\'?\d{2,4}', r'\d{4}\s?q\d{1,2}', # e.g., Q1 2024, 2024 Q1, Q1 '24
    r'(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter' # e.g., first quarter
]

# --- Original Functions (with minor improvements) ---

@st.cache_data(ttl=86400, show_spinner=False) # Cache results for a day
def find_investor_relations_page(company_name_or_ticker):
    """
    Attempts to find the Investor Relations (IR) page for a company/ticker.
    Uses a simple approach: search common subdomains/paths.
    """
    query = company_name_or_ticker.lower()
    # Basic cleaning for domain guessing
    base_domain = re.sub(r'\s+', '', query) # Remove spaces
    base_domain = re.sub(r'\.(com|org|net|inc|corp|ltd)$', '', base_domain, flags=re.I) # Remove common suffixes
    base_domain = re.sub(r'[^\w\-]+', '', base_domain) # Remove non-alphanumeric except hyphen

    if not base_domain:
        print("Could not generate base domain from query.")
        return None

    # Expanded list of common URL patterns
    common_urls_patterns = [
        f"https://investor.{base_domain}.com",
        f"https://investors.{base_domain}.com",
        f"https://ir.{base_domain}.com",
        f"https://{base_domain}.com/investors",
        f"https://{base_domain}.com/investor-relations",
        f"https://{base_domain}.com/investor",
        f"https://{base_domain}.com/company/investors",
        f"https://{base_domain}.com/ir",
        f"https://{base_domain}.com/shareholders",
        f"https://{base_domain}.com/financial-information",
        # Try www subdomain as well
        f"https://www.{base_domain}.com/investors",
        f"https://www.{base_domain}.com/investor-relations",
    ]

    print(f"Trying IR URLs for base domain '{base_domain}': {common_urls_patterns}")

    for url in common_urls_patterns:
        try:
            # Increase timeout slightly, allow redirects
            response = requests.get(url, headers=HEADERS, timeout=12, allow_redirects=True, verify=True) # Added verify=True (default but explicit)
            response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

            # Check if response text contains IR patterns (more robust check)
            if response.text and any(re.search(p, response.text, re.I) for p in IR_PATTERNS):
                # Check if the final URL after redirects still seems related
                final_url = response.url
                if base_domain in urlparse(final_url).netloc or any(p in final_url.lower() for p in ['investor', 'ir', 'shareholder']):
                    print(f"Found potential IR page: {final_url} (from initial guess: {url})")
                    return final_url # Return the final URL after redirects
            else:
                 print(f"URL {url} accessible but doesn't seem like an IR page (final URL: {response.url}).")


        except requests.exceptions.Timeout:
            print(f"Timeout connecting to {url}")
        except requests.exceptions.ConnectionError as e:
            # This includes DNS errors (like NameResolutionError seen before)
            print(f"Connection error for {url}: {e}")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error for {url}: {e.response.status_code} {e.response.reason}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url}: {e}")
        # Small delay between attempts
        time.sleep(0.2)

    print(f"Could not automatically find IR page for '{company_name_or_ticker}' using common patterns.")
    return None

@st.cache_data(ttl=3600, show_spinner=False) # Cache for 1 hour
def find_latest_earnings_doc(ir_page_url):
    """
    Scrapes an Investor Relations page to find links to recent earnings documents (PDF, Transcript).
    """
    if not ir_page_url:
        return None, None # No IR page found

    try:
        response = requests.get(ir_page_url, headers=HEADERS, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'lxml') # Use lxml for potentially better parsing

        potential_links = []
        current_year = pd.Timestamp.now().year
        years_to_check = [str(current_year), str(current_year - 1)]

        # Find all links on the page
        for link in soup.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text(strip=True).lower() # Get clean text
            combined_text = link_text + " " + href.lower() # Check text and URL

            # Check if link text or URL seems related to earnings and recent year
            is_earnings_link = any(re.search(p, combined_text, re.I) for p in EARNINGS_PATTERNS)
            mentions_recent_year = any(year in combined_text for year in years_to_check)

            # Require both earnings pattern and recent year for higher relevance
            if is_earnings_link and mentions_recent_year:
                # Resolve relative URLs properly
                full_url = urljoin(ir_page_url, href)

                # Basic check for invalid URLs
                if not urlparse(full_url).scheme in ['http', 'https']:
                     continue

                doc_type = None
                if '.pdf' in full_url.lower():
                    doc_type = 'pdf'
                # Be more specific about transcript links
                elif 'transcript' in combined_text or 'earnings call' in combined_text:
                     # Avoid linking to general event pages if possible
                     if '.pdf' not in full_url.lower() and '.mp3' not in full_url.lower() and '.wav' not in full_url.lower() and 'webcast' not in combined_text:
                           doc_type = 'transcript_link'

                # Add more weight if 'transcript' or 'pdf' is explicit in text/url
                weight = 0
                if doc_type == 'pdf': weight += 2
                if doc_type == 'transcript_link': weight += 1
                if 'transcript' in combined_text: weight +=1
                if 'release' in combined_text: weight +=1
                if 'results' in combined_text: weight +=1
                if str(current_year) in combined_text: weight += 2 # Prioritize current year

                if doc_type: # Only consider if we identified a potential type
                    potential_links.append({'url': full_url, 'text': link_text, 'type': doc_type, 'weight': weight})

        # Sort potential links by weight (higher first), then maybe by text length?
        potential_links.sort(key=lambda x: x['weight'], reverse=True)

        if not potential_links:
             print("No links matching earnings patterns and recent year found on IR page.")
             return None, None

        # Return the highest weighted link
        best_link = potential_links[0]
        print(f"Found potential earnings document (Type: {best_link['type']}, Weight: {best_link['weight']}): {best_link['url']}")
        return best_link['type'], best_link['url']


    except requests.exceptions.RequestException as e:
        print(f"Error scraping IR page {ir_page_url}: {e}")
        return None, None
    except Exception as e:
        print(f"Error processing IR page {ir_page_url}: {e}")
        return None, None

# --- Fallback Function ---

def search_for_earnings_doc_fallback(query):
    """
    Uses DuckDuckGo to search directly for recent earnings transcripts or PDFs.
    """
    if not DDGS:
        print("DuckDuckGo Search library not available for fallback.")
        return None, None

    current_year = pd.Timestamp.now().year
    search_query = f'"{query}" earnings call transcript OR report pdf {current_year} OR {current_year-1}'
    print(f"Attempting fallback search: {search_query}")

    try:
        with DDGS(headers=HEADERS, timeout=20) as ddgs:
            # Get maybe top 5-10 results
            results = list(ddgs.text(search_query, max_results=7))

        if not results:
            print("Fallback search returned no results.")
            return None, None

        potential_links = []
        for result in results:
            url = result.get('href')
            title = result.get('title','').lower()
            body = result.get('body','').lower()
            combined_text = title + " " + body + " " + url.lower()

            doc_type = None
            weight = 0

            # Check if URL is a PDF
            if url and '.pdf' in url.lower():
                doc_type = 'pdf'
                weight += 3 # High weight for direct PDF link
            # Check if title/body/URL mention transcript explicitly
            elif url and ('transcript' in combined_text or 'earnings call' in combined_text):
                 doc_type = 'transcript_link'
                 weight += 1

            # Add weight if it seems relevant
            if 'earnings' in combined_text: weight += 1
            if str(current_year) in combined_text: weight += 2
            if str(current_year-1) in combined_text: weight += 1
            if query.lower() in combined_text: weight += 1 # Check if query term is present

            if doc_type and url:
                 potential_links.append({'url': url, 'text': title, 'type': doc_type, 'weight': weight})

        # Sort by weight
        potential_links.sort(key=lambda x: x['weight'], reverse=True)

        if potential_links:
            best_fallback = potential_links[0]
            print(f"Fallback search found potential link (Type: {best_fallback['type']}, Weight: {best_fallback['weight']}): {best_fallback['url']}")
            return best_fallback['type'], best_fallback['url']
        else:
            print("Fallback search results did not yield a likely PDF or transcript link.")
            return None, None

    except Exception as e:
        print(f"Error during fallback search: {e}")
        return None, None

# --- Document Fetching (remains the same) ---

def fetch_document_content(doc_url, doc_type):
    """Fetches content of a PDF or tries to get text from a transcript webpage."""
    print(f"Fetching content for {doc_type} from: {doc_url}")
    try:
        # Use stream=True for potentially large files (like PDF)
        response = requests.get(doc_url, headers=HEADERS, timeout=25, stream=(doc_type=='pdf'), verify=True)
        response.raise_for_status()

        if doc_type == 'pdf':
            # Return raw content bytes for pdfplumber
            # Check content type if possible
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' not in content_type:
                 print(f"Warning: URL ending in .pdf did not return PDF content-type ({content_type}). Trying anyway.")
            return response.content
        elif doc_type == 'transcript_link':
            # Basic text extraction from HTML page
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                 print(f"Warning: Expected HTML for transcript link, got {content_type}. Trying basic text extraction.")
                 return response.text # Return raw text if not HTML

            # Parse HTML
            soup = BeautifulSoup(response.content, 'lxml') # Use response.content for correct encoding handling by bs4

            # Attempt to find main content area (heuristic, needs site-specific adaptation)
            # Look for common tags/attributes used for main content/articles
            main_content = soup.find('article') or \
                           soup.find('main') or \
                           soup.find('div', class_=re.compile(r'content|transcript|body|main|article', re.I)) or \
                           soup.find('div', id=re.compile(r'content|transcript|body|main|article', re.I))

            if main_content:
                 # Remove common noise like headers, footers, navs if possible (basic attempt)
                 for tag_name in ['header', 'footer', 'nav', 'aside', 'script', 'style']:
                     for tag in main_content.find_all(tag_name):
                         tag.decompose()
                 # Get text, trying to preserve paragraphs
                 text = main_content.get_text(separator='\n', strip=True)
            else:
                 # Fallback to just body text if specific content area not found
                 print("Could not find specific main content area, falling back to body text.")
                 body = soup.body
                 if body:
                      for tag_name in ['header', 'footer', 'nav', 'aside', 'script', 'style']:
                          for tag in body.find_all(tag_name):
                              tag.decompose()
                      text = body.get_text(separator='\n', strip=True)
                 else: # Absolute fallback
                     text = soup.get_text(separator='\n', strip=True)

            # Basic cleaning of excessive whitespace
            text = re.sub(r'\n{3,}', '\n\n', text) # Replace 3+ newlines with 2
            return text.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching document {doc_url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing document {doc_url}: {e}")
        return None


# --- NEW Wrapper Function for Primary + Fallback Logic ---

def attempt_fetch_earnings_data(query):
    """
    Tries the primary method (IR page guessing + scraping) first,
    then uses the fallback search if the primary fails.

    Args:
        query (str): Company name or ticker.

    Returns:
        tuple: (content_type, content) where content_type is 'pdf' or 'transcript'
               and content is the bytes (for pdf) or text (for transcript).
               Returns (None, None) if both methods fail.
    """
    print(f"\n--- Attempting Primary Fetch for '{query}' ---")
    ir_page_url = find_investor_relations_page(query)
    doc_type, doc_url = None, None
    if ir_page_url:
        doc_type, doc_url = find_latest_earnings_doc(ir_page_url)

    # If primary method failed to find a doc link
    if not (doc_type and doc_url):
        print(f"\n--- Primary fetch failed. Attempting Fallback Search for '{query}' ---")
        doc_type, doc_url = search_for_earnings_doc_fallback(query)

    # If we have a document URL (from either primary or fallback)
    if doc_type and doc_url:
        print(f"\n--- Fetching content from URL: {doc_url} ---")
        content = fetch_document_content(doc_url, doc_type)
        if content:
            # Determine final type based on successful fetch
            final_type = 'pdf' if doc_type == 'pdf' else 'transcript'
            return final_type, content
        else:
            print("Failed to fetch content from the URL.")
            return None, None
    else:
        print("Could not find a suitable earnings document URL using primary or fallback methods.")
        return None, None