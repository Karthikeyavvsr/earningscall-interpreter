# utils/glossary_utils.py
import os
import json # <-- ADD THIS LINE
import re

def load_glossary():
    """Load the financial glossary JSON from project root."""
    # Assumes glossary/finance_terms.json exists at the project root level (parallel to app/, utils/)
    # If it's elsewhere (e.g., inside data/), adjust the path.
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        glossary_path = os.path.join(project_root, "glossary", "finance_terms.json")

        if not os.path.exists(glossary_path):
             # Try alternative path (e.g., inside data/)
             alt_glossary_path = os.path.join(project_root, "data", "finance_terms.json")
             if os.path.exists(alt_glossary_path):
                  glossary_path = alt_glossary_path
             else:
                  # If neither found, raise the original error pointing to the first tried path
                  raise FileNotFoundError(f"Glossary file not found at {os.path.join(project_root, 'glossary', 'finance_terms.json')} or {alt_glossary_path}")

        with open(glossary_path, "r", encoding='utf-8') as f: # Added encoding
            return json.load(f)
    except FileNotFoundError as e:
         # Re-raise for app.py to catch and display nicely
         raise e
    except json.JSONDecodeError as e:
         # Handle potential errors if the JSON file is malformed
         print(f"Error decoding JSON from glossary file: {glossary_path}, Error: {e}")
         raise ValueError(f"Glossary file '{os.path.basename(glossary_path)}' is not valid JSON.") from e
    except Exception as e:
         # Catch other potential errors during loading
         print(f"Unexpected error loading glossary: {e}")
         raise e


def find_terms_in_text(glossary, text):
    """Find glossary terms that appear in the given transcript text."""
    if not glossary or not text:
        return {}
    found = {}
    # Sort terms by length descending to match longer terms first (e.g., "Net Income" before "Income")
    sorted_terms = sorted(glossary.keys(), key=len, reverse=True)
    for term in sorted_terms:
        definition = glossary[term]
        # Use word boundaries (\b) to match whole words/phrases only, case-insensitive
        pattern = rf"\b{re.escape(term)}\b"
        try:
            if re.search(pattern, text, re.IGNORECASE):
                # Optional: could add logic here to avoid adding sub-terms if a longer term already matched
                # For now, add all matches
                found[term] = definition
        except re.error as e:
             print(f"Regex error searching for term '{term}': {e}") # Handle potential regex errors
             continue # Skip problematic terms
    return found

def search_glossary(glossary, keyword):
    """Search glossary for a term or partial keyword (case-insensitive)."""
    if not glossary or not keyword:
        return {}
    results = {}
    keyword_lower = keyword.lower()
    for term, definition in glossary.items():
        # Check if keyword is in term or definition
        if keyword_lower in term.lower() or keyword_lower in definition.lower():
            results[term] = definition
    return results