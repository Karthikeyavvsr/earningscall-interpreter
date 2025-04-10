import os
import json
import re

def load_glossary():
    """Load the financial glossary JSON from project root."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    glossary_path = os.path.join(project_root, "glossary", "finance_terms.json")

    if not os.path.exists(glossary_path):
        raise FileNotFoundError(f"Glossary file not found at {glossary_path}")

    with open(glossary_path, "r") as f:
        return json.load(f)

def find_terms_in_text(glossary, text):
    """Find glossary terms that appear in the given transcript text."""
    found = {}
    for term, definition in glossary.items():
        pattern = rf"\b{re.escape(term)}\b"
        if re.search(pattern, text, re.IGNORECASE):
            found[term] = definition
    return found

def search_glossary(glossary, keyword):
    """Search glossary for a term or partial keyword."""
    results = {}
    for term, definition in glossary.items():
        if keyword.lower() in term.lower():
            results[term] = definition
    return results