import re
import numpy as np
from textblob import TextBlob

# Expandable keyword lists for scoring
UNCERTAINTY_KEYWORDS = ["uncertain", "uncertainty", "doubt", "depends", "possibly", "potential"]
LITIGIOUS_KEYWORDS = ["lawsuit", "regulation", "investigation", "regulatory", "compliance", "litigation"]
MODAL_VERBS = ["may", "might", "could", "would", "should", "possibly", "likely"]
CONFIDENT_PHRASES = ["we will", "we are confident", "we expect", "we anticipate"]
FUTURE_KEYWORDS = ["next quarter", "next year", "we expect", "we anticipate", "forecast", "guidance"]


def get_basic_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def count_keyword_occurrences(text, keywords):
    return sum(len(re.findall(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE)) for word in keywords)


def get_uncertainty_score(text):
    return count_keyword_occurrences(text, UNCERTAINTY_KEYWORDS)


def get_litigious_score(text):
    return count_keyword_occurrences(text, LITIGIOUS_KEYWORDS)


def get_modal_verb_score(text):
    return count_keyword_occurrences(text, MODAL_VERBS)


def get_confidence_score(text):
    return count_keyword_occurrences(text, CONFIDENT_PHRASES)


def get_forward_looking_sentiment(text):
    forward_sentences = [s for s in text.split('.') if any(kw in s.lower() for kw in FUTURE_KEYWORDS)]
    if not forward_sentences:
        return 0.0
    sentiments = [TextBlob(s).sentiment.polarity for s in forward_sentences]
    return np.mean(sentiments)


def get_sentiment_drift(text):
    parts = np.array_split(text.split('.'), 3)
    drifts = [TextBlob(".".join(p)).sentiment.polarity for p in parts if p is not None and len(p) > 0]
    return drifts if len(drifts) == 3 else [0, 0, 0]


def get_sentiment_volatility(text):
    paragraphs = text.split('\n')
    scores = [TextBlob(p).sentiment.polarity for p in paragraphs if len(p.strip()) > 10]
    return np.std(scores) if scores else 0


def extract_qa_section(text):
    match = re.search(r'(question.*answer|q&a|qa session)', text, re.IGNORECASE)
    if match:
        return text[match.start():]
    return ""


def get_analyst_sentiment(text):
    qa_text = extract_qa_section(text)
    if not qa_text:
        return 0.0
    return TextBlob(qa_text).sentiment.polarity


def get_entity_sentiment(text, entities):
    result = {}
    for ent in entities:
        # Get a few surrounding words for context
        pattern = rf"(.{{0,50}}\b{re.escape(ent)}\b.{{0,50}})"
        windows = re.findall(pattern, text, re.IGNORECASE)
        scores = [TextBlob(w).sentiment.polarity for w in windows]
        result[ent] = np.mean(scores) if scores else 0.0
    return result
