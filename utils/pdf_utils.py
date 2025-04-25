# utils/pdf_utils.py
import pdfplumber
import re
from transformers import pipeline # Reusing summarization pipeline

# Initialize summarization pipeline (consider moving to a central place if used elsewhere)
# Using a different model potentially better suited for longer documents
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file object.

    Args:
        pdf_file: A file-like object from st.file_uploader.

    Returns:
        str: Extracted text content or None if error.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text: # Check if text extraction returned something
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

def generate_summary(text, max_length=250, min_length=50):
    """
    Generates a summary of the provided text using a HuggingFace pipeline.

    Args:
        text (str): The text to summarize.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: The generated summary or an error message.
    """
    if not text or len(text.strip()) < min_length * 2: # Basic check if text is too short
        return "Text too short to summarize effectively."
    try:
        # Summarizer can handle ~1024 tokens, chunking might be needed for very long docs
        # For simplicity, we'll summarize the first ~4000 characters if too long
        max_chunk_size = 4000 # Adjust based on model limits and typical report length
        input_text = text[:max_chunk_size]

        summary_result = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary_result[0]['summary_text']
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# You could add more functions here, e.g., finding specific sections
# using regex if needed for trend analysis, but the request was simpler.