# utils/chatbot_utils.py
import google.generativeai as genai
import os
import streamlit as st # To access secrets

# --- Gemini API Configuration ---
# IMPORTANT: Configure your API key securely!
# Option 1: Use Streamlit Secrets (Recommended for deployed apps)
# Ensure you have GOOGLE_API_KEY set in your Streamlit secrets (e.g., .streamlit/secrets.toml)
try:
    # Access the key from Streamlit secrets if available
    gemini_api_key = st.secrets.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        # Fallback to environment variable if not in secrets
        gemini_api_key = os.environ.get("GOOGLE_API_KEY")

    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        print("Gemini API Key configured.")
        # Initialize the model (e.g., gemini-1.5-flash)
        # Choose model based on your needs (cost, performance)
        # gemini-1.5-flash is generally fast and capable for this task
        model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_configured = True
    else:
        print("ERROR: GOOGLE_API_KEY not found in Streamlit secrets or environment variables.")
        model = None
        gemini_configured = False

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None
    gemini_configured = False


# --- Updated Q&A Function ---

def answer_question(context, question):
    """
    Uses the Gemini API to answer a question based on the provided context.

    Args:
        context (str): The transcript or report text.
        question (str): The user's query.

    Returns:
        str: The generated answer, or an error message if failed.
             Note: Confidence score is not directly applicable like extractive QA.
    """
    if not gemini_configured or not model:
        return "Error: Gemini API is not configured. Please set the GOOGLE_API_KEY."

    if not context:
        return "Error: No context (document text) provided."
    if not question:
        return "Error: No question provided."

    # Construct a clear prompt for the Gemini model
    # Emphasize using ONLY the provided document
    prompt = f"""Based *only* on the information contained in the following document, please answer the question provided. Do not use any external knowledge or information not present in the text. If the answer cannot be found within the document, please state that explicitly.

DOCUMENT:
---
{context[:30000]}
---

QUESTION: {question}

ANSWER:""" # Limit context length passed to API if necessary (e.g., first 30k characters)

    try:
        # Set safety settings if needed (optional, defaults are usually reasonable)
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        # ]
        response = model.generate_content(
            prompt,
            # safety_settings=safety_settings # Uncomment to apply specific settings
            )

        # Check for safety flags or empty response before accessing text
        if response.parts:
            # Access the text content safely
            answer_text = response.text
            return answer_text.strip()
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             return f"Error: Request blocked due to {response.prompt_feedback.block_reason}. The question or context might violate safety policies."
        else:
             # Handle cases where response might be empty without explicit blocking
             print(f"Gemini response received but has no parts: {response}")
             return "Error: Received an empty response from the API."

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Provide a more specific error if possible
        if "API key not valid" in str(e):
             return "Error: Invalid Gemini API Key. Please check your configuration."
        return f"Error: An error occurred while generating the answer: {str(e)}"