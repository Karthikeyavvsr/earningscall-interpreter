import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import sentiment_utils as su  
from utils import glossary_utils as gu
from utils.chatbot_utils import answer_question
from utils.stock_info import extract_company_name, map_to_ticker, get_stock_info
from utils.financials_utils import extract_financials

import streamlit as st
import whisper
import os
import tempfile
from textblob import TextBlob
import spacy
import yfinance as yf
import pandas as pd



# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Page setup
st.set_page_config(page_title="EarningsPulse", layout="wide")
st.title("\U0001F4CA EarningsPulse: AI-Driven Earnings Call Interpreter")
st.markdown("Analyze earnings calls, financial reports, and generate actionable insights in real time.")

# ---------------------------- Sidebar ---------------------------- #
st.sidebar.header("Upload Data or Search")
source = st.sidebar.radio("Select input type:", ["Audio Upload", "Transcript Text", "PDF Report (Coming Soon)", "Search by Ticker (Coming Soon)"])
ticker = st.sidebar.text_input("Stock Ticker (optional)", "")

# ---------------------------- Tabbed Layout ---------------------------- #
tabs = st.tabs(["\U0001F4C4 Transcript", "\U0001F4CA Sentiment & Entities", "\U0001F4DA Glossary", "\U0001F4F0 Q&A", "\U0001F4C8 Stock Info", "\U0001F4DD Summary"])

# ---------------------------- Tab 1: Transcript ---------------------------- #
with tabs[0]:
    st.header("Transcript View")

    transcript_text = ""

    if source == "Audio Upload":
        audio_file = st.file_uploader("Upload Earnings Call Audio", type=["mp3", "wav"])
        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_file_path = tmp_file.name

            st.info("Transcribing audio... please wait ⏳")
            model = whisper.load_model("base")
            result = model.transcribe(tmp_file_path)
            transcript_text = result["text"]
            st.success("✅ Transcription complete!")
            st.text_area("Transcript Output", transcript_text, height=400)
            st.session_state["transcript_text"] = transcript_text

    elif source == "Transcript Text":
        transcript_text = st.text_area("Paste Transcript", height=400)
        st.session_state["transcript_text"] = transcript_text

# ---------------------------- Tab 2: Sentiment & Entities ---------------------------- #
with tabs[1]:
    st.header("Sentiment & Named Entities")

    transcript = st.session_state.get("transcript_text", "")
    if not transcript:
        st.warning("Transcript not available. Please upload or paste it in Tab 1.")
    else:
        # ----------- NLP Metrics ----------- #
        polarity, subjectivity = su.get_basic_sentiment(transcript)
        uncertainty = su.get_uncertainty_score(transcript)
        litigious = su.get_litigious_score(transcript)
        modal_score = su.get_modal_verb_score(transcript)
        confidence_score = su.get_confidence_score(transcript)
        forward_sentiment = su.get_forward_looking_sentiment(transcript)
        drift = su.get_sentiment_drift(transcript)
        volatility = su.get_sentiment_volatility(transcript)
        analyst_sentiment = su.get_analyst_sentiment(transcript)

        # ----------- Display as Metrics ----------- #
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Polarity", f"{polarity:.2f}")
            st.metric("Uncertainty Score", uncertainty)
            st.metric("Modal Verbs Score", modal_score)
        with col2:
            st.metric("Subjectivity", f"{subjectivity:.2f}")
            st.metric("Litigious Score", litigious)
            st.metric("Confidence Score", confidence_score)
        with col3:
            st.metric("Forward-Looking Sentiment", f"{forward_sentiment:.2f}")
            st.metric("Sentiment Volatility", f"{volatility:.2f}")
            st.metric("Analyst Sentiment", f"{analyst_sentiment:.2f}")

        # ----------- Drift Visualization ----------- #
        st.subheader("\U0001F9EE Sentiment Drift Across Sections")
        st.line_chart({"Start": [drift[0]], "Middle": [drift[1]], "End": [drift[2]]})

        # ----------- Named Entities ----------- #
        st.subheader("\U0001F4CC Named Entities")
        doc = nlp(transcript)
        entity_data = [{"Text": ent.text, "Label": ent.label_} for ent in doc.ents]
        if entity_data:
            st.dataframe(entity_data)
        else:
            st.info("No named entities detected.")

        # ----------- Entity-Specific Sentiment ----------- #
        st.subheader("\U0001F9EA Entity-Specific Sentiment")
        if entity_data:
            entities = list(set(ent['Text'] for ent in entity_data))[:10]  # limit for performance
            entity_sentiments = su.get_entity_sentiment(transcript, entities)
            st.json(entity_sentiments)
    
    # ---------------------------- Tab 3: Glossary ---------------------------- #
with tabs[2]:
    st.header("📘 Glossary - Financial Jargon Buster")

    glossary = gu.load_glossary()
    transcript = st.session_state.get("transcript_text", "")

    # Highlighted glossary matches in transcript
    if transcript:
        st.subheader("📌 Terms Detected in Transcript")
        matches = gu.find_terms_in_text(glossary, transcript)
        if matches:
            for term, definition in matches.items():
                st.markdown(f"**{term}**: {definition}")
        else:
            st.info("No glossary terms matched the transcript.")
    else:
        st.warning("Transcript is empty. Upload or paste it in Tab 1.")

    # Manual search
    st.subheader("🔎 Search Glossary")
    keyword = st.text_input("Search by term or keyword")
    if keyword:
        results = gu.search_glossary(glossary, keyword)
        if results:
            for term, definition in results.items():
                st.markdown(f"**{term}**: {definition}")
        else:
            st.error("No results found for that term.")

    # ---------------------------- Tab 4: Q&A ---------------------------- #
with tabs[3]:
    st.header("💬 Ask a Question about the Earnings Call")
    transcript = st.session_state.get("transcript_text", "")
    if not transcript:
        st.warning("Transcript not available. Please upload or paste it in Tab 1.")
    else:
        question = st.text_input("Ask a question about the call (e.g., 'What was said about guidance?')")
        if question:
            answer, score = answer_question(transcript, question)
            st.success(f"Answer: {answer}")
            st.caption(f"Confidence Score: {score:.2f}")

# ---------------------------- Tab 5: Stock Info ---------------------------- #
with tabs[4]:
    st.header("📊 Stock Info")

    # Automatically extract company name from transcript
    transcript = st.session_state.get("transcript_text", "")
    if transcript:
        companies = extract_company_name(transcript)
        if companies:
            company_name = companies[0]  # Let's use the first recognized company
            ticker = map_to_ticker(company_name)
            if ticker:
                st.write(f"Detected company: {company_name} (Ticker: {ticker})")

                # Fetch stock data using the new function
                stock_info = get_stock_info(ticker)

                if stock_info:
                    st.subheader(f"Stock Info for {company_name}")
                    st.write(f"**Company Name**: {stock_info['company_name']}")
                    st.write(f"**Current Price**: ${stock_info['current_price']:.2f}")
                    st.write(f"**Market Cap**: {stock_info['market_cap']}")
                    st.write(f"**P/E Ratio**: {stock_info['pe_ratio']}")
                    st.write(f"**52 Week High**: ${stock_info['fifty_two_week_high']}")
                    st.write(f"**52 Week Low**: ${stock_info['fifty_two_week_low']}")
                    
                    # Handle dividend yield properly
                    dividend_yield = stock_info.get('dividend_yield', None)
                    if isinstance(dividend_yield, (int, float)):
                        st.write(f"**Dividend Yield**: {dividend_yield * 100:.2f}%")
                    else:
                        st.write(f"**Dividend Yield**: {dividend_yield if dividend_yield is not None else 'N/A'}")

                    st.write(f"**Beta**: {stock_info['beta']}")

                    # Display stock chart (Last 7 days)
                    hist_data = yf.Ticker(ticker).history(period="1d", interval="1h")  # Hourly data
                    st.subheader("📈 Stock Price Chart (Last 7 Days)")
                    st.line_chart(hist_data['Close'])
                else:
                    st.error(f"Error fetching stock data for {ticker}.")
            else:
                st.error(f"Could not map company {company_name} to a stock ticker.")
        else:
            st.warning("No company name detected in the transcript.")
    else:
        st.warning("Transcript is empty. Please upload or paste it in Tab 1.")


# ---------------------------- Tab 6: Summary ---------------------------- #
with tabs[5]:
    st.header("📝 Earnings Call Summary")

    # Get transcript and perform NLP analysis
    transcript = st.session_state.get("transcript_text", "")
    
    if not transcript:
        st.warning("Transcript is empty. Please upload or paste it in Tab 1.")
    else:
        # Basic Summary of Metrics (Sentiment)
        polarity, subjectivity = su.get_basic_sentiment(transcript)
        forward_sentiment = su.get_forward_looking_sentiment(transcript)
        analyst_sentiment = su.get_analyst_sentiment(transcript)

        # Extract company name from the transcript
        companies = extract_company_name(transcript)
        if companies:
            company_name = companies[0]  # Use the first detected company
            ticker = map_to_ticker(company_name)

            if ticker:
                stock_info = get_stock_info(ticker)  # Fetch stock data from Yahoo Finance

                if stock_info:
                    st.subheader("📊 Summary of Earnings Call")
                    st.write(f"**Polarity**: {polarity:.2f} (Indicates sentiment direction)")
                    st.write(f"**Subjectivity**: {subjectivity:.2f} (Objective vs. Subjective)")
                    st.write(f"**Forward-Looking Sentiment**: {forward_sentiment:.2f}")
                    st.write(f"**Analyst Sentiment**: {analyst_sentiment:.2f}")

                    # Display financial highlights from Yahoo Finance
                    st.subheader("Key Financial Highlights")

                    # Display revenue (check if it's a valid value)
                    if isinstance(stock_info["revenue"], (int, float)) and not pd.isna(stock_info["revenue"]):
                        st.write(f"**Revenue**: ${stock_info['revenue']:,}")
                    else:
                        st.write("**Revenue**: Data not available.")
                    
                    # Display net income (check if it's a valid value)
                    if isinstance(stock_info["net_income"], (int, float)) and not pd.isna(stock_info["net_income"]):
                        st.write(f"**Net Income**: ${stock_info['net_income']:,}")
                    else:
                        st.write("**Net Income**: Data not available.")
                    
                    # Display guidance (check if it's a valid value)
                    if stock_info["guidance"] and stock_info["guidance"] != 'N/A':
                        st.write(f"**Guidance**: {stock_info['guidance']}")
                    else:
                        st.write("**Guidance**: Data not available.")
                else:
                    st.error(f"Could not fetch stock data for {ticker}.")
            else:
                st.error(f"Could not map company {company_name} to a stock ticker.")
        else:
            st.warning("No company name detected in the transcript.")
