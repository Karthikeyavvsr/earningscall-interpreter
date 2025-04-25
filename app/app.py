# app/app.py
import sys
import os
import json # Needed for glossary loading check

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import libraries FIRST (standard practice)
import streamlit as st
import whisper
import tempfile
from textblob import TextBlob
import spacy
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import io

# --- Call set_page_config() as the FIRST Streamlit command ---
st.set_page_config(page_title="EarningsPulse", layout="wide")
# --- END set_page_config() ---

# Import utilities AFTER set_page_config (generally safe, but crucial commands come first)
from utils import sentiment_utils as su
from utils import glossary_utils as gu
# --- Import chatbot utils ---
from utils.chatbot_utils import answer_question, gemini_configured # Import both
# --- Import stock utils ---
from utils.stock_info import extract_company_name, map_to_ticker, get_stock_info
# --- Import other utils ---
from utils.financials_utils import extract_financials
from utils import pdf_utils
from utils import predictive_model_utils as pmu
from utils import web_fetch_utils as wfu


# --- Load Models (Consider loading only when needed) ---
# Load spaCy model safely
@st.cache_resource # Cache the loaded model
def load_spacy_model(model_name="en_core_web_sm"):
    """Loads the spaCy model and handles errors."""
    try:
        return spacy.load(model_name)
    except OSError:
        st.error(f"SpaCy model '{model_name}' not found. Please run 'python -m spacy download {model_name}'")
        st.stop()
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        st.stop()

# Load the model AFTER set_page_config
nlp = load_spacy_model()

# --- App Title and Markdown (Now safe to call after set_page_config) ---
st.title("📈 EarningsPulse: AI-Driven Earnings Call Interpreter")
st.markdown("Analyze earnings calls, financial reports, predict trends, and fetch data automatically.")

# --- Session State Initialization ---
default_keys = {
    "transcript_text": "", "pdf_text": "", "active_tab": "📑 Transcript / PDF",
    "ticker_symbol": "", "company_name": "", "last_source": "Transcript Text",
    "fetch_trigger": None, "analysis_text": "", "error_message": None,
    "ai_summary": None, # To store generated summary
    # Keys to manage file processing state and avoid reprocessing on reruns
    "audio_uploader_processed_id": None, "audio_uploader_trigger_process": False,
    "pdf_uploader_processed_id": None, "pdf_uploader_trigger_process": False,
    # Keys to store prediction results
    "pred_result_Intraday (Next Hour)": None, "pred_confidence_Intraday (Next Hour)": 0.0, "pred_explanation_Intraday (Next Hour)": "Prediction not run yet.",
    "pred_result_Short-Term (Next 7 Days)": None, "pred_confidence_Short-Term (Next 7 Days)": 0.0, "pred_explanation_Short-Term (Next 7 Days)": "Prediction not run yet.",
    "pred_result_Long-Term (Next 30 Days)": None, "pred_confidence_Long-Term (Next 30 Days)": 0.0, "pred_explanation_Long-Term (Next 30 Days)": "Prediction not run yet.",
}
for key, default_value in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Sidebar ---
# (Sidebar code remains the same as the previous complete version)
st.sidebar.header("Upload Data or Search")
source_options = ["Transcript Text", "Audio Upload", "PDF Upload", "Smart Fetch"]
try:
    current_source_index = source_options.index(st.session_state.get("last_source", "Transcript Text"))
except ValueError:
    current_source_index = 0

source = st.sidebar.radio(
    "Select input source:", source_options, index=current_source_index, key="source_radio")
st.session_state["last_source"] = source

# Conditional input logic
transcript_input = ""
audio_uploader_key = "audio_uploader"
pdf_uploader_key = "pdf_uploader"

if source == "Transcript Text":
    transcript_input = st.sidebar.text_area("Paste Transcript", height=150, key="transcript_paste_area")
    if transcript_input and transcript_input != st.session_state.get("transcript_text", ""):
        st.session_state["transcript_text"] = transcript_input
        st.session_state["pdf_text"] = ""
        st.session_state["analysis_text"] = transcript_input
        st.session_state["active_tab"] = "📑 Transcript / PDF"
        st.session_state["error_message"] = None
        st.session_state["ai_summary"] = None # Clear summary on new text
        st.rerun()
elif source == "Audio Upload":
    audio_file = st.sidebar.file_uploader(
        "Upload Earnings Call Audio", type=["mp3", "wav", "m4a", "ogg"], key=audio_uploader_key) # Added more types
    if audio_file is not None and audio_file.id != st.session_state.get(f'{audio_uploader_key}_processed_id'):
         st.session_state[f'{audio_uploader_key}_trigger_process'] = True
         st.session_state[f'{audio_uploader_key}_processed_id'] = audio_file.id
         st.rerun()
elif source == "PDF Upload":
    pdf_file = st.sidebar.file_uploader(
        "Upload Earnings Report PDF", type=["pdf"], key=pdf_uploader_key)
    if pdf_file is not None and pdf_file.id != st.session_state.get(f'{pdf_uploader_key}_processed_id'):
        st.session_state[f'{pdf_uploader_key}_trigger_process'] = True
        st.session_state[f'{pdf_uploader_key}_processed_id'] = pdf_file.id
        st.rerun()
elif source == "Smart Fetch":
    fetch_query = st.sidebar.text_input("Enter Company Name or Ticker for Auto-Fetch", key="fetch_query_input")
    if st.sidebar.button("Fetch Latest Earnings Data", key="fetch_button"):
        if fetch_query:
            st.session_state["fetch_trigger"] = fetch_query
            st.session_state["error_message"] = None
            st.session_state["ai_summary"] = None # Clear summary
            st.rerun()
        else:
            st.sidebar.warning("Please enter a company name or ticker.")

# Ticker Input
manual_ticker_input_key = "manual_ticker"
current_ticker = st.session_state.get("ticker_symbol", "")
manual_ticker_input = st.sidebar.text_input(
    "Stock Ticker (Manual Override)", value=current_ticker, key=manual_ticker_input_key)
if manual_ticker_input and manual_ticker_input.strip().upper() != current_ticker:
    st.session_state["ticker_symbol"] = manual_ticker_input.strip().upper()
    st.rerun()

# Display Errors
if st.session_state.get("error_message"):
    st.error(st.session_state["error_message"])
    st.session_state["error_message"] = None # Clear after display


# --- Main Content Area Processing Logic ---
# (Smart Fetch, Audio Upload, PDF Upload logic remains the same as previous version)

# Handle Smart Fetch Trigger (with immediate ticker resolution)
if st.session_state.get("fetch_trigger"):
    query = st.session_state.pop("fetch_trigger")
    st.info(f"Processing Smart Fetch query: '{query}'...")
    fetched_content, fetched_source_type, final_content_type = None, None, None
    resolved_ticker_from_query = None

    # 1. Resolve query to ticker BEFORE fetching
    with st.spinner(f"Checking if '{query}' is or maps to a ticker..."):
        if 1 <= len(query) <= 6 and query.isalnum() and not query.isdigit(): # Basic check for ticker-like format
             temp_info = get_stock_info(query)
             if temp_info and "error" not in temp_info:
                  resolved_ticker_from_query = query.upper()
             else: print(f"Query '{query}' looks like ticker but validation failed.")
        if not resolved_ticker_from_query:
             mapped_ticker = map_to_ticker(query)
             if mapped_ticker: resolved_ticker_from_query = mapped_ticker.strip().upper()

        if resolved_ticker_from_query:
             st.session_state["ticker_symbol"] = resolved_ticker_from_query
             temp_info = get_stock_info(resolved_ticker_from_query) # Get info for name
             if temp_info and "error" not in temp_info and temp_info.get("company") != "N/A":
                  st.session_state["company_name"] = temp_info.get("company")
             st.success(f"Identified ticker '{resolved_ticker_from_query}' from query.")
        else: st.warning(f"Could not resolve '{query}' to a ticker directly.")

    # 2. Attempt fetch
    with st.spinner("Searching for earnings data (trying primary then fallback)..."):
        try:
            final_content_type, content_data = wfu.attempt_fetch_earnings_data(query)
        except Exception as e:
             st.error(f"Fetch attempt error: {e}"); st.session_state["error_message"] = f"Fetch error: {e}"
             final_content_type, content_data = None, None

    # 3. Process result
    if final_content_type and content_data:
        st.success("Successfully fetched potential earnings document content.")
        if final_content_type == 'pdf':
            try:
                extracted_pdf_text = pdf_utils.extract_text_from_pdf(io.BytesIO(content_data))
                if extracted_pdf_text: fetched_content, fetched_source_type = extracted_pdf_text, "PDF Upload"
                else: st.session_state["error_message"] = "Fetched PDF, but couldn't extract text."
            except Exception as e: st.session_state["error_message"] = f"Error processing fetched PDF: {e}"
        elif final_content_type == 'transcript':
            fetched_content, fetched_source_type = content_data, "Transcript Text"

        if fetched_content and fetched_source_type:
            st.session_state["transcript_text"] = fetched_content if fetched_source_type == "Transcript Text" else ""
            st.session_state["pdf_text"] = fetched_content if fetched_source_type == "PDF Upload" else ""
            st.session_state["analysis_text"] = fetched_content
            st.session_state["active_tab"] = "📑 Transcript / PDF"
            st.session_state["last_source"] = fetched_source_type
            company_name_fetched, ticker_fetched = extract_company_name(fetched_content)
            if company_name_fetched and not st.session_state.get("company_name"): st.session_state["company_name"] = company_name_fetched
            current_ticker = st.session_state.get("ticker_symbol", "")
            if ticker_fetched and not current_ticker: st.session_state["ticker_symbol"] = ticker_fetched.strip().upper()
            st.rerun()
        else: st.rerun()
    else:
        if not st.session_state.get("error_message"): st.session_state["error_message"] = f"Could not automatically find and fetch recent earnings data for '{query}'. Please upload manually."
        st.rerun()

# Handle Audio Upload
if st.session_state.get(f'{audio_uploader_key}_trigger_process'):
    st.session_state[f'{audio_uploader_key}_trigger_process'] = False # Consume trigger
    audio_file_to_process = st.session_state.get(audio_uploader_key)
    if audio_file_to_process:
        st.info("Transcribing audio...")
        audio_bytes = io.BytesIO(audio_file_to_process.getvalue())
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(audio_bytes.read()); tmp_file_path = tmp_file.name
        try:
            with st.spinner("Loading transcription model..."): whisper_model = whisper.load_model("base")
            with st.spinner("Transcribing..."): result = whisper_model.transcribe(tmp_file_path)
            os.remove(tmp_file_path)
            transcript_text_audio = result.get("text", "")
            if transcript_text_audio:
                 st.session_state.update({
                     "transcript_text": transcript_text_audio, "pdf_text": "", "analysis_text": transcript_text_audio,
                     "ai_summary": None })
                 st.success("✅ Transcription complete!")
                 company_audio, ticker_audio = extract_company_name(transcript_text_audio)
                 if company_audio: st.session_state["company_name"] = company_audio
                 current_manual_ticker = st.session_state.get("ticker_symbol", "")
                 if ticker_audio and not current_manual_ticker: st.session_state["ticker_symbol"] = ticker_audio.strip().upper()
                 st.rerun()
            else: st.session_state["error_message"] = "Transcription failed or produced empty text."; st.rerun()
        except Exception as e:
            st.session_state["error_message"] = f"Error during transcription: {e}"
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path): os.remove(tmp_file_path)
            st.rerun()

# Handle PDF Upload
if st.session_state.get(f'{pdf_uploader_key}_trigger_process'):
    st.session_state[f'{pdf_uploader_key}_trigger_process'] = False # Consume trigger
    pdf_file_to_process = st.session_state.get(pdf_uploader_key)
    if pdf_file_to_process:
        st.info("Extracting text from PDF...")
        try:
            pdf_text_content = pdf_utils.extract_text_from_pdf(pdf_file_to_process)
            if pdf_text_content:
                st.session_state.update({
                    "pdf_text": pdf_text_content, "transcript_text": "", "analysis_text": pdf_text_content,
                    "ai_summary": None })
                st.success("✅ PDF text extraction complete!")
                company_name_pdf, ticker_pdf = extract_company_name(pdf_text_content)
                if company_name_pdf: st.session_state["company_name"] = company_name_pdf
                current_manual_ticker = st.session_state.get("ticker_symbol", "")
                if ticker_pdf and not current_manual_ticker: st.session_state["ticker_symbol"] = ticker_pdf.strip().upper()
                st.rerun()
            else:
                st.session_state["error_message"] = "Could not extract text from the PDF."; st.rerun()
        except Exception as e: st.session_state["error_message"] = f"Error processing PDF: {e}"; st.rerun()


# --- Define Tabs ---
tab_titles = [ "📑 Transcript / PDF", "📊 Sentiment & Entities", "📘 Glossary", "💬 Q&A",
    "📈 Stock Info & Charts", "🔮 Predictive Analytics", "📝 Summary", ]
tabs = st.tabs(tab_titles)
analysis_text = st.session_state.get("analysis_text", "") # Use state value

# --- Tab Implementations ---
# (Tab 1: Transcript / PDF View - remains the same)
with tabs[0]:
    st.header("Document Content")
    if analysis_text:
        display_text_type = "Extracted PDF Content" if st.session_state.get("pdf_text") else "Transcript"
        st.subheader(display_text_type)
        st.text_area("Content:", analysis_text, height=500, key="doc_view_area")
    else: st.info("Load content using the sidebar.")

# (Tab 2: Sentiment & Entities - remains the same)
with tabs[1]:
    st.header("Sentiment Analysis & Named Entities")
    if not analysis_text: st.warning("No content loaded.")
    else:
        try:
            polarity, subjectivity = su.get_basic_sentiment(analysis_text)
            uncertainty, litigious = su.get_uncertainty_score(analysis_text), su.get_litigious_score(analysis_text)
            modal_score, confidence_score = su.get_modal_verb_score(analysis_text), su.get_confidence_score(analysis_text)
            forward_sentiment, drift = su.get_forward_looking_sentiment(analysis_text), su.get_sentiment_drift(analysis_text)
            volatility = su.get_sentiment_volatility(analysis_text)
            analyst_sentiment = su.get_analyst_sentiment(analysis_text) if st.session_state.get("transcript_text") else "N/A"
            st.subheader("Overall Sentiment Scores"); col1, col2, col3 = st.columns(3)
            def fmt(v, p=2, is_int=False): return f"{v:.{p}f}" if isinstance(v, (int, float)) else ("{:,}".format(v) if is_int and isinstance(v, int) else str(v))
            with col1: st.metric("Polarity", fmt(polarity)); st.metric("Uncertainty", fmt(uncertainty, 0, True)); st.metric("Modal Verbs", fmt(modal_score, 0, True))
            with col2: st.metric("Subjectivity", fmt(subjectivity)); st.metric("Litigious", fmt(litigious, 0, True)); st.metric("Confidence", fmt(confidence_score, 0, True))
            with col3: st.metric("Fwd-Looking", fmt(forward_sentiment)); st.metric("Volatility", fmt(volatility)); st.metric("Analyst (Q&A)", fmt(analyst_sentiment))
            if isinstance(drift, list) and len(drift)==3 and all(isinstance(x,(int,float)) for x in drift):
                 st.subheader("🧭 Sentiment Drift"); drift_df=pd.DataFrame({'Sentiment':drift}, index=['Start','Middle','End']); st.line_chart(drift_df)
            st.divider(); st.subheader("📌 Named Entities")
            try:
                doc = nlp(analysis_text[:1000000]); entity_data = [{"Text": ent.text, "Label": ent.label_} for ent in doc.ents]
                if entity_data: st.dataframe(entity_data, height=300, use_container_width=True)
                else: st.info("No named entities detected.")
            except Exception as e: st.error(f"Entity processing error: {e}")
        except Exception as e: st.error(f"Sentiment analysis error: {e}")

# (Tab 3: Glossary - remains the same)
with tabs[2]:
    st.header("📘 Financial Glossary Lookup")
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        glossary_path = os.path.join(project_root, "glossary", "finance_terms.json")
        if not os.path.exists(glossary_path):
             alt_glossary_path = os.path.join(project_root, "data", "finance_terms.json")
             if os.path.exists(alt_glossary_path): glossary_path = alt_glossary_path
             else: raise FileNotFoundError("Glossary file 'finance_terms.json' not found.")
        with open(glossary_path, "r", encoding='utf-8') as f: glossary = json.load(f)
        if not analysis_text: st.warning("No content loaded.")
        else:
            st.subheader("📌 Terms Detected in Document")
            with st.spinner("Finding glossary terms..."): matches = gu.find_terms_in_text(glossary, analysis_text)
            if matches:
                cols = st.columns(2); match_items = list(matches.items())
                for i, (term, definition) in enumerate(match_items):
                    with cols[i % 2]: st.markdown(f"**{term}**: {definition}")
            else: st.info("No glossary terms matched document.")
        st.divider(); st.subheader("🔍 Search Glossary Manually")
        keyword = st.text_input("Search term or keyword", key="glossary_search")
        if keyword:
            results = gu.search_glossary(glossary, keyword)
            if results: [st.markdown(f"**{term}**: {definition}") for term, definition in results.items()]
            else: st.info(f"No results for '{keyword}'.")
    except FileNotFoundError as e: st.error(f"{e}")
    except Exception as e: st.error(f"Error loading/searching glossary: {e}")


# (Tab 4: Q&A - remains the same, uses corrected imports/variables)
with tabs[3]:
    st.header("💬 Ask a Question about the Document (Using Gemini)")
    if not analysis_text: st.warning("No content loaded.")
    else:
        question = st.text_input("Enter question", key="qa_question_gemini")
        if question:
            if not gemini_configured: st.error("Gemini API not configured (check GOOGLE_API_KEY).")
            else:
                 with st.spinner("Asking Gemini..."):
                    try:
                        answer = answer_question(analysis_text, question)
                        if answer.startswith("Error:"): st.error(answer)
                        else: st.success("**Answer:**"); st.markdown(answer)
                    except Exception as e: st.error(f"Unexpected Q&A error: {e}")

# (Tab 5: Stock Info & Charts - remains the same)
with tabs[4]:
    st.header("📈 Stock Information & Price Charts")
    ticker_to_use = st.session_state.get("ticker_symbol", "")
    if not ticker_to_use and analysis_text:
         company_name_auto, detected_ticker_auto = extract_company_name(analysis_text)
         if not st.session_state.get("company_name"): st.session_state["company_name"] = company_name_auto
         if detected_ticker_auto:
             ticker_to_use = detected_ticker_auto.strip().upper()
             st.session_state["ticker_symbol"] = ticker_to_use; st.rerun()
         elif company_name_auto:
             with st.spinner(f"Mapping '{company_name_auto}'..."): mapped_ticker = map_to_ticker(company_name_auto)
             if mapped_ticker: ticker_to_use = mapped_ticker.strip().upper(); st.session_state["ticker_symbol"] = ticker_to_use; st.rerun()
    display_name = st.session_state.get("company_name") or ticker_to_use or "N/A"
    st.write(f"Selected Company/Ticker: **{display_name}**")
    if not ticker_to_use: st.warning("No stock ticker identified or provided.")
    else:
        with st.spinner(f"Fetching stock data for {ticker_to_use}..."): stock_info = get_stock_info(ticker_to_use)
        if stock_info and "error" not in stock_info:
            st.subheader(f"Key Data for {stock_info.get('company', ticker_to_use)}")
            col1, col2, col3 = st.columns(3)
            def fmt_stock(v, is_curr=True, is_pct=False, dec=2):
                if isinstance(v,(int,float)):
                    if is_pct: return f"{v*100:.{dec}f}%"
                    if is_curr: return f"${v:,.{dec}f}"
                    return f"{v:,.{dec}f}"
                return "N/A"
            with col1: st.metric("Price", fmt_stock(stock_info.get('current_price'))); st.metric("Mkt Cap", fmt_stock(stock_info.get('market_cap'), dec=0)); st.metric("P/E", fmt_stock(stock_info.get('pe_ratio'), False))
            with col2: st.metric("52W High", fmt_stock(stock_info.get('fifty_two_week_high'))); st.metric("52W Low", fmt_stock(stock_info.get('fifty_two_week_low'))); st.metric("Beta", fmt_stock(stock_info.get('beta'), False))
            with col3: st.metric("Div Yield", fmt_stock(stock_info.get('dividend_yield'), False, True)); st.metric("Sector", stock_info.get('sector', 'N/A')); st.metric("Industry", stock_info.get('industry', 'N/A'))
            st.divider(); st.subheader("📊 Stock Price Chart")
            periods = {"1D":"1d", "5D":"5d", "1M":"1mo", "6M":"6mo", "YTD":"ytd", "1Y":"1y", "5Y":"5y", "Max":"max"}
            sel_lbl = st.selectbox("Period:", list(periods.keys()), index=5, key="chart_period")
            sel_val = periods[sel_lbl]; interval="1d"
            if sel_val=="1d": interval="2m"
            elif sel_val=="5d": interval="15m"
            elif sel_val=="1mo": interval="60m"
            with st.spinner(f"Loading {sel_lbl} chart..."): hist_df=pmu.get_historical_data(ticker_to_use,sel_val,interval)
            if not hist_df.empty:
                cols_ohlc = ['Open','High','Low','Close']; can_plot_ohlc = all(c in hist_df for c in cols_ohlc) and not hist_df[cols_ohlc].isnull().all().all()
                fig = None; chart_type = "N/A"
                if can_plot_ohlc:
                    fig=go.Figure(data=[go.Candlestick(open=hist_df['Open'],high=hist_df['High'],low=hist_df['Low'],close=hist_df['Close'],x=hist_df.index,name=ticker_to_use)]); chart_type="Candlestick"
                elif 'Close' in hist_df and not hist_df['Close'].isnull().all():
                    fig=go.Figure(data=[go.Scatter(x=hist_df.index,y=hist_df['Close'],mode='lines',name='Close')]); chart_type="Line (Close)"
                else: st.warning("Required price data missing.")
                if fig: fig.update_layout(title=f"{stock_info.get('company',ticker_to_use)} ({ticker_to_use}) - {sel_lbl} ({chart_type})", xaxis_title='Date/Time', yaxis_title='Price (USD)', xaxis_rangeslider_visible=False); st.plotly_chart(fig, use_container_width=True)
            else: st.warning(f"No historical data for {ticker_to_use} ({sel_lbl}/{interval}).")
        else: st.error(f"Could not retrieve stock info for {ticker_to_use}. Reason: {stock_info.get('error', 'Unknown')}")


# (Tab 6: Predictive Analytics - remains the same)
with tabs[5]:
    st.header("🔮 Predictive Analytics (Experimental)")
    ticker = st.session_state.get("ticker_symbol", "")
    if not ticker: st.warning("No stock ticker identified.")
    else:
        st.info(f"Predictions for **{ticker}**. Simple models, demo only, not financial advice.")
        pred_types = ["Intraday (Next Hour)", "Short-Term (Next 7 Days)", "Long-Term (Next 30 Days)"]
        sel_pred = st.selectbox("Prediction Horizon:", pred_types, key="pred_horizon")
        key_base = f"pred_{sel_pred.replace(' ','_')}"
        pred_res, pred_conf, pred_expl = (st.session_state.get(f"{key_base}_res", None),
                                          st.session_state.get(f"{key_base}_conf", 0.0),
                                          st.session_state.get(f"{key_base}_expl", "Not run yet."))
        if st.button(f"Run {sel_pred} Prediction", key=f"run_{key_base}"):
            with st.spinner(f"Calculating..."):
                res, conf, expl = None, 0.0, "Calc failed."
                try:
                    if sel_pred == pred_types[0]: res, conf, expl = pmu.predict_intraday(ticker)
                    elif sel_pred == pred_types[1]: res, conf, expl = pmu.predict_short_term(ticker, 7)
                    elif sel_pred == pred_types[2]: res, conf, expl = pmu.predict_long_term(ticker, 30)
                    st.session_state[f"{key_base}_res"], st.session_state[f"{key_base}_conf"], st.session_state[f"{key_base}_expl"] = res, conf, expl
                    st.rerun()
                except Exception as e: st.error(f"Prediction error: {e}"); st.session_state[f"{key_base}_expl"] = f"Error: {e}"
        st.subheader(f"Result: {sel_pred}")
        if pred_res is not None:
             c1,c2=st.columns(2); c1.metric("Predicted Price",f"${pred_res:.2f}"); c2.metric("Confidence",f"{pred_conf:.1%}")
             st.caption(f"Explanation: {pred_expl}")
        elif pred_expl != "Not run yet.": st.error(f"Failed: {pred_expl}")
        else: st.info("Click button to run.")
        st.divider(); st.subheader("Historical Context (1Y)")
        hist_ctx = pmu.get_historical_data(ticker, "1y", "1d")
        if not hist_ctx.empty and 'Close' in hist_ctx and not hist_ctx['Close'].isnull().all():
            fig_ctx=go.Figure(data=[go.Scatter(x=hist_ctx.index,y=hist_ctx['Close'],mode='lines',name='Close')])
            fig_ctx.update_layout(title=f'{ticker} - Past 1Y',xaxis_rangeslider_visible=False); st.plotly_chart(fig_ctx,use_container_width=True)
        else: st.warning("Could not load 1Y history.")

# (Tab 7: Summary - includes financial mentions check)
with tabs[6]:
    st.header("📝 Document & Call Summary")
    if not analysis_text: st.warning("No content loaded.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
             st.subheader("🔑 Key Insights (Sentiment)")
             try:
                 pol, subj = su.get_basic_sentiment(analysis_text)
                 fwd_s, conf_s, unc_s = su.get_forward_looking_sentiment(analysis_text), su.get_confidence_score(analysis_text), su.get_uncertainty_score(analysis_text)
                 def s_lvl(s,p=0.05,n=-0.05): return "Positive" if isinstance(s,(int,float)) and s>p else ("Negative" if isinstance(s,(int,float)) and s<n else "Neutral")
                 def fmt_s(v,p=2): return f"{v:.{p}f}" if isinstance(v,(int,float)) else "N/A"
                 st.markdown(f"- Sentiment: {s_lvl(pol)} (`{fmt_s(pol)}`)")
                 st.markdown(f"- Subjectivity: {'High' if isinstance(subj,float) and subj>0.6 else 'Medium' if isinstance(subj,float) and subj>0.4 else 'Low'} (`{fmt_s(subj)}`)")
                 st.markdown(f"- Fwd-Looking: {s_lvl(fwd_s,0.1,-0.1)} (`{fmt_s(fwd_s)}`)")
                 st.markdown(f"- Confidence Indicators: {conf_s}")
                 st.markdown(f"- Uncertainty Indicators: {unc_s}")
             except Exception as e: st.error(f"Sentiment insight error: {e}")

             st.subheader("💰 Financial Mentions (Extracted)")
             try:
                 # Calling the function from financials_utils
                 financials = extract_financials(analysis_text)

                 def format_financial(value): # Helper to format numbers or return N/A
                     if value is None or value == '': return "N/A"
                     try:
                         num_val = float(str(value).replace(',',''))
                         if abs(num_val) >= 1e9: return f"${num_val/1e9:.2f} B"
                         if abs(num_val) >= 1e6: return f"${num_val/1e6:.2f} M"
                         if abs(num_val) >= 1e3: return f"${num_val/1e3:,.0f} K"
                         return f"${num_val:,.2f}"
                     except (ValueError, TypeError): return str(value)[:50] # Return as string if not convertible

                 # Display results, using .get() which returns None if key is missing
                 rev = financials.get('revenue')
                 inc = financials.get('net_income')
                 gui = financials.get('guidance')
                 st.markdown(f"- **Revenue Mentioned:** {format_financial(rev)}")
                 st.markdown(f"- **Net Income Mentioned:** {format_financial(inc)}")
                 st.markdown(f"- **Guidance Mentioned:** {format_financial(gui)}")

                 # Add a message if nothing was found
                 if rev is None and inc is None and gui is None:
                      st.caption("_(No specific financial figures for Revenue, Net Income, or Guidance were matched by the extraction patterns.)_")

             except Exception as e: st.error(f"Error extracting financial mentions: {e}")

        with col2:
            st.subheader("📄 Automated Summary (AI)")
            if st.button("Generate AI Summary", key="gen_summary_button"):
                 if not gemini_configured: st.error("Gemini API not configured.")
                 else:
                      with st.spinner("Generating summary..."):
                           try:
                                summary = pdf_utils.generate_summary(analysis_text, max_length=250, min_length=50)
                                st.session_state['ai_summary'] = summary; st.rerun()
                           except Exception as e: st.error(f"Summary generation error: {e}")
            if st.session_state.get('ai_summary'): st.markdown(st.session_state['ai_summary'])
            else: st.caption("Click button to generate AI summary.")

        st.divider(); st.subheader("🔗 Related Stock Info")
        ticker_summary = st.session_state.get("ticker_symbol")
        if ticker_summary:
             stock_info_summary = get_stock_info(ticker_summary) # Cached
             if stock_info_summary and "error" not in stock_info_summary:
                  def fmt_stock_s(v, is_c=True, is_p=False, d=2):
                      if isinstance(v,(int,float)): return f"{v*100:.{d}f}%" if is_p else (f"${v:,.{d}f}" if is_c else f"{v:,.{d}f}")
                      return "N/A"
                  st.write(f"**{stock_info_summary.get('company', ticker_summary)} ({ticker_summary})**")
                  st.write(f" - Price: {fmt_stock_s(stock_info_summary.get('current_price'))}")
                  st.write(f" - Mkt Cap: {fmt_stock_s(stock_info_summary.get('market_cap'), d=0)}")
                  st.write(f" - P/E Ratio: {fmt_stock_s(stock_info_summary.get('pe_ratio'), False)}")
             else: st.warning(f"Could not retrieve stock info for {ticker_summary}.")
        else: st.info("No ticker identified.")