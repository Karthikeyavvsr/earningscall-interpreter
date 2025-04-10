import shutil
import importlib.util
import subprocess

REQUIRED_LIBRARIES = [
    "whisper",
    "ffmpeg",
    "textblob",
    "spacy",
    "yfinance",
    "streamlit",
    "nltk",
    "sklearn",
    "numpy",
    "pandas"
]


def check_dependencies():
    issues = []

    # --------- System dependency: ffmpeg --------- #
    if shutil.which("ffmpeg") is None:
        issues.append("❌ FFmpeg not found. Please install it using `brew install ffmpeg` or `apt install ffmpeg`.")

    # --------- Python packages --------- #
    for lib in REQUIRED_LIBRARIES:
        if not importlib.util.find_spec(lib):
            issues.append(f"❌ {lib} not installed. Run: `pip install {lib}`")

    # --------- spaCy model --------- #
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except:
        issues.append("❌ spaCy model 'en_core_web_sm' not found. Run: `python -m spacy download en_core_web_sm`.")

    # --------- TextBlob corpora --------- #
    try:
        import textblob
        subprocess.run(["python", "-m", "textblob.download_corpora"], check=True)
    except subprocess.CalledProcessError:
        issues.append("❌ TextBlob corpora not available. Run: `python -m textblob.download_corpora`.")

    return issues


if __name__ == "__main__":
    results = check_dependencies()
    if results:
        print("\n\n".join(results))
    else:
        print("✅ All dependencies are installed and configured correctly.")