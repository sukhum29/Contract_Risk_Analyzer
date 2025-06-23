import streamlit as st
import fitz  # PyMuPDF
import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from time import sleep
from nltk.tokenize.punkt import PunktSentenceTokenizer

# Ensure 'punkt' is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ----- Constants ----- #
RISK_PATTERNS = {
    "Termination at Will": r"\bterminat(e|ion) at (will|any time)\b",
    "Vague Liability": r"\blimitation of liability\b.*\b(vague|unclear)\b",
    "Subject to Change": r'\bsubject to change\b',
    "Sole Discretion": r'\bat the sole discretion\b',
    "Dispute Event": r'\bin the event of dispute\b',
    "Written Request": r'\bupon written request\b',
    "May Terminate": r'\bmay terminate\b',
}

RISK_WEIGHTS = {
    "Termination at Will": 0.90,
    "Vague Liability": 0.95,
    "Subject to Change": 0.90,
    "Sole Discretion": 0.85,
    "Dispute Event": 0.80,
    "Written Request": 0.55,
    "May Terminate": 0.75,
}

RISK_LABELS = ["High Risk", "Medium Risk", "Low Risk", "No Risk"]
BART_RISK_SCORES = {
    "High Risk": 1.0,
    "Medium Risk": 0.6,
    "Low Risk": 0.3,
    "No Risk": 0.0,
}

# ----- Functions ----- #
def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def detect_risks(text):
    risks = []
    for risk_type, pattern in RISK_PATTERNS.items():
        matches = re.finditer(pattern, text, flags=re.IGNORECASE)
        for match in matches:
            risks.append({
                "type": risk_type,
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            })
    return risks

def calculate_score(risks):
    total = sum(RISK_WEIGHTS.get(risk["type"], 0) for risk in risks)
    return min(total * 10, 100)

def calculate_bart_score(df):
    if df.empty:
        return 0.0
    numeric_scores = df["Top Label"].map(BART_RISK_SCORES).fillna(0)
    return round(numeric_scores.mean() * 100, 2)

def determine_risk_level(score):
    if score >= 75:
        return "High Risk"
    elif score >= 40:
        return "Medium Risk"
    elif score > 0:
        return "Low Risk"
    else:
        return "No Risk"

# ----- Load Models ----- #
st.cache_resource(show_spinner=False)
def load_models():
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return zero_shot_classifier

# ----- Streamlit App ----- #
st.title("Legal Clause Risk Analyzer")

uploaded_file = st.file_uploader("Upload a legal PDF document", type="pdf")

if uploaded_file:
    st.info("Extracting text from the uploaded PDF document...")
    raw_text = extract_text(uploaded_file)
    tokenizer = PunktSentenceTokenizer()
    clauses = tokenizer.tokenize(raw_text)
    clauses = [c.strip() for c in clauses if c.strip() and not c.strip().isdigit() and not re.fullmatch(r'\d+\.?', c.strip())]

    st.success(f"Successfully extracted {len(clauses)} clauses from the document.")

    st.info("Running multi-model risk analysis on the extracted clauses...")
    zero_shot = load_models()

    combined_risks = []
    total_risks = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, clause in enumerate(clauses):
        rule_matches = detect_risks(clause)
        zero_shot_result = zero_shot(clause, candidate_labels=RISK_LABELS)

        clause_data = {
            "Clause": clause[:300],
            "Rule-Based": ", ".join(match["type"] for match in rule_matches),
        }

        for label, score in zip(zero_shot_result["labels"], zero_shot_result["scores"]):
            clause_data[label] = round(score, 3)

        clause_data["Top Label"] = zero_shot_result["labels"][0]
        clause_data["Top Confidence"] = round(zero_shot_result["scores"][0], 3)
        clause_data["Rule Risk Score"] = calculate_score(rule_matches)

        total_risks.extend(rule_matches)
        combined_risks.append(clause_data)

        progress_bar.progress((i + 1) / len(clauses))
        status_text.text(f"Processing clause {i + 1} of {len(clauses)}")
        sleep(0.01)

    df = pd.DataFrame(combined_risks)

    st.subheader("Risk Assessment Table")
    st.dataframe(df)

    st.subheader("Distribution of Predicted Risk Levels")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Top Label", order=df["Top Label"].value_counts().index, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.download_button("Download Results as CSV", data=df.to_csv(index=False), file_name="risk_analysis.csv", mime="text/csv")
    # ----- Show Risk Scores ----- #
    bart_score = calculate_bart_score(df)
    final_score = round( (0.6 * bart_score), 2)
    final_label = determine_risk_level(final_score)

    st.subheader("Overall Risk Scoring Summary")
    st.metric(label="Final Combined Risk Score (0–100)", value=f"{final_score:.2f}")
    st.info(f"Final Risk Classification: {final_label}")
