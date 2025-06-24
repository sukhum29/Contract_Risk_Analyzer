# Legal GPT
import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
import tempfile

# Title and description
st.set_page_config(page_title="Contract QA", layout="wide")
st.title("Contract Question Answering App")
st.markdown("""
This application lets you upload a contract in PDF format and ask questions about its contents.
""")

# Load QA pipeline
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="deepset/deberta-v3-base-squad2")

qa_pipeline = load_model()

# PDF Text Extractor
def extract_text_from_pdf(pdf_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    reader = PdfReader(tmp_file_path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# File uploader
st.subheader("Upload a Contract PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Main logic
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        context = extract_text_from_pdf(uploaded_file)
    st.success("Text extracted from the contract.")

    st.divider()

    st.subheader("Full Extracted Contract Text")
    st.text_area("Contract Text", context, height=400)

    st.divider()

    st.subheader("Ask a Question About the Contract")
    question = st.text_input("Enter your question below:")

    if question:
        with st.spinner("Finding the answer using DeBERTa..."):
            try:
                result = qa_pipeline({
                    'question': question,
                    'context': context
                })
                st.markdown("#### Answer")
                st.write(result['answer'])
            except Exception as e:
                st.error(f"Error: {str(e)}")