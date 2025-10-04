import streamlit as st
import PyPDF2
import re
from sentence_transformers import SentenceTransformer, util

# --- Load embedding model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --- Helper Functions ---
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF."""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def preprocess(text):
    """Clean and split text into paragraphs for better context."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    paragraphs = text.split(". ")  # split by sentence-ending periods
    return [p.strip() for p in paragraphs if p.strip()]

def answer_question(question, sentences, top_n=3):
    """Return top N most semantically relevant sentences for the question."""
    # Encode all sentences and question
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]

    # Get top N indices
    top_indices = cosine_scores.argsort(descending=True)[:top_n]
    top_answers = [sentences[i] for i in top_indices]
    return top_answers

# --- Streamlit UI ---
st.set_page_config(page_title="📖 AskItAll", layout="wide", page_icon="📚")
st.title("📖 AskItAll – Instant Answers from Text & PDFs")
st.markdown("Ask questions from your text or PDF book easily!")

# Store sentences in session state
if "sentences" not in st.session_state:
    st.session_state.sentences = None

# --- Sidebar ---
st.sidebar.header("📌 Input Options")
st.sidebar.info(
    """
    1. Paste your text or upload a PDF.
    2. Click 'Process' button below the input.
    3. Once processed, type your question and click 'Get Answers'.
    """
)
option = st.sidebar.radio("Choose Input Type:", ["Paste Text", "Upload PDF"])

# --- Input Section with Buttons ---
if option == "Paste Text":
    text_input = st.text_area("Paste or type your text here:")
    if st.button("Process Text"):
        if text_input.strip() == "":
            st.warning("❗ Please enter some text first.")
        else:
            st.session_state.sentences = preprocess(text_input)
            st.success("✅ Text processed! You can now ask questions.")

elif option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                st.session_state.sentences = preprocess(text)
            st.success("✅ PDF uploaded and processed! You can now ask questions.")

# --- Q&A Section ---
if st.session_state.sentences:
    question = st.text_input("Ask your question here:")

    if st.button("Get Answers"):
        if question.strip() == "":
            st.warning("❗ Please enter a question first.")
        else:
            with st.spinner("Searching for answers..."):
                answers = answer_question(question, st.session_state.sentences, top_n=3)

            st.subheader("🔍 Top Answers:")
            for i, ans in enumerate(answers, 1):
                st.markdown(
                    f"""
                    <div style='
                        background-color:#f0f2f6;
                        color: #000000;
                        padding:15px;
                        border-radius:10px;
                        margin-bottom:10px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    '>
                    <b>{i}.</b> {ans}
                    </div>
                    """, unsafe_allow_html=True
                )
