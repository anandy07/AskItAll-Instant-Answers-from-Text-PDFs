import streamlit as st
import re
import PyPDF2
from sentence_transformers import SentenceTransformer, util

# ================================
# Load Embedding Model (Cached)
# ================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ================================
# Extract Text From PDF
# ================================
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except:
                page_text = ""
            text += page_text + " "
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# ================================
# Clean & Split Text Into Sentences
# ================================
def preprocess(text):
    text = re.sub(r"\s+", " ", text)  # normalize spaces

    # Split into sentences using punctuation marks
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Remove extremely short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    return sentences

# ================================
# Answer Questions Using Semantic Search
# ================================
def answer_question(question, sentences, sentence_embeddings, top_n=3):
    question_embedding = model.encode(question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]

    top_indices = cosine_scores.argsort(descending=True)[:top_n]

    results = []
    for i in top_indices:
        results.append({
            "sentence": sentences[i],
            "score": float(cosine_scores[i])
        })

    return results

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="📖 AskItAll", layout="wide", page_icon="📚")
st.title("📖 AskItAll – Instant Answers from Text & PDFs")
st.markdown("Ask questions from your text or PDF and get instant AI-powered answers!")

# Initialize memory
if "sentences" not in st.session_state:
    st.session_state.sentences = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# Sidebar
st.sidebar.header("📌 Input Options")
option = st.sidebar.radio("Choose Input Type:", ["Paste Text", "Upload PDF"])

st.sidebar.info(
    """
    ✔ Paste text OR upload a PDF  
    ✔ Click **Process**  
    ✔ Ask questions  
    ✔ Get AI-powered answers instantly  
    """
)

# ================================
# Input Section
# ================================
if option == "Paste Text":
    text_input = st.text_area("Paste or type your text here:", height=200)

    if st.button("Process Text"):
        if text_input.strip() == "":
            st.warning("❗ Please enter some text first.")
        else:
            sentences = preprocess(text_input)
            embeddings = model.encode(sentences, convert_to_tensor=True)

            st.session_state.sentences = sentences
            st.session_state.embeddings = embeddings

            st.success("✅ Text processed! You can now ask questions.")

elif option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            sentences = preprocess(text)
            embeddings = model.encode(sentences, convert_to_tensor=True)

            st.session_state.sentences = sentences
            st.session_state.embeddings = embeddings

        st.success("✅ PDF uploaded & processed! You can now ask questions.")

# ================================
# Question Answering Section
# ================================
if st.session_state.sentences:
    question = st.text_input("Ask your question here:")

    if st.button("Get Answers"):
        if question.strip() == "":
            st.warning("❗ Please enter a question first.")
        else:
            with st.spinner("Searching for answers..."):
                results = answer_question(
                    question,
                    st.session_state.sentences,
                    st.session_state.embeddings,
                    top_n=3
                )

            st.subheader("🔍 Top Answers:")
            for i, r in enumerate(results, 1):
                st.markdown(
                    f"""
                    <div style='
                        background-color:#f0f2f6;
                        color:#000;
                        padding:15px;
                        border-radius:10px;
                        margin-bottom:10px;
                        box-shadow:2px 2px 5px rgba(0,0,0,0.15);
                    '>
                        <b>{i}. (Score: {r["score"]:.4f})</b><br>
                        {r["sentence"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
