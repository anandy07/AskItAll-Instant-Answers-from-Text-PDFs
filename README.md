# AskItAll 📖

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An intelligent Q&A application that provides instant, semantically-correct answers from your text and PDF documents.

**AskItAll** allows you to bypass the need for manual reading and searching. Simply upload a document, ask a question, and get the most relevant information delivered to you instantly.

---

## ✨ Demo

![AskItAll Demo GIF](https://your-link-to-a-demo-gif.com/demo.gif)
*(**Note**: You should record a short GIF of your app working and replace the link above.)*

---

## 🧐 About The Project

This tool was built to solve a common problem: finding specific information buried within large text documents or PDFs. Instead of spending hours reading, you can use AskItAll to query your documents directly and get the most relevant sentences that answer your question. It uses a powerful sentence-transformer model to understand the context and meaning behind your questions, ensuring highly accurate results.

### Key Features:

-   **Dual Input Modes**: Supports both raw text input and PDF file uploads.
-   **Semantic Search**: Leverages the `all-MiniLM-L6-v2` sentence-transformer model to understand the contextual meaning of your questions, not just keywords.
-   **Top Answer Highlighting**: Returns the top 3 most relevant sentences from the document that answer your question.
-   **Interactive UI**: A clean, simple, and user-friendly interface built with Streamlit.

---

## 🛠️ Built With

-   [Streamlit](https://streamlit.io/) - For the web application framework.
-   [SentenceTransformers](https://sbert.net/) - For generating text embeddings.
-   [PyPDF2](https://pypdf2.readthedocs.io/) - For extracting text from PDF files.
-   [Python](https://www.python.org/)

---

## 🧠 How It Works

The application follows a simple yet powerful workflow to find answers:

1.  **Text Extraction**: The user either pastes text or uploads a PDF. For PDFs, the `PyPDF2` library extracts all text content.
2.  **Preprocessing**: The extracted text is cleaned and split into individual sentences or small paragraphs to create a searchable knowledge base.
3.  **Embedding Generation**: When a user asks a question, the `SentenceTransformer` model converts the question and all the preprocessed sentences into numerical vectors (embeddings).
4.  **Similarity Calculation**: The application calculates the cosine similarity between the question's vector and every sentence's vector to find the best matches.
5.  **Display Results**: The top 3 sentences with the highest similarity scores are presented to the user as the most likely answers.

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have Python 3.9+ and pip installed on your system.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your_username/AskItAll.git](https://github.com/your_username/AskItAll.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd AskItAll
    ```
3.  **Install the required packages:**
    *(**Note**: Create a `requirements.txt` file with all the libraries like `streamlit`, `sentence-transformers`, `PyPDF2`, `torch`, etc.)*
    ```sh
    pip install -r requirements.txt
    ```

---

## Usage

1.  Run the application from your terminal:
    ```sh
    streamlit run app.py
    ```
2.  Open your web browser and navigate to `http://localhost:8501`.
3.  In the sidebar, choose your input type ("Paste Text" or "Upload PDF").
4.  Provide your document and click the "Process" button.
5.  Once processed, type your question in the main text area and click "Get Answers".

---

## 📄 License

Distributed under the MIT License. See `LICENSE.txt` for more information.

---

## 👤 Author

**Anand Yadav**

-   **GitHub**: [@anandy07](https://github.com/anandy07)
-   **LinkedIn**: [Anand Yadav](https://www.linkedin.com/in/anand-yadav-04b75324a/)
