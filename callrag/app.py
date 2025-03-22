import os
import time
import re
import fitz  # PyMuPDF for reading PDFs
import streamlit as st
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.google import GeminiEmbedding


#  Set Browser Tab Title & Icon
st.set_page_config(page_title="CallRag", page_icon="😎")

#  Configure Gemini Embeddings
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

#  Load API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])
genai.configure(api_key=GEMINI_API_KEY)

#  PDF Text Extraction Function


def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = f"Document: {os.path.basename(pdf_path)}\n\n"
    text += "\n".join(page.get_text("text") for page in doc)
    return text

# Load PDFs & Create Index


@st.cache_resource(show_spinner=False)
def load_data():
    """Loads and indexes PDF documents for retrieval."""
    # data_dir = "./data"  # Directory containing PDF files
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    docs = []

    if not os.path.exists(data_dir):
        st.warning(
            f"Data directory `{data_dir}` not found. Please upload PDFs.")
        return None

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        st.warning("No PDF files found in the data directory.")
        return None

    with st.spinner("Loading and indexing documents..."):
        for filename in pdf_files:
            pdf_path = os.path.join(data_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            docs.append(Document(text=text))

        return VectorStoreIndex.from_documents(docs)


index = load_data()

#  Gemini Response Generator with Streaming


def generate_gemini_response(prompt):
    """Generates response using Gemini AI and streams it sentence by sentence."""
    model = genai.GenerativeModel(
        "gemini-1.5-flash-latest")  # Optimized for speed
    response = model.generate_content(prompt)

    response_text = response.text if response and response.text else "Sorry, I couldn't generate a response."
    # Splitting while preserving punctuation
    sentences = re.split(r'(?<=[.!?]) +', response_text)

    for sentence in sentences:
        yield sentence.strip()


#  Streamlit UI
st.title("📖 RAG-Based Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask questions!"}]

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
if user_input := st.chat_input("Type a message"):
    #  Show User Input Immediately
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    #  Generate Response Immediately
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for sentence in generate_gemini_response(user_input):
            full_response += sentence + " "
            response_placeholder.write(full_response)
            time.sleep(0.5)  # Simulated streaming effect

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
