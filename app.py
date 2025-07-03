import streamlit as st
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# --- Load environment variables ---
load_dotenv()
IBM_API_KEY = os.getenv("IBM_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")

MODEL_OPTIONS = {
    "Granite 3.2-8B Instruct (IBM)": "ibm/granite-3-8b-instruct",
    "Granite 13B Instruct V2 (IBM)": "ibm/granite-13b-instruct-v2"
}

summary_keywords = ['summary', 'summarize', 'brief', 'overview', 'main points', 'key points']

def is_summary_request(question):
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in summary_keywords)

def get_access_token(api_key):
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "apikey": api_key,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }
    resp = requests.post(url, headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def ask_ibm_watsonx(question, context, model_id, project_id, access_token):
    endpoint = f"{WATSONX_URL}/ml/v1/text/generation?version=2025-02-11"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    payload = {
        "model_id": model_id,
        "input": prompt,
        "parameters": {"max_new_tokens": 300},
        "project_id": project_id
    }
    resp = requests.post(endpoint, headers=headers, json=payload)
    if resp.status_code == 200:
        return resp.json()['results'][0]['generated_text']
    else:
        return f"Error {resp.status_code}: {resp.text}"

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def embed_chunks(chunks, embedder):
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    return np.array(embeddings).astype('float32')

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_similar_chunks(question, index, chunks, embedder, top_k=3):
    question_embedding = embedder.encode([question], convert_to_tensor=False)
    D, I = index.search(np.array(question_embedding).astype('float32'), top_k)
    relevant_chunks = [chunks[i] for i in I[0]]
    return "\n\n".join(relevant_chunks)

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6a/JavaScript-logo.png", width=80)  # Replace or remove
    st.title("NotesenseAI Q&A BOT")
    st.markdown("""
    <div style='font-size: 16px; color: #555;'>
    <b>Upload a PDF and ask questions about its content.<br>
    Powered by IBM watsonx.ai.</b>
    </div>
    """, unsafe_allow_html=True)
    st.header("Choose LLM Model")
    selected_model_name = st.selectbox("Select a model", list(MODEL_OPTIONS.keys()))
    selected_model_id = MODEL_OPTIONS[selected_model_name]
    st.markdown("---")
    st.subheader("Instructions")
    st.markdown("""
    1. Upload a PDF file<br>
    2. Type your question<br>
    3. Ask for PDF summary<br>
    4. View the answer and the relevant PDF section(s)
    """, unsafe_allow_html=True)

# --- Main UI ---
st.markdown("<h1 style='color:#4F8BF9;'>NotesenseAI Q&A BOT</h1>", unsafe_allow_html=True)
st.write(f"**Current Model:** :blue[{selected_model_name}]")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file:
    st.success("PDF uploaded!")

    # Extract text from PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    with st.expander("Show Extracted Text Preview"):
        st.write(text[:1000] + '...')

    # --- RAG Pipeline: Chunk, Embed, Build Index ---
    with st.spinner("Processing PDF and building knowledge base..."):
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        chunks = chunk_text(text, chunk_size=500)
        embeddings = embed_chunks(chunks, embedder)
        index = build_faiss_index(embeddings)
        st.session_state['chunks'] = chunks
        st.session_state['index'] = index
        st.session_state['embedder'] = embedder

    question = st.text_input("Ask a question about your PDF:")
    if question:
        with st.spinner("Searching and generating answer..."):
            try:
                context = search_similar_chunks(
                    question, st.session_state['index'], st.session_state['chunks'], st.session_state['embedder'], top_k=3
                )
                access_token = get_access_token(IBM_API_KEY)
                answer = ask_ibm_watsonx(
                    question, context, selected_model_id, PROJECT_ID, access_token
                )
                # Highlight summary requests
                if is_summary_request(question):
                    st.markdown("""
                    <div style='background-color:#ffe9b0; color:#222; padding:16px; border-radius:10px; margin-bottom:10px;'>
                        <b>Summary of the PDF:</b><br>
                        {}</div>
                    """.format(answer), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background-color:#eaf4ff; color:#222; padding:16px; border-radius:10px; margin-bottom:10px;'>
                        <b>Answer:</b><br>
                        {}</div>
                    """.format(answer), unsafe_allow_html=True)
                st.markdown("""
                <div style='background-color:#f9f9f9; color:#222; padding:12px; border-radius:8px;'>
                    <b>Relevant PDF Section(s):</b><br>
                    {}</div>
                """.format(context), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
