import streamlit as st
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# --- Load environment variables ---
load_dotenv()
IBM_API_KEY = os.getenv("IBM_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")

MODEL_OPTIONS = {
    "Granite 13B Instruct V2 (Deprecated)": "ibm/granite-13b-instruct-v2",
    "Granite 3.8B Instruct": "ibm/granite-3-8b-instruct"
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
        result = resp.json()['results']
        if isinstance(result, list):
            return result[0]['generated_text']
        elif isinstance(result, dict):
            return result['generated_text']
        else:
            return str(result)
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

def remove_repeated_sentences(text):
    sentences = text.split('. ')
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)
    result = '. '.join(unique_sentences).strip()
    if not result.endswith('.'):
        result += '.'
    return result

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6a/JavaScript-logo.png", width=80)
    st.title("NotesenseAI Q&A BOT")
    st.markdown("""
    <div style='font-size: 16px; color: #555;'>
    <b>Upload a PDF and ask questions about its content.<br>
    Powered by IBM watsonx.ai.</b>
    </div>
    """, unsafe_allow_html=True)
    st.header("Choose Model for Q&A and Summary")
    selected_model_name = st.radio(
        "Select a model (recommended: Granite 3.8B Instruct for future-proofing):",
        list(MODEL_OPTIONS.keys()),
        index=1  # Default to Granite 3.8B Instruct
    )
    selected_model_id = MODEL_OPTIONS[selected_model_name]
    st.markdown(f"""
    ### Models Used
    - Selected Model: `{selected_model_id}`
    """)
    st.header("Instructions")
    st.markdown("""
    1. Upload a PDF file<br>
    2. Type your question<br>
    3. Ask for PDF summary<br>
    4. View the answer and the relevant PDF section(s)<br>
    5. You can ask multiple questions after one upload!
    """, unsafe_allow_html=True)

# --- Main UI ---
st.markdown("<h1 style='color:#4F8BF9;'>NotesenseAI Q&A BOT</h1>", unsafe_allow_html=True)
st.write(f"**Current Model:** :blue[{selected_model_name}]")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and 'pdf_text' not in st.session_state:
    st.success("PDF uploaded!")
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    st.session_state['pdf_text'] = text

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = chunk_text(text, chunk_size=500)
    embeddings = embed_chunks(chunks, embedder)
    index = build_faiss_index(embeddings)
    st.session_state['chunks'] = chunks
    st.session_state['index'] = index
    st.session_state['embedder'] = embedder

if 'pdf_text' in st.session_state:
    with st.expander("Show Extracted Text Preview"):
        st.write(st.session_state['pdf_text'][:1000] + '...')
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
                if is_summary_request(question):
                    answer = remove_repeated_sentences(answer)
                    st.markdown(f"""
                    <div style='background-color:#ffe9b0; color:#222; padding:16px; border-radius:10px; margin-bottom:10px;'>
                        <b>Summary of the PDF:</b><br>
                        {answer}</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color:#eaf4ff; color:#222; padding:16px; border-radius:10px; margin-bottom:10px;'>
                        <b>Answer:</b><br>
                        {answer}</div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background-color:#f9f9f9; color:#222; padding:12px; border-radius:8px;'>
                    <b>Relevant PDF Section(s):</b><br>
                    {context}</div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload a PDF to get started.")
