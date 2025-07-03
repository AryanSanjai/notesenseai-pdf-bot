# NotesenseAI Q&A BOT

An AI-powered PDF Question Answering app using Retrieval-Augmented Generation (RAG) and IBM watsonx.ai.

## Features

- Upload any PDF and ask natural language questions.
- Retrieval-Augmented Generation (RAG) for precise, context-aware answers.
- Summary highlighting for summary requests.
- Displays relevant PDF sections used for answers.
- Choose between IBM watsonx.ai foundation models.
- Clean, card-based UI built with Streamlit.

## Setup

1. Clone the repository:

2. (Recommended) Create and activate a virtual environment:
python -m venv .venv
.venv\Scripts\activate # Windows PowerShell

3. Install dependencies:
pip install -r requirements.txt
Or directly:
pip install streamlit pymupdf requests python-dotenv sentence-transformers faiss-cpu

4. Create a `.env` file in the project root with your IBM credentials:
IBM_API_KEY=your_actual_api_key_here
PROJECT_ID=your_actual_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com

5. Run the app:
streamlit run app.py

6. Open [http://localhost:8501](http://localhost:8501) in your browser if it doesn’t open automatically.

## Usage

- Upload a PDF file.
- Type your question or ask for a summary.
- View the AI-generated answer and the relevant PDF sections.
- Switch models from the sidebar to experiment.

## License

MIT License

---

Built with IBM watsonx.ai, Streamlit, PyMuPDF, sentence-transformers, and FAISS.
