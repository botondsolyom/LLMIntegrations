# LLMIntegrations
Smart Librarian – AI with RAG + Tool

This is an AI chatbot that recommends books based on user interests.
It uses the OpenAI GPT + RAG (ChromaDB) for semantic retrieval, and a custom tool to provide a full detailed summary of the chosen book.

1.) Features

Database of book summaries (data/book_summaries.md, 12+ titles).
Vector store: ChromaDB (not OpenAI’s vector store).
Retriever: semantic search using OpenAI embeddings.
Chatbot:
CLI (app_cli.py)
Streamlit UI (streamlit_app.py)
Tool: get_summary_by_title(title) loads the full description from data/full_summaries.json.
Automatic function calling: GPT recommends a book, then the tool returns the detailed summary.
2.) The Project Structure

smart-librarian/ │ ├─ data/ │ ├─ book_summaries.md # short thematic summaries (10+ books) │ └─ full_summaries.json # detailed summaries (tool uses this source) │ ├─ smart_librarian/ │ ├─ init.py │ ├─ rag.py # retriever logic (semantic search with ChromaDB) │ ├─ tools.py # tool: get_summary_by_title + OpenAI tool schema │ ├─ moderation.py # simple offensive language filter │ ├─ tts.py # Text-to-Speech helper (OpenAI TTS → mp3 output) │ ├─ stt.py # Speech-to-Text helper (Whisper-1 transcription) │ └─ images.py # book illustration generation (OpenAI image API) │ ├─ scripts/ │ └─ ingest.py # loads book summaries into ChromaDB with embeddings │ ├─ chroma/ # (GENERATED) persistent ChromaDB database │ └─ ... # created after running ingest.py │ ├─ out/ │ └─ audio/ # (GENERATED) audio files from TTS │ └─ last_recommendation.mp3 │ ├─ app_cli.py # CLI chatbot (RAG + tool + optional TTS) ├─ streamlit_app.py # Streamlit UI (RAG + tool + moderation + TTS + STT + images) │ ├─ requirements.txt # dependencies (Streamlit, ChromaDB, OpenAI SDK, etc.) ├─ .env # environment variables (e.g. OPENAI_API_KEY, model names) ├─ .env.example # example env file template └─ README.md # project documentation and setup guide

3.) Setup

Clone the project
git clone cd smart-librarian

Create environment (venv)
python -m venv .venv

Activate:

Windows: .\.venv\Scripts\activate
Linux/Mac: source .venv/bin/activate
Install dependencies
pip install -r requirements.txt

Key packages:

chromadb==0.5.5
hnswlib==0.8.0
openai==1.30.1
python-dotenv==1.0.1
tqdm==4.66.4
streamlit==1.36.0
Configure .env Create .env in the project root:
OPENAI_API_KEY=sk-... EMBED_MODEL=text-embedding-3-small CHAT_MODEL=gpt-4o CHROMA_DIR=./chroma CHROMA_TELEMETRY_ENABLED=false

4.) Data Preparation

Edit book summaries
data/book_summaries.md → short (3–5 lines) thematic summaries
data/full_summaries.json → detailed summaries
Ingest into ChromaDB Run in PyCharm or terminal:
python scripts/ingest.py

Output should show:

✅ Uploaded 12 documents into Chroma at ./chroma

5.) Usage

CLI chatbot
python app_cli.py

Example session:

Smart Librarian (CLI). Type 'exit' to quit. I want a book about freedom and social control. === Smart Librarian === "1984" by George Orwell is a gripping exploration...

Streamlit UI Run with PyCharm config or terminal:
streamlit run streamlit_app.py

Open browser: [http://localhost:8501]

Example Queries

I want a book about friendship and magic
What do you recommend if I love war stories?
I want a book about freedom and social control
What is 1984?
The expected output:

Recommendation

"1984" by George Orwell delves into the life of Winston Smith, who navigates a world where thought is policed, history is rewritten, and citizens are constantly monitored. His journey of resistance through seeking truth, maintaining a diary, and falling in love highlights the struggle against a system designed to crush individuality. This novel powerfully examines how language, fear, and surveillance can distort reality and undermine freedom. Enjoy the read!
