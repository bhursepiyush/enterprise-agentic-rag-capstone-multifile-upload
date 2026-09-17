# Capstone Project: Autonomous Enterprise Document QA Agent

## 1. System Architecture
- **User Interface**: Streamlit web dashboard allowing multi-format file uploads (PDF, TXT, CSV, Excel) and chat interface.
- **Backend API**: FastAPI service exposing `/upload` and `/query` endpoints.
- **Data Ingestion**: Custom chunking pipeline (500 tokens, 50 token overlap) with `all-MiniLM-L6-v2` dense embeddings.
- **Vector Database**: ChromaDB vector store for persistent semantic retrieval.
- **Agentic RAG Engine**: Multi-stage reasoning pipeline powered by local Llama 3 via Ollama.

## 2. Agent Roles
1. **Planner Agent**: Analyzes user intent, checks if retrieval is required, and reformulates query keywords.
2. **Retrieval Agent**: Queries the vector store to fetch top relevant chunks.
3. **Response / Guardrail Agent**: Synthesizes grounded answers strictly using context; emits fallback statements when data is absent to eliminate hallucinations.

## 3. Deployment & Execution
1. Start Ollama: `ollama pull llama3 && ollama serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Run backend: `uvicorn app.main:app --port 8000 --reload`
4. Run frontend: `streamlit run frontend/app.py`

## 4. Limitations & Challenges Faced
- **OCR Limitations**: Basic PDF extractors cannot read scanned/image-only PDFs. Production deployments require an OCR layer (e.g., Tesseract).
- **Inference Latency**: Running local LLMs on standard CPUs introduces response delays compared to cloud GPUs.