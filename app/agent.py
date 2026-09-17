import json
import requests
from sentence_transformers import SentenceTransformer
import chromadb

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./data/chroma")
collection = chroma_client.get_or_create_collection(name="documents")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

def query_vector_db(query: str, top_k: int = 3) -> list[str]:
    query_vector = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=top_k)
    if results and results.get("documents"):
        return results["documents"][0]
    return []

def call_llm(prompt: str) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return res.json().get("response", "Error generating response.")
    except Exception as e:
        return f"LLM connection error: {str(e)}"

def run_agentic_workflow(user_query: str) -> dict:
    # 1. Agent Planner: checks if search is needed & reformulates the prompt
    plan_prompt = f"""
    You are an AI Document Assistant. Determine if the user question requires document search.
    User Question: {user_query}
    Respond ONLY with a valid JSON object matching this structure:
    {{"requires_search": true, "rewritten_query": "better search query"}}
    """
    plan_response = call_llm(plan_prompt)
    try:
        plan_data = json.loads(plan_response[plan_response.find("{"):plan_response.rfind("}")+1])
    except Exception:
        plan_data = {"requires_search": True, "rewritten_query": user_query}

    # 2. Retrieval Agent
    retrieved_chunks = []
    if plan_data.get("requires_search", True):
        retrieved_chunks = query_vector_db(plan_data.get("rewritten_query", user_query))

    context_str = "\n---\n".join(retrieved_chunks) if retrieved_chunks else "No relevant enterprise context."

    # 3. Grounded Response & Guardrail Agent (Task 9)
    rag_prompt = f"""
    You are an enterprise AI agent. Answer the user question strictly using the provided context below.
    If the context does not contain enough information, state: "The requested information is not available in the uploaded documents."
    Do NOT hallucinate or extrapolate facts.

    Context:
    {context_str}

    Question: {user_query}
    Answer:
    """
    answer = call_llm(rag_prompt)
    return {
        "query": user_query,
        "plan": plan_data,
        "retrieved_context": retrieved_chunks,
        "answer": answer
    }