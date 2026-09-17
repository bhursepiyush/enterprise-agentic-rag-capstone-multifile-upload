import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.ingest import index_document
from app.agent import run_agentic_workflow

app = FastAPI(title="Capstone Document QA Agent")
os.makedirs("./data/uploads", exist_ok=True)

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    dest_path = os.path.join("./data/uploads", file.filename)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    chunks_count = index_document(dest_path, file.filename)
    return {"message": f"Processed {file.filename}", "chunks_stored": chunks_count}

@app.post("/query")
def answer_query(req: QueryRequest):
    return run_agentic_workflow(req.question)