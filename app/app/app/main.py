from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from .rag_engine import RAGEngine
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Enterprise RAG API",
    description="A production-grade Retrieval-Augmented Generation API with pgvector.",
    version="1.0.0"
)

# Initialize the RAG Engine (In production, use Dependency Injection)
DB_URL = os.getenv("DATABASE_URL", "postgresql://ai_engineer:secure_password_123@localhost:5432/rag_database")
rag_engine = RAGEngine(DB_URL)

class IngestRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict]] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/ingest", status_code=201)
def ingest_documents(payload: IngestRequest):
    """Endpoint to embed and store new documents."""
    try:
        rag_engine.ingest_documents(payload.texts, payload.metadatas)
        return {"message": f"Successfully ingested {len(payload.texts)} documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_documents(payload: QueryRequest):
    """Endpoint to perform semantic search over the ingested documents."""
    try:
        results = rag_engine.search(payload.query, payload.top_k)
        return {"query": payload.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Enterprise RAG API"}
