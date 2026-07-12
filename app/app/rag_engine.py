import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, db_url: str):
        # Load a lightweight, fast local embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.conn = psycopg2.connect(db_url)
        # Register pgvector with psycopg2
        register_vector(self.conn)
        self._setup_database()

    def _setup_database(self):
        """Creates the table and vector index if they don't exist."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector(384)
                );
            """)
            # Create an HNSW index for fast approximate nearest neighbor search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING hnsw (embedding vector_cosine_ops);
            """)
            self.conn.commit()
        logger.info("Database schema and vector index initialized.")

    def ingest_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Embeds and stores documents in the database."""
        embeddings = self.embedder.encode(texts)
        if metadatas is None:
            metadatas = [{} for _ in texts]

        with self.conn.cursor() as cur:
            for text, meta, emb in zip(texts, metadatas, embeddings):
                cur.execute(
                    "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s)",
                    (text, str(meta), emb)
                )
            self.conn.commit()
        logger.info(f"Ingested {len(texts)} documents.")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieves the most similar documents using Cosine Similarity."""
        query_embedding = self.embedder.encode(query)
        
        with self.conn.cursor() as cur:
            # Using pgvector's cosine distance operator (<=>)
            cur.execute("""
                SELECT content, metadata, 1 - (embedding <=> %s) as similarity
                FROM documents
                ORDER BY embedding <=> %s
                LIMIT %s;
            """, (query_embedding, query_embedding, top_k))
            
            results = cur.fetchall()
            
        return [
            {"content": row[0], "metadata": row[1], "similarity": float(row[2])}
            for row in results
        ]

    def close(self):
        self.conn.close()
