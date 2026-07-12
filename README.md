# 🏢 Enterprise RAG Pipeline with Hybrid Search

A production-grade Retrieval-Augmented Generation (RAG) API designed for high-accuracy, low-latency semantic search over enterprise documents. 

Built with **FastAPI**, **PostgreSQL (pgvector)**, and **SentenceTransformers**, this system moves beyond naive vector search by implementing a robust, scalable architecture suitable for production environments.

## 🏗️ System Architecture

Unlike standard RAG implementations that rely on managed vector databases (like Pinecone), this system is self-hosted using PostgreSQL. This reduces infrastructure costs and allows for complex relational queries alongside vector similarity search.

**Data Flow:**
1. **Ingestion:** Documents are chunked and embedded using a lightweight local model (`all-MiniLM-L6-v2`).
2. **Storage:** Embeddings are stored in a PostgreSQL database using the `pgvector` extension, alongside raw text and JSON metadata.
3. **Indexing:** An HNSW (Hierarchical Navigable Small World) index is automatically created for $O(\log N)$ approximate nearest neighbor (ANN) lookups.
4. **Retrieval:** User queries are embedded and searched using Cosine Similarity.

## 🛠️ Tech Stack

| Component | Technology | Why this choice? |
| :--- | :--- | :--- |
| **API Framework** | FastAPI | High performance, async support, and automatic OpenAPI docs. |
| **Vector Database** | PostgreSQL + pgvector | Unifies relational data and vector search; reduces vendor lock-in. |
| **Embeddings** | SentenceTransformers | Fast, local inference without relying on paid external APIs. |
| **Infrastructure** | Docker Compose | Ensures one-command setup for the database environment. |

## 🚀 Key Features

- **Zero External API Dependencies:** Embeddings are generated locally, ensuring data privacy and zero per-token API costs.
- **Automated Indexing:** Automatically builds HNSW indexes on startup for sub-second query latency, even with millions of vectors.
- **Metadata Filtering:** Supports JSONB metadata storage, allowing for pre-filtering or post-filtering during retrieval.
- **Containerized Database:** Spins up a fully configured `pgvector` database instantly via Docker.

## ⚙️ Local Development Setup

*Prerequisites: Docker and Python 3.10+*

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[Your-Username]/enterprise-rag-pipeline.git
   cd enterprise-rag-pipeline
