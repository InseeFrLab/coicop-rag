# RAG classifier for product 

Using coicop classification.

## Starting

```python
uv sync
```

## Workflow

### 1_create_vector_db.py

Embed coicop's notices into a Qdrant vector database, using several different strategies.

1. **Embedding Generation**:
   - Text notices are processed through the VLLM embedding model
   - Embeddings are generated using the `VLLM_EMBEDDING_URL` endpoint
   - Authentication is handled via `VLLM_EMBEDDING_API_KEY`

2. **Vector Storage**:
   - Generated embeddings are stored in Qdrant vector database
   - Connection is established using `QDRANT_URL`, `QDRANT_API_KEY` and `QDRANT_API_PORT`

### 2_run_rag.py

3. **Prompt Management**:
   - Prompt templates are stored and managed in Langfuse
   - Connection is established using `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`

4. **Retrieval and Generation**:
   - Relevant context is retrieved from Qdrant based on query embeddings
   - Retrieved context is passed to the VLLM generation model
   - Final response is generated using the `VLLM_GENERATION_URL` endpoint
   - Authentication is handled via `VLLM_GENERATION_API_KEY`

5. **MLflow Logging**:
   - Model performance metrics are logged to MLflow
   - Connection is established using `MLFLOW_TRACKING_URI`



