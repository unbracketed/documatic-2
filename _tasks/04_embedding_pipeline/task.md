# Task: Embedding and Vector Storage

## Technical Requirements
- Create `documatic/embeddings.py` module
- Configure embedding model (OpenAI ada-002 or similar)
- Implement batch embedding with retry logic
- Create LanceDB schema with:
  - vector field for embeddings
  - text fields for content and metadata
  - full-text search index on content
- Design table structure supporting hybrid search
- Implement upsert logic to handle document updates
- Add vector indexing (IVF_PQ or HNSW)

## Addresses Requirement
"chunk and embed the content into a table(s); suggest and capture metadata about the source docs" and "run indexing if needed" - creates the searchable knowledge base.