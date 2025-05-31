# Task: Search and Retrieval Layer

## Technical Requirements
- Create `documatic/search.py` module
- Implement multiple search strategies:
  - Vector similarity search
  - Full-text search with BM25
  - Hybrid search combining both
- Create reranking pipeline using cross-encoder or LLM
- Support query expansion and reformulation
- Return configurable number of results (default: 5)
- Include relevance scores and metadata in results
- Optimize for technical documentation queries

## Addresses Requirement
"powered by RAG, vector, full-text, and/or hybrid search techniques" - provides flexible retrieval options for different query types.