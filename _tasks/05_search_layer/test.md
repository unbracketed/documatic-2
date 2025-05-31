# Test Specification: Search and Retrieval Layer

## Overview
This test specification covers the search module (`documatic/search.py`) which implements multiple search strategies and reranking for optimal retrieval from the vector database.

## Unit Tests

### 1. Vector Search Tests (`test_vector_search.py`)
- **Test query embedding**
  - Generate query embeddings
  - Handle different query lengths
  - Validate embedding dimensions
  
- **Test similarity search**
  - Cosine similarity calculation
  - Top-k retrieval
  - Score normalization
  - Empty result handling
  
- **Test distance metrics**
  - Cosine distance (default)
  - Euclidean distance option
  - Dot product option
  - Metric comparison tests

### 2. Full-Text Search Tests (`test_fulltext_search.py`)
- **Test BM25 implementation**
  - Term frequency calculation
  - Inverse document frequency
  - Length normalization
  - Parameter tuning (k1, b)
  
- **Test query processing**
  - Tokenization
  - Stopword removal
  - Stemming/lemmatization
  - Query expansion
  
- **Test scoring**
  - BM25 score calculation
  - Score normalization
  - Multi-term queries
  - Phrase matching

### 3. Hybrid Search Tests (`test_hybrid_search.py`)
- **Test score fusion**
  - Reciprocal rank fusion (RRF)
  - Linear combination
  - Weighted scoring
  - Score normalization
  
- **Test result merging**
  - Duplicate detection
  - Consistent ordering
  - Metadata preservation
  - Score aggregation
  
- **Test weight optimization**
  - Dynamic weight adjustment
  - Query-type detection
  - Performance tracking

### 4. Reranking Tests (`test_reranking.py`)
- **Test cross-encoder reranking**
  - Model initialization
  - Pairwise scoring
  - Batch processing
  - Score calibration
  
- **Test LLM reranking**
  - Prompt construction
  - Context window limits
  - Result parsing
  - Fallback handling
  
- **Test reranking strategies**
  - Top-k reranking only
  - Full result reranking
  - Cascaded reranking
  - Performance vs quality tradeoff

### 5. Query Processing Tests (`test_query_processing.py`)
- **Test query expansion**
  - Synonym expansion
  - Acronym resolution
  - Technical term handling
  - Context preservation
  
- **Test query reformulation**
  - Question to statement
  - Keyword extraction
  - Intent detection
  - Multiple reformulations
  
- **Test query analysis**
  - Query type classification
  - Length analysis
  - Complexity scoring
  - Language detection

## Integration Tests

### 1. End-to-End Search Tests (`test_search_integration.py`)
- **Test complete search pipeline**
  - Query → Results flow
  - All search strategies
  - Reranking application
  - Metadata inclusion
  
- **Test different query types**
  - Keyword queries: "apppack deployment"
  - Natural language: "How do I deploy an app?"
  - Code queries: "python flask example"
  - Error messages: "ModuleNotFoundError numpy"
  
- **Test result quality**
  - Relevance validation
  - Ranking order
  - Diversity of results
  - Coverage of topics

### 2. Performance Tests (`test_search_performance.py`)
- **Test search latency**
  - Vector search: <50ms
  - Full-text search: <100ms
  - Hybrid search: <150ms
  - With reranking: <500ms
  
- **Test throughput**
  - Queries per second
  - Concurrent query handling
  - Cache effectiveness
  - Resource utilization
  
- **Test scalability**
  - 100k+ document corpus
  - Complex queries
  - Multiple concurrent users
  - Memory usage patterns

## Edge Cases and Error Handling

### 1. Query Edge Cases (`test_query_edge_cases.py`)
- **Test unusual queries**
  - Empty queries
  - Single character
  - Very long queries (>1000 chars)
  - Special characters only
  - SQL injection attempts
  
- **Test language variations**
  - Mixed languages
  - Code in queries
  - Mathematical symbols
  - Emoji and unicode
  
- **Test malformed input**
  - Invalid JSON
  - Null values
  - Type mismatches
  - Encoding issues

### 2. System Error Tests (`test_search_errors.py`)
- **Test database failures**
  - Connection timeout
  - Index corruption
  - Partial results
  - Recovery strategies
  
- **Test model failures**
  - Embedding API down
  - Reranking model errors
  - Fallback to basic search
  - Graceful degradation
  
- **Test resource limits**
  - Memory exhaustion
  - CPU throttling
  - Result size limits
  - Timeout handling

## Mock/Stub Requirements

### 1. Search Backend Mock
```python
class MockSearchBackend:
    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.call_history = []
    
    def vector_search(self, query_embedding: List[float], k: int) -> List[dict]:
        self.call_history.append(('vector', query_embedding, k))
        # Return mock results with scores
        return self._mock_results(k, score_range=(0.7, 0.95))
    
    def fulltext_search(self, query: str, k: int) -> List[dict]:
        self.call_history.append(('fulltext', query, k))
        # Return mock results with BM25 scores
        return self._mock_results(k, score_range=(5.0, 25.0))
    
    def _mock_results(self, k: int, score_range: tuple) -> List[dict]:
        # Generate consistent mock results
        pass
```

### 2. Reranking Model Mock
```python
class MockReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.rerank_calls = 0
    
    def rerank(self, query: str, documents: List[str]) -> List[tuple]:
        self.rerank_calls += 1
        # Return documents with mock scores
        scores = [0.9 - i * 0.1 for i in range(len(documents))]
        return list(zip(documents, scores))
```

### 3. Query Processor Mock
```python
class MockQueryProcessor:
    def __init__(self):
        self.expansions = {
            "app": ["application", "app", "software"],
            "deploy": ["deployment", "deploy", "release"],
            "k8s": ["kubernetes", "k8s"]
        }
    
    def expand_query(self, query: str) -> List[str]:
        # Simple word-based expansion
        words = query.lower().split()
        expanded = []
        for word in words:
            expanded.extend(self.expansions.get(word, [word]))
        return expanded
```

## Test Data Requirements

### 1. Query Test Sets (`tests/fixtures/queries/`)
```json
{
  "keyword_queries": [
    "apppack deployment",
    "docker configuration",
    "environment variables"
  ],
  "natural_language_queries": [
    "How do I deploy a Python application?",
    "What are the best practices for scaling?",
    "Why is my deployment failing?"
  ],
  "code_queries": [
    "flask app.py example",
    "dockerfile FROM python:3.9",
    "yaml configuration syntax"
  ],
  "complex_queries": [
    "deploy python flask app with postgresql database and redis cache to production",
    "troubleshoot 502 bad gateway error nginx ingress controller kubernetes"
  ]
}
```

### 2. Expected Results (`tests/fixtures/expected_results/`)
- Ground truth relevance judgments
- Expected ranking orders
- Minimum relevance scores
- Required metadata fields

### 3. Document Corpus (`tests/fixtures/search_corpus/`)
```python
# Generate test corpus with known characteristics
def create_test_corpus():
    return [
        {
            "id": "doc1",
            "content": "AppPack deployment guide...",
            "embedding": [0.1, 0.2, ...],
            "metadata": {"type": "guide", "section": "deployment"}
        },
        # ... more documents with varied content
    ]
```

### 4. Search Configuration
```python
# tests/fixtures/search_configs.py
DEFAULT_CONFIG = {
    "vector_weight": 0.7,
    "fulltext_weight": 0.3,
    "rerank_top_k": 20,
    "final_top_k": 5,
    "enable_query_expansion": True
}

PERFORMANCE_CONFIG = {
    "vector_weight": 1.0,  # Vector only for speed
    "fulltext_weight": 0.0,
    "rerank_top_k": 0,     # No reranking
    "final_top_k": 5
}

QUALITY_CONFIG = {
    "vector_weight": 0.5,
    "fulltext_weight": 0.5,
    "rerank_top_k": 50,    # Rerank more documents
    "final_top_k": 5,
    "enable_llm_rerank": True
}
```

## Test Execution Strategy

1. **Unit test coverage**: All search strategies independently
2. **Integration testing**: Complete search pipelines
3. **Quality evaluation**: Result relevance metrics
4. **Performance profiling**: Latency and throughput
5. **A/B testing**: Compare search strategies

## Quality Metrics

### Search Quality Measurements
- **Precision@k**: Relevant results in top-k
- **Recall@k**: Coverage of relevant documents
- **NDCG**: Normalized discounted cumulative gain
- **MRR**: Mean reciprocal rank
- **User satisfaction**: Click-through rate proxy

### Performance Benchmarks
- **Latency targets**:
  - P50: <100ms
  - P95: <300ms
  - P99: <500ms
- **Throughput**: >100 QPS
- **Cache hit rate**: >70%
- **Memory usage**: <1GB per instance

## Success Criteria

- All unit tests pass with >90% coverage
- Integration tests show consistent results
- Search quality metrics meet thresholds:
  - Precision@5 > 0.8
  - Recall@10 > 0.9
  - NDCG@5 > 0.85
- Performance meets latency targets
- System handles edge cases gracefully
- Results are explainable and debuggable
- Technical queries perform well