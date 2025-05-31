# Test Specification: Embedding and Vector Storage

## Overview
This test specification covers the embedding pipeline module (`documatic/embeddings.py`) which handles text embedding generation and vector storage in LanceDB.

## Unit Tests

### 1. Embedding Model Tests (`test_embedding_model.py`)
- **Test model initialization**
  - Correct API key configuration
  - Model selection (ada-002 or alternatives)
  - Connection validation
  - Error handling for invalid credentials
  
- **Test embedding generation**
  - Single text embedding
  - Embedding dimension validation
  - Deterministic results for same input
  - Handle empty strings
  - Handle special characters and unicode
  
- **Test token limits**
  - Respect model token limits
  - Truncation strategy for long texts
  - Warning logs for truncation

### 2. Batch Processing Tests (`test_batch_processing.py`)
- **Test batch sizing**
  - Optimal batch size determination
  - Dynamic batch adjustment based on content
  - Handle mixed content lengths
  
- **Test retry logic**
  - Retry on rate limits (429 errors)
  - Exponential backoff implementation
  - Max retry limits
  - Partial batch failure handling
  
- **Test error recovery**
  - Continue processing after failures
  - Track failed embeddings
  - Provide failure report

### 3. LanceDB Schema Tests (`test_lancedb_schema.py`)
- **Test schema creation**
  - Vector field with correct dimensions
  - Text fields for content and metadata
  - Proper data types for each field
  - Index creation
  
- **Test schema validation**
  - Reject invalid vector dimensions
  - Enforce required fields
  - Validate metadata structure
  
- **Test schema migration**
  - Handle schema updates
  - Backward compatibility
  - Data preservation during migration

### 4. Vector Storage Tests (`test_vector_storage.py`)
- **Test insertion operations**
  - Single document insertion
  - Batch insertion
  - Duplicate handling
  - Transaction support
  
- **Test upsert logic**
  - Update existing documents
  - Preserve document history
  - Maintain referential integrity
  - Version tracking
  
- **Test indexing**
  - IVF_PQ index creation
  - HNSW index as alternative
  - Index performance metrics
  - Index rebuilding

## Integration Tests

### 1. End-to-End Pipeline Tests (`test_embedding_pipeline_integration.py`)
- **Test full document processing**
  - From chunks to stored vectors
  - Metadata preservation
  - Batch processing of multiple documents
  
- **Test incremental updates**
  - Add new documents
  - Update existing documents
  - Delete outdated documents
  - Verify index consistency
  
- **Test search functionality**
  - Vector similarity search
  - Full-text search
  - Hybrid search combining both
  - Result ranking accuracy

### 2. Performance Tests (`test_embedding_performance.py`)
- **Test throughput**
  - Embeddings per second
  - Optimal batch size finding
  - API rate limit handling
  
- **Test latency**
  - Single embedding latency
  - Batch processing latency
  - End-to-end pipeline latency
  
- **Test scalability**
  - Handle 10k+ documents
  - Memory usage monitoring
  - Database size growth

## Edge Cases and Error Handling

### 1. API Error Tests (`test_api_errors.py`)
- **Test rate limiting**
  - 429 error handling
  - Automatic retry with backoff
  - Rate limit tracking
  
- **Test network errors**
  - Connection timeouts
  - DNS failures
  - Partial response handling
  
- **Test API changes**
  - Model deprecation warnings
  - API version compatibility
  - Fallback strategies

### 2. Data Error Tests (`test_data_errors.py`)
- **Test malformed input**
  - Binary data in text fields
  - Extremely long texts
  - Invalid UTF-8 sequences
  
- **Test database errors**
  - Connection failures
  - Disk space issues
  - Concurrent access conflicts
  
- **Test corruption recovery**
  - Detect corrupted embeddings
  - Rebuild from source
  - Maintain data integrity

## Mock/Stub Requirements

### 1. Embedding API Mock
```python
class MockEmbeddingAPI:
    def __init__(self, dimension=1536):
        self.dimension = dimension
        self.call_count = 0
        self.rate_limit_after = None
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        if self.rate_limit_after and self.call_count > self.rate_limit_after:
            raise RateLimitError()
        return [self._generate_embedding(text) for text in texts]
    
    def _generate_embedding(self, text: str) -> List[float]:
        # Deterministic embedding based on text hash
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(seed * i) % 1000 / 1000 for i in range(self.dimension)]
```

### 2. LanceDB Mock
```python
class MockLanceDB:
    def __init__(self):
        self.tables = {}
        self.data = {}
    
    def create_table(self, name: str, schema: dict):
        self.tables[name] = schema
        self.data[name] = []
    
    def insert(self, table: str, records: List[dict]):
        self.data[table].extend(records)
    
    def search(self, table: str, vector: List[float], k: int):
        # Simple mock search implementation
        return self.data[table][:k]
```

### 3. Network Mock
- Simulate various network conditions
- Control response times
- Inject specific errors

## Test Data Requirements

### 1. Text Samples (`tests/fixtures/embedding_texts/`)
- **normal_text.txt**: Standard documentation paragraph
- **code_heavy.txt**: Code-heavy content
- **unicode_text.txt**: Various unicode characters
- **empty_text.txt**: Empty and whitespace-only
- **large_text.txt**: 8000+ tokens (requires truncation)
- **special_chars.txt**: Special characters and symbols

### 2. Expected Embeddings (`tests/fixtures/expected_embeddings/`)
- Pre-calculated embeddings for validation
- Known similar/dissimilar text pairs
- Edge case embeddings

### 3. Database Fixtures
```python
# tests/fixtures/db_configs.py
TEST_SCHEMA = {
    "vector": "vector(1536)",
    "content": "text",
    "chunk_id": "string",
    "document_id": "string",
    "metadata": "json",
    "created_at": "timestamp",
    "updated_at": "timestamp"
}

PERFORMANCE_CONFIG = {
    "index_type": "IVF_PQ",
    "nlist": 100,
    "nprobe": 10,
    "metric": "cosine"
}
```

### 4. Batch Test Data
```python
# tests/fixtures/batch_data.py
def generate_test_chunks(count: int) -> List[dict]:
    """Generate test chunks with varied characteristics"""
    chunks = []
    for i in range(count):
        chunks.append({
            "content": f"Test content {i}" * (i % 100 + 10),
            "metadata": {
                "source": f"doc_{i % 10}.md",
                "section": f"Section {i % 5}"
            }
        })
    return chunks
```

## Test Execution Strategy

1. **Unit tests**: Verify individual components in isolation
2. **Integration tests**: Test component interactions
3. **Load tests**: Process large datasets
4. **Stress tests**: Push system limits
5. **Recovery tests**: Test failure recovery

## Performance Benchmarks

### Expected Performance Metrics
- **Embedding generation**: <100ms per text (API dependent)
- **Batch processing**: >100 texts/second with batching
- **Database insertion**: >1000 records/second
- **Index building**: <5 minutes for 100k documents
- **Search latency**: <50ms for vector search
- **Memory usage**: <2GB for 100k documents

## Success Criteria

- All unit tests pass with >95% code coverage
- Integration tests complete without errors
- Performance meets or exceeds benchmarks
- Retry logic handles all transient failures
- Zero data loss during updates
- Search results are relevant and properly ranked
- System handles 100k+ documents efficiently