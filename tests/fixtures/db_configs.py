"""Database configuration fixtures for testing."""

import pyarrow as pa

TEST_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("source_file", pa.string()),
    pa.field("title", pa.string()),
    pa.field("section_hierarchy", pa.list_(pa.string())),
    pa.field("chunk_index", pa.int64()),
    pa.field("content", pa.string()),
    pa.field("content_type", pa.string()),
    pa.field("document_type", pa.string()),
    pa.field("token_count", pa.int64()),
    pa.field("word_count", pa.int64()),
    pa.field("overlap_previous", pa.bool_()),
    pa.field("frontmatter", pa.string()),  # JSON string
    pa.field("embedding_hash", pa.string()),
    pa.field("created_at", pa.float64()),
    pa.field("updated_at", pa.float64()),
    pa.field("vector", pa.list_(pa.float32(), list_size=1536)),
])

PERFORMANCE_CONFIG = {
    "index_type": "IVF_PQ",
    "nlist": 100,
    "nprobe": 10,
    "metric": "cosine"
}

ALTERNATIVE_PERFORMANCE_CONFIG = {
    "index_type": "HNSW",
    "M": 16,
    "ef_construction": 200,
    "metric": "cosine"
}
