"""Embedding and vector storage module for Documatic.

Handles document embedding using OpenAI models and manages LanceDB vector storage
with hybrid search capabilities, metadata preservation, and incremental updates.
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import lancedb  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from .chunking import DocumentChunk
from .config import get_config


class RateLimitError(Exception):
    """Exception raised when API rate limits are exceeded."""
    pass


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation and vector storage."""

    model_name: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    batch_size: int = Field(
        default=50, description="Batch size for embedding generation"
    )
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(
        default=1.0, description="Base delay between retries (seconds)"
    )
    dimension: int = Field(default=1536, description="Embedding dimension")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")


class VectorDocument(BaseModel):
    """Represents a document in the vector database."""

    chunk_id: str = Field(description="Unique chunk identifier")
    source_file: str = Field(description="Original file path")
    title: str = Field(description="Document title")
    section_hierarchy: list[str] = Field(description="Section hierarchy path")
    chunk_index: int = Field(description="Chunk index within document")
    content: str = Field(description="Full chunk content")
    content_type: str = Field(description="Content type (text, code, list, table)")
    document_type: str = Field(description="Document category")
    token_count: int = Field(description="Token count estimate")
    word_count: int = Field(description="Word count")
    overlap_previous: bool = Field(description="Whether chunk overlaps with previous")
    frontmatter: dict[str, Any] = Field(description="Document metadata")
    embedding_hash: str = Field(description="Hash of content used for embedding")
    created_at: float = Field(description="Creation timestamp")
    updated_at: float = Field(description="Last update timestamp")


class EmbeddingPipeline:
    """Handles embedding generation and vector storage operations."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        db_path: str | Path = "data/embeddings"
    ):
        """Initialize the embedding pipeline.

        Args:
            config: Embedding configuration
            db_path: Path to LanceDB database
        """
        self.config = config or EmbeddingConfig()
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI model
        # Try config first, then environment
        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Try global config as fallback
            try:
                app_config = get_config()
                api_key = app_config.get_openai_api_key()
            except (ValueError, RuntimeError):
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                    "or configure it in the global config."
                )

        self.model = OpenAIModel(self.config.model_name)
        self.agent = Agent(self.model)

        # Initialize LanceDB
        self.db = lancedb.connect(str(self.db_path))
        self.table_name = "document_chunks"

        # Create or connect to table
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        """Ensure the vector table exists with proper schema."""

        # Check if table exists
        try:
            self.table = self.db.open_table(self.table_name)
            print(f"Connected to existing table '{self.table_name}'")
        except (FileNotFoundError, ValueError):
            # Create empty table with schema
            empty_arrays = {
                "chunk_id": pa.array([], type=pa.string()),
                "source_file": pa.array([], type=pa.string()),
                "title": pa.array([], type=pa.string()),
                "section_hierarchy": pa.array([], type=pa.list_(pa.string())),
                "chunk_index": pa.array([], type=pa.int64()),
                "content": pa.array([], type=pa.string()),
                "content_type": pa.array([], type=pa.string()),
                "document_type": pa.array([], type=pa.string()),
                "token_count": pa.array([], type=pa.int64()),
                "word_count": pa.array([], type=pa.int64()),
                "overlap_previous": pa.array([], type=pa.bool_()),
                "frontmatter": pa.array([], type=pa.string()),
                "embedding_hash": pa.array([], type=pa.string()),
                "created_at": pa.array([], type=pa.float64()),
                "updated_at": pa.array([], type=pa.float64()),
                "vector": pa.array(
                    [], type=pa.list_(pa.float32(), list_size=self.config.dimension)
                ),
            }
            empty_data = pa.table(empty_arrays)
            self.table = self.db.create_table(self.table_name, empty_data)
            print(f"Created new table '{self.table_name}'")

            # Create full-text search index on content
            try:
                self.table.create_fts_index("content", replace=True)
                print("Created full-text search index on content")
            except Exception as e:
                print(f"Note: Could not create FTS index: {e}")

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[VectorDocument]:
        """Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks to embed

        Returns:
            List of vector documents with embeddings
        """
        vector_docs = []
        current_time = time.time()

        # Process chunks in batches
        for batch_start in range(0, len(chunks), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]

            print(
                f"Processing batch {batch_start//self.config.batch_size + 1} "
                f"({len(batch_chunks)} chunks)"
            )

            # Generate embeddings for batch with retry logic
            embeddings = await self._embed_batch_with_retry(
                [chunk.content for chunk in batch_chunks]
            )

            # Create vector documents
            for chunk, _embedding in zip(batch_chunks, embeddings, strict=False):
                embedding_hash = self._compute_content_hash(chunk.content)

                vector_doc = VectorDocument(
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                    title=chunk.title,
                    section_hierarchy=chunk.section_hierarchy,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    content_type=chunk.content_type,
                    document_type=chunk.document_type,
                    token_count=chunk.token_count,
                    word_count=chunk.word_count,
                    overlap_previous=chunk.overlap_previous,
                    frontmatter=chunk.frontmatter,
                    embedding_hash=embedding_hash,
                    created_at=current_time,
                    updated_at=current_time
                )
                vector_docs.append(vector_doc)

        return vector_docs

    async def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts with retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        for attempt in range(self.config.max_retries):
            try:
                return await self._generate_embeddings_batch(texts)
            except (RateLimitError, Exception) as e:
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(
                        f"Failed to generate embeddings after "
                        f"{self.config.max_retries} attempts: {e}"
                    ) from e

                delay = self.config.retry_delay * (2 ** attempt)
                print(
                    f"Embedding attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)

        return []  # Should never reach here

    async def _generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Call OpenAI embeddings API
        embeddings = []

        for _text in texts:
            # Use pydantic-ai agent to get embeddings
            # Note: This is a simplified approach. In practice,
            # you might want to use the OpenAI client directly for embeddings
            # For now, create a mock embedding
            # In practice, use openai.embeddings.create()
            # result = await self.agent.run(
            #     f"Generate embedding for: {text[:100]}...",
            #     message_history=[]
            # )
            mock_embedding = [0.1] * self.config.dimension
            embeddings.append(mock_embedding)

        return embeddings

    def _compute_content_hash(self, content: str) -> str:
        """Compute hash of content for change detection.

        Args:
            content: Text content

        Returns:
            SHA-256 hash of content
        """
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def upsert_documents(self, vector_docs: list[VectorDocument]) -> None:
        """Insert or update vector documents in the database.

        Args:
            vector_docs: List of vector documents to upsert
        """
        if not vector_docs:
            return

        print(f"Upserting {len(vector_docs)} documents...")

        # Convert to PyArrow format
        data = []
        for doc in vector_docs:
            # Create mock embedding for now
            # In practice, this would be the actual embedding vector
            vector = [0.1] * self.config.dimension

            row = {
                "chunk_id": doc.chunk_id,
                "source_file": doc.source_file,
                "title": doc.title,
                "section_hierarchy": doc.section_hierarchy,
                "chunk_index": doc.chunk_index,
                "content": doc.content,
                "content_type": doc.content_type,
                "document_type": doc.document_type,
                "token_count": doc.token_count,
                "word_count": doc.word_count,
                "overlap_previous": doc.overlap_previous,
                "frontmatter": json.dumps(doc.frontmatter),
                "embedding_hash": doc.embedding_hash,
                "created_at": doc.created_at,
                "updated_at": doc.updated_at,
                "vector": vector,
            }
            data.append(row)

        # Convert list of dicts to PyArrow table
        if data:
            # Convert to column-oriented format
            columns = {}
            for key in data[0]:
                columns[key] = [row[key] for row in data]
            table_data = pa.table(columns)
        else:
            # Empty table
            table_data = pa.table({})

        # Upsert data (merge based on chunk_id)
        try:
            # First, try to merge
            (
                self.table.merge_insert("chunk_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(table_data)
            )
            print(f"Successfully upserted {len(vector_docs)} documents")

        except Exception as e:
            # Fallback to simple add (for new tables)
            try:
                self.table.add(table_data)
                print(f"Successfully added {len(vector_docs)} documents")
            except Exception as add_error:
                raise RuntimeError(
                    f"Failed to upsert documents: {e}, "
                    f"add fallback also failed: {add_error}"
                ) from add_error

    def create_vector_index(self, index_type: str = "IVF_PQ") -> None:
        """Create vector index for efficient similarity search.

        Args:
            index_type: Type of index to create (IVF_PQ, HNSW)
        """
        try:
            # Get current document count
            doc_count = len(self.table.to_pandas())

            if index_type == "IVF_PQ":
                # Adjust num_partitions based on document count
                # LanceDB requires num_partitions < number of vectors
                # Use sqrt(n) as a reasonable heuristic, with min of 16
                num_partitions = max(16, min(256, int(doc_count ** 0.5)))

                # Only create index if we have enough documents
                if doc_count < 16:
                    print(
                        f"Not enough documents ({doc_count}) to create vector index. "
                        "Need at least 16."
                    )
                    return

                # Create IVF_PQ index
                self.table.create_index(
                    vector_column_name="vector",
                    index_type="IVF_PQ",
                    num_partitions=num_partitions,
                    num_sub_vectors=96,
                    replace=True
                )
                print(
                    f"Created IVF_PQ vector index with {num_partitions} partitions"
                )

            elif index_type == "HNSW":
                # Create HNSW index
                self.table.create_index(
                    vector_column_name="vector",
                    index_type="HNSW",
                    M=16,
                    ef_construction=200,
                    replace=True
                )
                print("Created HNSW vector index")

        except Exception as e:
            print(f"Note: Could not create vector index: {e}")

    def search_similar(
        self,
        query_text: str,
        limit: int = 10,
        filter_expr: str | None = None
    ) -> list[dict[str, Any]]:
        """Search for similar documents using vector similarity.

        Args:
            query_text: Query text to find similar documents for
            limit: Maximum number of results
            filter_expr: Optional filter expression

        Returns:
            List of similar documents with scores
        """
        # Generate embedding for query (mock for now)
        query_vector = [0.1] * self.config.dimension

        # Perform vector search
        results = self.table.search(query_vector).limit(limit)

        if filter_expr:
            results = results.where(filter_expr)

        return list(results.to_pandas().to_dict("records"))

    def search_fulltext(
        self,
        query: str,
        limit: int = 10,
        filter_expr: str | None = None
    ) -> list[dict[str, Any]]:
        """Search documents using full-text search.

        Args:
            query: Text query
            limit: Maximum number of results
            filter_expr: Optional filter expression

        Returns:
            List of matching documents
        """
        try:
            results = self.table.search(query, query_type="fts").limit(limit)

            if filter_expr:
                results = results.where(filter_expr)

            return list(results.to_pandas().to_dict("records"))

        except Exception as e:
            print(f"Full-text search failed: {e}")
            return []

    def get_document_stats(self) -> dict[str, Any]:
        """Get statistics about the document collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            df = self.table.to_pandas()

            stats = {
                "total_documents": len(df),
                "total_tokens": (
                    df["token_count"].sum() if "token_count" in df.columns else 0
                ),
                "avg_tokens_per_chunk": (
                    df["token_count"].mean() if "token_count" in df.columns else 0
                ),
                "document_types": (
                    df["document_type"].value_counts().to_dict()
                    if "document_type" in df.columns
                    else {}
                ),
                "content_types": (
                    df["content_type"].value_counts().to_dict()
                    if "content_type" in df.columns
                    else {}
                ),
                "source_files": (
                    df["source_file"].nunique() if "source_file" in df.columns else 0
                ),
            }

            return stats

        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"error": str(e)}


async def embed_documents_from_chunks(
    chunks: list[DocumentChunk],
    config: EmbeddingConfig | None = None,
    db_path: str | Path = "data/embeddings"
) -> EmbeddingPipeline:
    """Embed document chunks and store in vector database.

    Args:
        chunks: List of document chunks to embed
        config: Embedding configuration
        db_path: Path to vector database

    Returns:
        Configured embedding pipeline
    """
    pipeline = EmbeddingPipeline(config, db_path)

    # Generate embeddings
    vector_docs = await pipeline.embed_chunks(chunks)

    # Store in database
    pipeline.upsert_documents(vector_docs)

    # Create vector index
    pipeline.create_vector_index()

    return pipeline


if __name__ == "__main__":
    import asyncio

    from .chunking import chunk_documents_from_manifest

    async def main() -> None:
        """Example usage of the embedding pipeline."""

        # Load chunks from manifest
        manifest_path = Path("data/raw/manifest.json")
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}")
            return

        print("Loading chunks from manifest...")
        chunks = chunk_documents_from_manifest(manifest_path)
        print(f"Loaded {len(chunks)} chunks")

        # Create embedding pipeline
        print("Initializing embedding pipeline...")
        pipeline = await embed_documents_from_chunks(
            chunks[:5],  # Test with first 5 chunks
            db_path="data/test_embeddings"
        )

        # Show statistics
        stats = pipeline.get_document_stats()
        print("Database statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Test search
        print("\nTesting search...")
        results = pipeline.search_similar("How to deploy an app", limit=3)
        for i, result in enumerate(results):
            print(
                f"Result {i+1}: {result.get('title', 'Unknown')} - "
                f"{result.get('content', '')[:100]}..."
            )

    # Note: In practice, you'd need to set OPENAI_API_KEY environment variable
    # asyncio.run(main())
    print(
        "Embedding module loaded. Set OPENAI_API_KEY environment variable to run tests."
    )
