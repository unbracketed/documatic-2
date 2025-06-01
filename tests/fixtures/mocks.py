"""Mock classes for testing the embedding pipeline."""

import hashlib
import time
from typing import Any
from unittest.mock import Mock


class RateLimitError(Exception):
    """Mock rate limit error for testing."""
    pass


class MockEmbeddingAPI:
    """Mock embedding API for testing."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.call_count = 0
        self.rate_limit_after: int | None = None
        self.fail_after: int | None = None
        self.response_delay = 0.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of mock embedding vectors
            
        Raises:
            RateLimitError: If rate limit is exceeded
            Exception: If configured to fail
        """
        self.call_count += 1

        # Simulate rate limiting
        if self.rate_limit_after and self.call_count > self.rate_limit_after:
            raise RateLimitError("Rate limit exceeded")

        # Simulate other failures
        if self.fail_after and self.call_count > self.fail_after:
            raise Exception("API failure")

        # Simulate response delay
        if self.response_delay > 0:
            time.sleep(self.response_delay)

        return [self._generate_embedding(text) for text in texts]

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding based on text hash.
        
        Args:
            text: Input text
            
        Returns:
            Deterministic embedding vector
        """
        # Create deterministic embedding based on text hash
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(seed * i) % 1000 / 1000 for i in range(self.dimension)]


class MockLanceDB:
    """Mock LanceDB for testing."""

    def __init__(self):
        self.tables: dict[str, dict[str, Any]] = {}
        self.data: dict[str, list[dict[str, Any]]] = {}

    def create_table(self, name: str, schema: dict[str, Any]) -> "MockTable":
        """Create a mock table.
        
        Args:
            name: Table name
            schema: Table schema
            
        Returns:
            Mock table instance
        """
        self.tables[name] = schema
        self.data[name] = []
        return MockTable(name, self)

    def open_table(self, name: str) -> "MockTable":
        """Open an existing mock table.
        
        Args:
            name: Table name
            
        Returns:
            Mock table instance
            
        Raises:
            FileNotFoundError: If table doesn't exist
        """
        if name not in self.tables:
            raise FileNotFoundError(f"Table {name} not found")
        return MockTable(name, self)


class MockTable:
    """Mock LanceDB table for testing."""

    def __init__(self, name: str, db: MockLanceDB):
        self.name = name
        self.db = db

    def add(self, data: Any) -> None:
        """Add data to mock table.
        
        Args:
            data: Data to add (PyArrow table or list of dicts)
        """
        if hasattr(data, 'to_pylist'):
            # PyArrow table
            try:
                records = data.to_pylist()
            except Exception:
                # Fallback: convert to dict manually
                records = []
                for i in range(len(data)):
                    record = {}
                    for col_name in data.column_names:
                        record[col_name] = data[col_name][i].as_py()
                    records.append(record)
        else:
            # Assume list of dicts
            records = data

        self.db.data[self.name].extend(records)

    def search(self, query: Any, query_type: str = "vector") -> "MockSearchResult":
        """Search the mock table.
        
        Args:
            query: Search query (vector or text)
            query_type: Type of search ("vector" or "fts")
            
        Returns:
            Mock search result
        """
        return MockSearchResult(self.db.data[self.name])

    def create_index(self, **kwargs: Any) -> None:
        """Create mock index (no-op)."""
        pass

    def create_fts_index(self, column: str, replace: bool = False) -> None:
        """Create mock full-text search index (no-op)."""
        pass

    def merge_insert(self, on_column: str) -> "MockMergeBuilder":
        """Create mock merge operation.
        
        Args:
            on_column: Column to merge on
            
        Returns:
            Mock merge builder
        """
        return MockMergeBuilder(self)

    def to_pandas(self) -> Mock:
        """Convert to pandas DataFrame (mocked)."""
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=len(self.db.data[self.name]))
        mock_df.to_dict = Mock(return_value={"records": self.db.data[self.name]})
        return mock_df


class MockSearchResult:
    """Mock search result for testing."""

    def __init__(self, data: list[dict[str, Any]]):
        self.data = data
        self._limit = len(data)
        self._where_clause: str | None = None

    def limit(self, n: int) -> "MockSearchResult":
        """Limit search results.
        
        Args:
            n: Maximum number of results
            
        Returns:
            Self for chaining
        """
        self._limit = n
        return self

    def where(self, condition: str) -> "MockSearchResult":
        """Add where clause to search.
        
        Args:
            condition: Where condition
            
        Returns:
            Self for chaining
        """
        self._where_clause = condition
        return self

    def to_pandas(self) -> Mock:
        """Convert to pandas DataFrame (mocked)."""
        # Apply limit
        limited_data = self.data[:self._limit]

        mock_df = Mock()
        mock_df.to_dict = Mock(return_value=limited_data)
        return mock_df


class MockMergeBuilder:
    """Mock merge builder for upsert operations."""

    def __init__(self, table: MockTable):
        self.table = table

    def when_matched_update_all(self) -> "MockMergeBuilder":
        """Configure update behavior (no-op)."""
        return self

    def when_not_matched_insert_all(self) -> "MockMergeBuilder":
        """Configure insert behavior (no-op)."""
        return self

    def execute(self, data: Any) -> None:
        """Execute merge operation (delegates to add)."""
        self.table.add(data)


class MockNetworkConditions:
    """Mock network conditions for testing."""

    def __init__(self):
        self.latency = 0.0
        self.failure_rate = 0.0
        self.timeout_rate = 0.0

    def apply_conditions(self) -> None:
        """Apply configured network conditions."""
        import random

        # Simulate latency
        if self.latency > 0:
            time.sleep(self.latency)

        # Simulate failures
        if random.random() < self.failure_rate:
            raise Exception("Network failure")

        # Simulate timeouts
        if random.random() < self.timeout_rate:
            raise TimeoutError("Network timeout")


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.call_count = 0

    def embed_text(self, text: str) -> list[float]:
        """Generate mock embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Mock embedding vector
        """
        self.call_count += 1
        # Create deterministic embedding based on text hash
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(seed * i) % 1000 / 1000 for i in range(self.dimension)]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of mock embedding vectors
        """
        return [self.embed_text(text) for text in texts]
