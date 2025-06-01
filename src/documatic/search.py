"""Search and retrieval layer for Documatic.

Implements multiple search strategies including vector similarity, full-text search
with BM25, hybrid search, and reranking capabilities for optimal document retrieval.
"""

import math
import re
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from .embeddings import EmbeddingPipeline


class SearchResult(BaseModel):
    """Represents a search result with score and metadata."""

    chunk_id: str = Field(description="Unique chunk identifier")
    source_file: str = Field(description="Original file path")
    title: str = Field(description="Document title")
    section_hierarchy: list[str] = Field(description="Section hierarchy path")
    content: str = Field(description="Chunk content")
    content_type: str = Field(description="Content type")
    document_type: str = Field(description="Document category")
    score: float = Field(description="Relevance score")
    search_method: str = Field(description="Search method used")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SearchConfig(BaseModel):
    """Configuration for search operations."""

    vector_weight: float = Field(
        default=0.7, description="Weight for vector search in hybrid"
    )
    fulltext_weight: float = Field(
        default=0.3, description="Weight for full-text search in hybrid"
    )
    rerank_top_k: int = Field(default=20, description="Number of results to rerank")
    final_top_k: int = Field(default=5, description="Final number of results to return")
    enable_query_expansion: bool = Field(
        default=True, description="Enable query expansion"
    )
    enable_reranking: bool = Field(default=False, description="Enable result reranking")
    bm25_k1: float = Field(default=1.2, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, description="BM25 b parameter")


class QueryProcessor:
    """Handles query preprocessing and expansion."""

    def __init__(self) -> None:
        """Initialize query processor."""
        # Technical term expansions for AppPack domain
        self.expansions = {
            "app": ["application", "app", "software", "service"],
            "deploy": ["deployment", "deploy", "release", "publish"],
            "k8s": ["kubernetes", "k8s", "cluster"],
            "db": ["database", "db", "datastore"],
            "env": ["environment", "env", "configuration"],
            "config": ["configuration", "config", "settings"],
            "docker": ["container", "docker", "image"],
            "ci": ["continuous integration", "ci", "build"],
            "cd": ["continuous deployment", "cd", "delivery"],
        }

        # Common acronyms
        self.acronyms = {
            "api": "application programming interface",
            "cli": "command line interface",
            "ui": "user interface",
            "url": "uniform resource locator",
            "ssl": "secure sockets layer",
            "tls": "transport layer security",
            "http": "hypertext transfer protocol",
            "https": "hypertext transfer protocol secure",
        }

    def preprocess_query(self, query: str) -> str:
        """Preprocess query text.

        Args:
            query: Raw query text

        Returns:
            Preprocessed query text
        """
        # Basic cleaning
        query = query.strip().lower()

        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)

        return query

    def expand_query(self, query: str) -> list[str]:
        """Expand query with synonyms and related terms.

        Args:
            query: Original query

        Returns:
            List of expanded query variants
        """
        expanded_queries = [query]
        words = query.lower().split()

        # Expand with synonyms
        for word in words:
            if word in self.expansions:
                for synonym in self.expansions[word]:
                    if synonym != word:
                        expanded_query = query.replace(word, synonym)
                        expanded_queries.append(expanded_query)

        # Expand acronyms
        for acronym, expansion in self.acronyms.items():
            if acronym in query:
                expanded_query = query.replace(acronym, expansion)
                expanded_queries.append(expanded_query)

        return list(set(expanded_queries))  # Remove duplicates

    def extract_keywords(self, query: str) -> list[str]:
        """Extract key terms from query.

        Args:
            query: Query text

        Returns:
            List of key terms
        """
        # Simple keyword extraction (remove stopwords)
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "how", "what", "where", "when", "why",
            "i", "me", "my", "we", "our", "you", "your", "do", "does", "can"
        }

        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in stopwords and len(word) > 2]


class BM25:
    """BM25 scoring implementation for full-text search."""

    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 scorer.

        Args:
            documents: Corpus of documents
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_count = len(documents)

        # Precompute document statistics
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0
        )

        # Build term frequency and document frequency
        self.term_frequencies: list[dict[str, int]] = []
        self.document_frequencies: dict[str, int] = {}

        for doc in documents:
            tf: dict[str, int] = {}
            words = doc.lower().split()
            for word in words:
                tf[word] = tf.get(word, 0) + 1
                if word not in self.document_frequencies:
                    self.document_frequencies[word] = 0
            self.term_frequencies.append(tf)

        # Count document frequencies
        for tf in self.term_frequencies:
            for word in tf:
                self.document_frequencies[word] += 1

    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for query against document.

        Args:
            query: Query text
            doc_idx: Document index

        Returns:
            BM25 score
        """
        if doc_idx >= len(self.documents):
            return 0.0

        score = 0.0
        query_terms = query.lower().split()
        doc_tf = self.term_frequencies[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        for term in query_terms:
            if term in doc_tf:
                tf = doc_tf[term]
                df = self.document_frequencies.get(term, 0)

                if df > 0:
                    # IDF calculation
                    idf = math.log((self.doc_count - df + 0.5) / (df + 0.5))

                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (
                        1 - self.b + self.b * doc_length / self.avg_doc_length
                    )
                    score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search documents using BM25.

        Args:
            query: Query text
            top_k: Number of top results to return

        Returns:
            List of (document_index, score) tuples
        """
        scores = []
        for i in range(len(self.documents)):
            score = self.score(query, i)
            if score > 0:
                scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class SearchLayer:
    """Main search interface supporting multiple search strategies."""

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        config: SearchConfig | None = None
    ):
        """Initialize search layer.

        Args:
            embedding_pipeline: Configured embedding pipeline
            config: Search configuration
        """
        self.pipeline = embedding_pipeline
        self.config = config or SearchConfig()
        self.query_processor = QueryProcessor()

        # Cache for BM25 index
        self._bm25_index: BM25 | None = None
        self._documents_cache: list[str] | None = None

        # Initialize reranking model if enabled
        self.reranker: Agent | None = None
        if self.config.enable_reranking:
            try:
                model = OpenAIModel("gpt-4o-mini")
                self.reranker = Agent(model)
            except Exception as e:
                print(f"Warning: Could not initialize reranker: {e}")
                self.config.enable_reranking = False

    def _get_bm25_index(self) -> BM25:
        """Get or create BM25 index from document corpus."""
        try:
            # Get all documents from vector database
            df = self.pipeline.table.to_pandas()
            documents = df['content'].tolist()

            # Check if we need to rebuild index
            if (self._bm25_index is None or
                self._documents_cache != documents):

                print(f"Building BM25 index for {len(documents)} documents...")
                self._bm25_index = BM25(
                    documents,
                    k1=self.config.bm25_k1,
                    b=self.config.bm25_b
                )
                self._documents_cache = documents

            return self._bm25_index

        except Exception as e:
            print(f"Error building BM25 index: {e}")
            # Return empty index as fallback

        return BM25([], k1=self.config.bm25_k1, b=self.config.bm25_b)

    def vector_search(
        self,
        query: str,
        limit: int = 10,
        filter_expr: str | None = None
    ) -> list[SearchResult]:
        """Perform vector similarity search.

        Args:
            query: Query text
            limit: Maximum number of results
            filter_expr: Optional filter expression

        Returns:
            List of search results
        """
        try:
            # Use existing search_similar method from embedding pipeline
            raw_results = self.pipeline.search_similar(
                query, limit=limit, filter_expr=filter_expr
            )

            # Convert to SearchResult objects
            results = []
            for result in raw_results:
                search_result = SearchResult(
                    chunk_id=result.get('chunk_id', ''),
                    source_file=result.get('source_file', ''),
                    title=result.get('title', ''),
                    section_hierarchy=result.get('section_hierarchy', []),
                    content=result.get('content', ''),
                    content_type=result.get('content_type', ''),
                    document_type=result.get('document_type', ''),
                    score=result.get('_distance', 0.0),  # LanceDB returns _distance
                    search_method="vector",
                    metadata={
                        "chunk_index": result.get('chunk_index', 0),
                        "token_count": result.get('token_count', 0),
                        "word_count": result.get('word_count', 0)
                    }
                )
                results.append(search_result)

            return results

        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def fulltext_search(
        self,
        query: str,
        limit: int = 10,
        filter_expr: str | None = None
    ) -> list[SearchResult]:
        """Perform full-text search using BM25.

        Args:
            query: Query text
            limit: Maximum number of results
            filter_expr: Optional filter expression

        Returns:
            List of search results
        """
        try:
            # Try LanceDB full-text search first
            raw_results = self.pipeline.search_fulltext(
                query, limit=limit, filter_expr=filter_expr
            )

            if raw_results:
                # Convert LanceDB FTS results
                results = []
                for result in raw_results:
                    search_result = SearchResult(
                        chunk_id=result.get('chunk_id', ''),
                        source_file=result.get('source_file', ''),
                        title=result.get('title', ''),
                        section_hierarchy=result.get('section_hierarchy', []),
                        content=result.get('content', ''),
                        content_type=result.get('content_type', ''),
                        document_type=result.get('document_type', ''),
                        score=result.get('score', 0.0),
                        search_method="fulltext_fts",
                        metadata={
                            "chunk_index": result.get('chunk_index', 0),
                            "token_count": result.get('token_count', 0),
                            "word_count": result.get('word_count', 0)
                        }
                    )
                    results.append(search_result)
                return results

            # Fallback to BM25 implementation
            print("Falling back to BM25 search...")
            bm25_index = self._get_bm25_index()
            bm25_results = bm25_index.search(query, top_k=limit)

            # Get document data
            df = self.pipeline.table.to_pandas()

            results = []
            for doc_idx, score in bm25_results:
                if doc_idx < len(df):
                    row = df.iloc[doc_idx]
                    search_result = SearchResult(
                        chunk_id=row.get('chunk_id', ''),
                        source_file=row.get('source_file', ''),
                        title=row.get('title', ''),
                        section_hierarchy=row.get('section_hierarchy', []),
                        content=row.get('content', ''),
                        content_type=row.get('content_type', ''),
                        document_type=row.get('document_type', ''),
                        score=score,
                        search_method="fulltext_bm25",
                        metadata={
                            "chunk_index": row.get('chunk_index', 0),
                            "token_count": row.get('token_count', 0),
                            "word_count": row.get('word_count', 0)
                        }
                    )
                    results.append(search_result)

            return results

        except Exception as e:
            print(f"Full-text search error: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        filter_expr: str | None = None
    ) -> list[SearchResult]:
        """Perform hybrid search combining vector and full-text search.

        Args:
            query: Query text
            limit: Maximum number of results
            filter_expr: Optional filter expression

        Returns:
            List of search results with fused scores
        """
        try:
            # Get results from both search methods
            vector_results = self.vector_search(query, limit * 2, filter_expr)
            fulltext_results = self.fulltext_search(query, limit * 2, filter_expr)

            # Implement Reciprocal Rank Fusion (RRF)
            rrf_scores: dict[str, dict[str, Any]] = {}
            k = 60  # RRF parameter

            # Process vector results
            for rank, result in enumerate(vector_results):
                rrf_score = self.config.vector_weight / (k + rank + 1)
                if result.chunk_id in rrf_scores:
                    rrf_scores[result.chunk_id]['score'] = (
                        rrf_scores[result.chunk_id]['score'] + rrf_score
                    )
                else:
                    rrf_scores[result.chunk_id] = {
                        'result': result,
                        'score': rrf_score
                    }

            # Process full-text results
            for rank, result in enumerate(fulltext_results):
                rrf_score = self.config.fulltext_weight / (k + rank + 1)
                if result.chunk_id in rrf_scores:
                    rrf_scores[result.chunk_id]['score'] = (
                        rrf_scores[result.chunk_id]['score'] + rrf_score
                    )
                else:
                    rrf_scores[result.chunk_id] = {
                        'result': result,
                        'score': rrf_score
                    }

            # Sort by combined score and create final results
            sorted_items = sorted(
                rrf_scores.values(),
                key=lambda x: float(x['score']),
                reverse=True
            )

            results = []
            for item in sorted_items[:limit]:
                result = item['result']
                if isinstance(result, SearchResult):
                    result.score = float(item['score'])
                result.search_method = "hybrid"
                results.append(result)

            return results

        except Exception as e:
            print(f"Hybrid search error: {e}")
            return []

    async def rerank_results(
        self,
        query: str,
        results: list[SearchResult]
    ) -> list[SearchResult]:
        """Rerank search results using LLM.

        Args:
            query: Original query
            results: Search results to rerank

        Returns:
            Reranked search results
        """
        if not self.reranker or len(results) <= 1:
            return results

        try:
            # Prepare context for reranking
            context = f"Query: {query}\n\nDocuments to rank:\n"
            for i, result in enumerate(results):
                context += f"{i+1}. {result.title}\n"
                context += f"   Content: {result.content[:200]}...\n\n"

            # Ask LLM to rerank
            prompt = f"""
            {context}

            Please rank these documents by relevance to the query.
            Return only the numbers (1, 2, 3, etc.) in order of relevance,
            separated by commas. Most relevant first.
            """

            response = await self.reranker.run(prompt)

            # Parse the response
            ranking_str = response.data.strip()
            rankings = [
                int(x.strip())
                for x in ranking_str.split(',')
                if x.strip().isdigit()
            ]

            # Reorder results based on LLM ranking
            reranked: list[SearchResult] = []
            for rank_num in rankings:
                if 1 <= rank_num <= len(results):
                    result = results[rank_num - 1]
                    result.metadata['rerank_position'] = len(reranked) + 1
                    reranked.append(result)

            # Add any results not in the ranking
            for _i, result in enumerate(results):
                if result not in reranked:
                    result.metadata['rerank_position'] = len(reranked) + 1
                    reranked.append(result)

            return reranked

        except Exception as e:
            print(f"Reranking error: {e}")
            return results

    async def search(
        self,
        query: str,
        method: str = "hybrid",
        limit: int | None = None,
        filter_expr: str | None = None
    ) -> list[SearchResult]:
        """Main search interface.

        Args:
            query: Search query
            method: Search method ("vector", "fulltext", "hybrid")
            limit: Maximum number of results (uses config default if None)
            filter_expr: Optional filter expression

        Returns:
            List of search results
        """
        if not query.strip():
            return []

        # Use config defaults if not specified
        if limit is None:
            limit = self.config.final_top_k

        # Preprocess query
        processed_query = self.query_processor.preprocess_query(query)

        # Expand query if enabled
        if self.config.enable_query_expansion:
            self.query_processor.expand_query(processed_query)
            # Use the original query for now, could combine multiple expansions
            search_query = processed_query
        else:
            search_query = processed_query

        # Perform search based on method
        if method == "vector":
            results = self.vector_search(search_query, limit, filter_expr)
        elif method == "fulltext":
            results = self.fulltext_search(search_query, limit, filter_expr)
        elif method == "hybrid":
            results = self.hybrid_search(search_query, limit, filter_expr)
        else:
            raise ValueError(f"Unknown search method: {method}")

        # Apply reranking if enabled
        if self.config.enable_reranking and len(results) > 1:
            # Rerank top results
            rerank_limit = min(self.config.rerank_top_k, len(results))
            top_results = results[:rerank_limit]
            remaining_results = results[rerank_limit:]

            reranked_top = await self.rerank_results(query, top_results)
            results = reranked_top + remaining_results

        return results[:limit]


# Convenience functions
def create_search_layer(
    embedding_pipeline: EmbeddingPipeline,
    config: SearchConfig | None = None
) -> SearchLayer:
    """Create a configured search layer.

    Args:
        embedding_pipeline: Configured embedding pipeline
        config: Search configuration

    Returns:
        Configured search layer
    """
    return SearchLayer(embedding_pipeline, config)


async def search_documents(
    query: str,
    embedding_pipeline: EmbeddingPipeline,
    method: str = "hybrid",
    limit: int = 5,
    config: SearchConfig | None = None
) -> list[SearchResult]:
    """Convenient function to search documents.

    Args:
        query: Search query
        embedding_pipeline: Configured embedding pipeline
        method: Search method
        limit: Maximum number of results
        config: Search configuration

    Returns:
        List of search results
    """
    search_layer = create_search_layer(embedding_pipeline, config)
    return await search_layer.search(query, method, limit)


if __name__ == "__main__":
    from pathlib import Path

    from .embeddings import EmbeddingPipeline

    async def main() -> None:
        """Example usage of the search layer."""

        # Initialize embedding pipeline
        db_path = Path("data/embeddings")
        if not db_path.exists():
            print(f"Vector database not found at {db_path}")
            print("Run the embedding pipeline first to create the database.")
            return

        try:
            pipeline = EmbeddingPipeline(db_path=db_path)

            # Create search layer with different configurations
            configs = {
                "performance": SearchConfig(
                    vector_weight=1.0,
                    fulltext_weight=0.0,
                    enable_reranking=False
                ),
                "quality": SearchConfig(
                    vector_weight=0.5,
                    fulltext_weight=0.5,
                    enable_reranking=True,
                    rerank_top_k=10
                ),
                "balanced": SearchConfig()  # Default config
            }

            # Test queries
            test_queries = [
                "How to deploy an application",
                "Docker configuration",
                "Environment variables setup",
                "Database connection",
                "AppPack CLI commands"
            ]

            for config_name, config in configs.items():
                print(f"\n{'='*50}")
                print(f"Testing with {config_name} configuration")
                print(f"{'='*50}")

                search_layer = create_search_layer(pipeline, config)

                for query in test_queries[:2]:  # Test first 2 queries
                    print(f"\nQuery: '{query}'")
                    print("-" * 40)

                    # Test different search methods
                    for method in ["vector", "fulltext", "hybrid"]:
                        try:
                            results = await search_layer.search(
                                query, method=method, limit=3
                            )

                            print(f"\n{method.capitalize()} search results:")
                            for i, result in enumerate(results):
                                print(
                                f"  {i+1}. {result.title} "
                                f"(score: {result.score:.3f})"
                            )
                                print(f"      {result.content[:100]}...")

                        except Exception as e:
                            print(f"  Error with {method} search: {e}")

        except Exception as e:
            print(f"Error initializing search: {e}")

    # Note: Requires vector database to exist
    print("Search module loaded. Run embedding pipeline first to test search.")
    # asyncio.run(main())
