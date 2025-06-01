"""Batch test data generation utilities."""

from typing import Any


def generate_test_chunks(count: int) -> list[dict[str, Any]]:
    """Generate test chunks with varied characteristics.
    
    Args:
        count: Number of test chunks to generate
        
    Returns:
        List of test chunk dictionaries
    """
    chunks = []
    for i in range(count):
        content_length = (i % 100 + 10)
        content = f"Test content {i} " * content_length
        chunks.append({
            "chunk_id": f"test_chunk_{i}",
            "source_file": f"docs/section_{i % 10}/doc_{i}.md",
            "title": f"Test Document {i}",
            "section_hierarchy": [f"Section {i % 5}", f"Subsection {i % 3}"],
            "chunk_index": i % 20,
            "content": content,
            "content_type": ["text", "code", "list", "table"][i % 4],
            "document_type": "documentation",
            "token_count": content_length * 3,  # Rough estimate
            "word_count": content_length * 2,
            "start_position": i * 1000,  # Mock character positions
            "end_position": i * 1000 + len(content),
            "overlap_previous": i % 3 == 0,
            "frontmatter": {
                "title": f"Test Document {i}",
                "category": f"Category {i % 5}",
                "tags": [f"tag{i % 3}", f"tag{(i+1) % 3}"]
            }
        })
    return chunks


def generate_performance_test_data(size: str = "medium") -> list[dict[str, Any]]:
    """Generate test data for performance testing.
    
    Args:
        size: Test data size ("small", "medium", "large")
        
    Returns:
        List of test chunks for performance testing
    """
    sizes = {
        "small": 100,
        "medium": 1000,
        "large": 10000
    }

    count = sizes.get(size, 1000)
    return generate_test_chunks(count)


def generate_edge_case_chunks() -> list[dict[str, Any]]:
    """Generate chunks for edge case testing.
    
    Returns:
        List of edge case test chunks
    """
    return [
        # Empty content
        {
            "chunk_id": "empty_chunk",
            "source_file": "empty.md",
            "title": "Empty Document",
            "section_hierarchy": [],
            "chunk_index": 0,
            "content": "",
            "content_type": "text",
            "document_type": "documentation",
            "token_count": 0,
            "word_count": 0,
            "start_position": 0,
            "end_position": 0,
            "overlap_previous": False,
            "frontmatter": {}
        },
        # Very long content
        {
            "chunk_id": "long_chunk",
            "source_file": "long.md",
            "title": "Very Long Document",
            "section_hierarchy": ["Long Section"],
            "chunk_index": 0,
            "content": "Very long content " * 2000,  # ~6000 words
            "content_type": "text",
            "document_type": "documentation",
            "token_count": 8000,
            "word_count": 6000,
            "start_position": 0,
            "end_position": 36000,  # Approximate length
            "overlap_previous": False,
            "frontmatter": {"warning": "very_long"}
        },
        # Unicode content
        {
            "chunk_id": "unicode_chunk",
            "source_file": "unicode.md",
            "title": "Unicode Document 中文",
            "section_hierarchy": ["Unicode Section"],
            "chunk_index": 0,
            "content": "Content with unicode: 中文, العربية, Ελληνικά, русский 🚀 📱",
            "content_type": "text",
            "document_type": "documentation",
            "token_count": 20,
            "word_count": 15,
            "start_position": 0,
            "end_position": 65,
            "overlap_previous": False,
            "frontmatter": {"language": "multi"}
        }
    ]
