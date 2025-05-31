# Test Specification: Document Chunking Strategy

## Overview
This test specification covers the document chunking module (`documatic/chunking.py`) which implements markdown-aware chunking with metadata preservation.

## Unit Tests

### 1. Markdown Parser Tests (`test_markdown_parser.py`)
- **Test heading extraction**
  - Extract h1-h6 headings correctly
  - Preserve heading hierarchy
  - Handle edge cases (empty headings, special characters)
  
- **Test code block detection**
  - Identify fenced code blocks (```)
  - Preserve language identifiers
  - Handle nested code blocks
  - Test inline code preservation
  
- **Test list parsing**
  - Ordered lists (1., 2., 3.)
  - Unordered lists (-, *, +)
  - Nested lists
  - Mixed list types
  
- **Test table detection**
  - Simple markdown tables
  - Tables with alignment
  - Multi-line cells

### 2. Chunking Algorithm Tests (`test_chunking_algorithm.py`)
- **Test chunk size constraints**
  - Chunks within 512-1024 token range
  - Proper token counting (mock tokenizer)
  - Handle documents smaller than min chunk size
  
- **Test overlap implementation**
  - 10-20% overlap between consecutive chunks
  - Overlap at semantic boundaries (not mid-sentence)
  - First and last chunk handling
  
- **Test boundary preservation**
  - Never split code blocks
  - Respect heading boundaries when possible
  - Keep related content together (e.g., heading + first paragraph)

### 3. Metadata Extraction Tests (`test_metadata_extraction.py`)
- **Test source file tracking**
  - Correct file path preservation
  - Handle relative vs absolute paths
  
- **Test title extraction**
  - Extract from h1 or frontmatter
  - Fallback to filename
  - Handle missing titles
  
- **Test section hierarchy**
  - Build correct hierarchy from headings
  - Handle missing heading levels
  - Preserve hierarchy in chunks
  
- **Test document type classification**
  - Identify tutorials (step-by-step content)
  - Identify reference docs (API/config)
  - Identify guides (conceptual content)
  - Default classification for ambiguous content

### 4. Pydantic Model Tests (`test_chunk_models.py`)
- **Test model validation**
  - Required fields validation
  - Type checking
  - Value constraints (e.g., chunk_index >= 0)
  
- **Test serialization**
  - JSON serialization/deserialization
  - Preserve all metadata fields
  - Handle special characters

## Integration Tests

### 1. End-to-End Chunking Tests (`test_chunking_integration.py`)
- **Test real markdown files**
  - Process actual AppPack documentation samples
  - Verify chunk quality and completeness
  - Ensure no content loss
  
- **Test different document types**
  - Tutorial documents with step-by-step instructions
  - API reference with code examples
  - Configuration guides with YAML/JSON blocks
  
- **Test edge cases**
  - Very large documents (>10k tokens)
  - Very small documents (<100 tokens)
  - Documents with only code blocks
  - Documents with deeply nested structure

### 2. Performance Tests (`test_chunking_performance.py`)
- **Test processing speed**
  - Benchmark chunking speed per MB
  - Memory usage during processing
  - Concurrent document processing
  
- **Test scalability**
  - Process entire documentation set
  - Measure total processing time
  - Identify bottlenecks

## Edge Cases and Error Handling

### 1. Malformed Input Tests (`test_error_handling.py`)
- **Test invalid markdown**
  - Unclosed code blocks
  - Malformed tables
  - Invalid heading syntax
  
- **Test encoding issues**
  - Non-UTF8 files
  - Special unicode characters
  - Mixed encodings
  
- **Test file system errors**
  - Missing files
  - Permission errors
  - Corrupted files

### 2. Boundary Condition Tests (`test_boundary_conditions.py`)
- **Test extreme chunk sizes**
  - Single word documents
  - Documents with one massive paragraph
  - Documents with only headings
  
- **Test special content**
  - Binary content in code blocks
  - Mathematical formulas
  - HTML within markdown

## Mock/Stub Requirements

### 1. Tokenizer Mock
```python
class MockTokenizer:
    def count_tokens(self, text: str) -> int:
        # Simple approximation: ~4 chars per token
        return len(text) // 4
```

### 2. File System Mock
- Mock file reading for unit tests
- Simulate various file conditions
- Control file content precisely

### 3. Document Type Classifier Mock
- Return predictable classifications
- Test classification logic separately

## Test Data Requirements

### 1. Sample Documents (`tests/fixtures/documents/`)
- **tutorial_sample.md**: Step-by-step AppPack setup guide
- **reference_sample.md**: API endpoint documentation
- **guide_sample.md**: Conceptual overview of AppPack
- **edge_case_huge.md**: 15k token document
- **edge_case_tiny.md**: 50 token document
- **edge_case_code_heavy.md**: 80% code blocks
- **edge_case_nested.md**: 6-level heading hierarchy
- **malformed_sample.md**: Various markdown errors

### 2. Expected Output Data (`tests/fixtures/expected/`)
- Pre-calculated expected chunks for each sample
- Expected metadata for validation
- Performance baselines

### 3. Configuration Fixtures
```python
# tests/fixtures/chunking_configs.py
DEFAULT_CONFIG = {
    "min_chunk_size": 512,
    "max_chunk_size": 1024,
    "overlap_percentage": 0.15,
    "preserve_code_blocks": True,
    "preserve_tables": True
}

AGGRESSIVE_CONFIG = {
    "min_chunk_size": 256,
    "max_chunk_size": 512,
    "overlap_percentage": 0.25
}
```

## Test Execution Strategy

1. **Unit tests first**: Run all unit tests to verify individual components
2. **Integration tests**: Test component interactions
3. **Performance tests**: Run separately with timing
4. **Load tests**: Process full documentation set

## Success Criteria

- All unit tests pass with 100% coverage of chunking.py
- Integration tests process sample docs without errors
- No content loss across all test documents
- Performance: <1 second per MB of markdown
- Memory usage: <100MB for typical document
- Chunk quality: Manual review shows logical boundaries