# Task: Document Chunking Strategy

## Technical Requirements
- Create `documatic/chunking.py` module
- Implement markdown-aware chunking that preserves:
  - Document boundaries
  - Heading hierarchy (h1-h6)
  - Code blocks as atomic units
  - Lists and tables
- Use semantic chunking with overlap for context preservation
- Target chunk size: 512-1024 tokens with 10-20% overlap
- Extract and preserve metadata per chunk:
  - source_file, title, section_hierarchy, chunk_index
  - document_type (tutorial, reference, guide)
- Output chunks in Pydantic models for validation

## Addresses Requirement
"Use a good chunking strategy to capture document boundaries, headings, and structure" - ensures high-quality retrieval by preserving document semantics.