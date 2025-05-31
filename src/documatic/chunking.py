"""Document chunking module for Documatic.

Implements markdown-aware chunking that preserves document structure,
headings, code blocks, and other semantic elements for optimal retrieval.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a processed document chunk with metadata."""

    chunk_id: str = Field(description="Unique identifier for the chunk")
    source_file: str = Field(description="Original file path")
    title: str = Field(description="Document title")
    section_hierarchy: list[str] = Field(
        description="Hierarchical list of headings leading to this chunk"
    )
    chunk_index: int = Field(description="Index of chunk within document")
    content: str = Field(description="Chunk content")
    content_type: str = Field(description="Type of content (text, code, list, table)")
    document_type: str = Field(
        description="Document category (tutorial, reference, guide)"
    )
    token_count: int = Field(description="Estimated token count")
    word_count: int = Field(description="Word count")
    start_position: int = Field(description="Character position in original document")
    end_position: int = Field(description="End character position in original document")
    overlap_previous: bool = Field(
        default=False, description="Whether this chunk overlaps with previous"
    )
    frontmatter: dict[str, Any] = Field(
        default_factory=dict, description="Document frontmatter metadata"
    )


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    min_chunk_size: int = 512
    max_chunk_size: int = 1024
    overlap_percentage: float = 0.15  # 15% overlap
    preserve_code_blocks: bool = True
    preserve_lists: bool = True
    preserve_tables: bool = True
    heading_weight: float = 1.5  # Weight headings higher for semantic boundaries


class MarkdownChunker:
    """Handles intelligent chunking of markdown documents."""

    def __init__(self, config: ChunkingConfig | None = None):
        """Initialize the chunker with configuration.

        Args:
            config: Chunking configuration, uses defaults if None
        """
        self.config = config or ChunkingConfig()

        # Patterns for markdown elements
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(
            r'^```(\w+)?\n(.*?)^```$', re.MULTILINE | re.DOTALL
        )
        self.list_pattern = re.compile(r'^(\s*[-*+]|\s*\d+\.)\s+', re.MULTILINE)
        self.table_pattern = re.compile(r'^\|.*\|$', re.MULTILINE)

    def chunk_document(self, file_path: Path | str) -> list[DocumentChunk]:
        """Chunk a markdown document while preserving structure.

        Args:
            file_path: Path to the markdown file

        Returns:
            List of document chunks with metadata
        """
        file_path = Path(file_path)

        # Read and parse document
        with open(file_path, encoding='utf-8') as f:
            raw_content = f.read()

        post = frontmatter.loads(raw_content)
        content = post.content
        metadata = post.metadata if hasattr(post, 'metadata') else {}

        # Extract document information
        title = metadata.get('title', file_path.stem)
        document_type = self._determine_document_type(file_path, metadata)

        # Parse document structure
        structure = self._parse_document_structure(content)

        # Generate chunks
        chunks = self._generate_chunks(
            content=content,
            file_path=str(file_path),
            title=title,
            document_type=document_type,
            structure=structure,
            frontmatter=metadata
        )

        return chunks

    def _determine_document_type(
        self, file_path: Path, metadata: dict[str, Any]
    ) -> str:
        """Determine the document type based on path and metadata.

        Args:
            file_path: Path to the document
            metadata: Document frontmatter metadata

        Returns:
            Document type classification
        """
        # Check metadata first
        doc_type = metadata.get('type')
        if doc_type:
            return str(doc_type)

        # Classify based on path
        path_str = str(file_path).lower()

        if 'tutorial' in path_str:
            return 'tutorial'
        elif 'how-to' in path_str:
            return 'guide'
        elif 'under-the-hood' in path_str:
            return 'reference'
        elif 'index' in path_str:
            return 'index'
        else:
            return 'reference'

    def _parse_document_structure(self, content: str) -> list[dict[str, Any]]:
        """Parse document structure including headings and content blocks.

        Args:
            content: Document content

        Returns:
            List of structure elements with positions and metadata
        """
        elements = []

        # Find all headings
        for match in self.heading_pattern.finditer(content):
            level = len(match.group(1))  # Count # characters
            heading_text = match.group(2).strip()
            start_pos = match.start()

            elements.append({
                'type': 'heading',
                'level': level,
                'text': heading_text,
                'start': start_pos,
                'end': match.end()
            })

        # Find code blocks
        for match in self.code_block_pattern.finditer(content):
            language = match.group(1) or 'text'
            code_content = match.group(2)

            elements.append({
                'type': 'code_block',
                'language': language,
                'content': code_content,
                'start': match.start(),
                'end': match.end()
            })

        # Sort by position
        elements.sort(key=lambda x: int(x['start']))  # type: ignore[arg-type]
        return elements

    def _generate_chunks(
        self,
        content: str,
        file_path: str,
        title: str,
        document_type: str,
        structure: list[dict[str, Any]],
        frontmatter: dict[str, Any]
    ) -> list[DocumentChunk]:
        """Generate chunks from document content and structure.

        Args:
            content: Document content
            file_path: Source file path
            title: Document title
            document_type: Type of document
            structure: Parsed document structure
            frontmatter: Document metadata

        Returns:
            List of document chunks
        """
        chunks = []
        current_hierarchy: list[str] = []
        current_pos = 0
        chunk_index = 0

        # Process structure elements and content between them
        for _i, element in enumerate(structure):
            # Handle content before this element
            if element['start'] > current_pos:
                content_chunk = content[current_pos:element['start']].strip()
                if content_chunk:
                    chunk = self._create_chunk(
                        content=content_chunk,
                        file_path=file_path,
                        title=title,
                        document_type=document_type,
                        hierarchy=current_hierarchy.copy(),
                        chunk_index=chunk_index,
                        start_pos=current_pos,
                        end_pos=element['start'],
                        content_type='text',
                        frontmatter=frontmatter
                    )
                    chunks.append(chunk)
                    chunk_index += 1

            # Handle the element itself
            if element['type'] == 'heading':
                # Update hierarchy
                level = element['level']
                heading_text = element['text']

                # Trim hierarchy to current level
                current_hierarchy = current_hierarchy[:level-1]

                # Add current heading
                if len(current_hierarchy) < level:
                    current_hierarchy.extend([''] * (level - len(current_hierarchy)))
                current_hierarchy[level-1] = heading_text

            elif element['type'] == 'code_block':
                # Create chunk for code block
                chunk = self._create_chunk(
                    content=content[element['start']:element['end']],
                    file_path=file_path,
                    title=title,
                    document_type=document_type,
                    hierarchy=current_hierarchy.copy(),
                    chunk_index=chunk_index,
                    start_pos=element['start'],
                    end_pos=element['end'],
                    content_type='code',
                    frontmatter=frontmatter
                )
                chunks.append(chunk)
                chunk_index += 1

            current_pos = element['end']

        # Handle remaining content
        if current_pos < len(content):
            remaining_content = content[current_pos:].strip()
            if remaining_content:
                chunk = self._create_chunk(
                    content=remaining_content,
                    file_path=file_path,
                    title=title,
                    document_type=document_type,
                    hierarchy=current_hierarchy.copy(),
                    chunk_index=chunk_index,
                    start_pos=current_pos,
                    end_pos=len(content),
                    content_type='text',
                    frontmatter=frontmatter
                )
                chunks.append(chunk)

        # Apply semantic chunking and overlap
        return self._apply_semantic_chunking(chunks)

    def _create_chunk(
        self,
        content: str,
        file_path: str,
        title: str,
        document_type: str,
        hierarchy: list[str],
        chunk_index: int,
        start_pos: int,
        end_pos: int,
        content_type: str,
        frontmatter: dict[str, Any]
    ) -> DocumentChunk:
        """Create a document chunk with metadata.

        Args:
            content: Chunk content
            file_path: Source file path
            title: Document title
            document_type: Document type
            hierarchy: Section hierarchy
            chunk_index: Chunk index
            start_pos: Start position in document
            end_pos: End position in document
            content_type: Type of content
            frontmatter: Document metadata

        Returns:
            Document chunk with complete metadata
        """
        # Generate unique chunk ID
        chunk_id = hashlib.md5(
            f"{file_path}:{chunk_index}:{start_pos}".encode()
        ).hexdigest()[:12]

        # Count tokens (approximate: 1 token ≈ 0.75 words)
        word_count = len(content.split())
        token_count = int(word_count * 1.33)  # Rough approximation

        # Clean hierarchy (remove empty entries)
        clean_hierarchy = [h for h in hierarchy if h.strip()]

        return DocumentChunk(
            chunk_id=chunk_id,
            source_file=file_path,
            title=title,
            section_hierarchy=clean_hierarchy,
            chunk_index=chunk_index,
            content=content,
            content_type=content_type,
            document_type=document_type,
            token_count=token_count,
            word_count=word_count,
            start_position=start_pos,
            end_position=end_pos,
            frontmatter=frontmatter
        )

    def _apply_semantic_chunking(
        self, initial_chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Apply semantic chunking rules to merge/split chunks as needed.

        Args:
            initial_chunks: Initial chunk list

        Returns:
            Optimized chunks with proper sizing and overlap
        """
        final_chunks = []

        for chunk in initial_chunks:
            # If chunk is too large, split it
            if chunk.token_count > self.config.max_chunk_size:
                split_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(split_chunks)
            # If chunk is too small, try to merge with next chunk
            elif (chunk.token_count < self.config.min_chunk_size and
                    chunk.content_type == 'text'):
                # For now, keep small chunks as-is to preserve structure
                # TODO: Implement intelligent merging
                final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)

        # Add overlap between adjacent text chunks
        return self._add_overlap(final_chunks)

    def _split_large_chunk(self, chunk: DocumentChunk) -> list[DocumentChunk]:
        """Split a large chunk into smaller ones.

        Args:
            chunk: Chunk to split

        Returns:
            List of smaller chunks
        """
        # For code blocks, keep as single chunk to preserve integrity
        if chunk.content_type == 'code':
            return [chunk]

        # Split text chunks by sentences/paragraphs
        content = chunk.content
        sentences = re.split(r'(?<=[.!?])\s+', content)

        split_chunks = []
        current_content = ""
        current_word_count = 0
        sub_index = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If adding this sentence would exceed max size, create a chunk
            if (current_word_count + sentence_words >
                int(self.config.max_chunk_size * 0.75)):  # Convert tokens to words

                if current_content:
                    split_chunk = self._create_split_chunk(
                        chunk, current_content, sub_index
                    )
                    split_chunks.append(split_chunk)
                    sub_index += 1
                    current_content = sentence
                    current_word_count = sentence_words
                else:
                    # Single sentence is too long, include it anyway
                    current_content = sentence
                    current_word_count = sentence_words
            else:
                current_content += (" " + sentence if current_content else sentence)
                current_word_count += sentence_words

        # Add remaining content
        if current_content:
            split_chunk = self._create_split_chunk(chunk, current_content, sub_index)
            split_chunks.append(split_chunk)

        return split_chunks if split_chunks else [chunk]

    def _create_split_chunk(
        self, original: DocumentChunk, content: str, sub_index: int
    ) -> DocumentChunk:
        """Create a split chunk from an original chunk.

        Args:
            original: Original chunk
            content: New content
            sub_index: Sub-chunk index

        Returns:
            New chunk with updated content and metadata
        """
        word_count = len(content.split())
        token_count = int(word_count * 1.33)

        return DocumentChunk(
            chunk_id=f"{original.chunk_id}_s{sub_index}",
            source_file=original.source_file,
            title=original.title,
            section_hierarchy=original.section_hierarchy,
            chunk_index=original.chunk_index,
            content=content,
            content_type=original.content_type,
            document_type=original.document_type,
            token_count=token_count,
            word_count=word_count,
            start_position=original.start_position,
            end_position=original.end_position,
            overlap_previous=False,
            frontmatter=original.frontmatter
        )

    def _add_overlap(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Add overlap between adjacent chunks for context preservation.

        Args:
            chunks: List of chunks

        Returns:
            Chunks with overlap added
        """
        if len(chunks) < 2:
            return chunks

        overlapped_chunks = [chunks[0]]  # First chunk has no overlap

        for i in range(1, len(chunks)):
            current = chunks[i]
            previous = chunks[i-1]

            # Only add overlap for text chunks from the same document
            if (current.content_type == 'text' and
                previous.content_type == 'text' and
                current.source_file == previous.source_file):

                # Calculate overlap size
                overlap_words = int(
                    previous.word_count * self.config.overlap_percentage
                )
                if overlap_words > 0:
                    # Extract last N words from previous chunk
                    prev_words = previous.content.split()
                    if len(prev_words) >= overlap_words:
                        overlap_text = " ".join(prev_words[-overlap_words:])
                        overlapped_content = f"{overlap_text} {current.content}"

                        # Update chunk with overlap
                        updated_chunk = DocumentChunk(
                            chunk_id=current.chunk_id,
                            source_file=current.source_file,
                            title=current.title,
                            section_hierarchy=current.section_hierarchy,
                            chunk_index=current.chunk_index,
                            content=overlapped_content,
                            content_type=current.content_type,
                            document_type=current.document_type,
                            token_count=int(len(overlapped_content.split()) * 1.33),
                            word_count=len(overlapped_content.split()),
                            start_position=current.start_position,
                            end_position=current.end_position,
                            overlap_previous=True,
                            frontmatter=current.frontmatter
                        )
                        overlapped_chunks.append(updated_chunk)
                    else:
                        overlapped_chunks.append(current)
                else:
                    overlapped_chunks.append(current)
            else:
                overlapped_chunks.append(current)

        return overlapped_chunks


def chunk_documents_from_manifest(
    manifest_path: Path | str, chunker: MarkdownChunker | None = None
) -> list[DocumentChunk]:
    """Chunk all documents referenced in a manifest file.

    Args:
        manifest_path: Path to the manifest.json file
        chunker: Chunker instance, creates default if None

    Returns:
        List of all document chunks
    """
    import json

    manifest_path = Path(manifest_path)
    chunker = chunker or MarkdownChunker()

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_chunks = []

    # Process each document in manifest
    for doc_info in manifest.get('documents', []):
        file_path = Path(doc_info['absolute_path'])

        # Only process markdown files
        if file_path.suffix.lower() == '.md':
            try:
                chunks = chunker.chunk_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                # Log error but continue processing
                print(f"Error chunking {file_path}: {e}")

    return all_chunks


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    # Test with a single document
    chunker = MarkdownChunker()
    test_file = Path("data/raw/apppack-docs/src/how-to/apps/config-variables.md")

    if test_file.exists():
        chunks = chunker.chunk_document(test_file)
        print(f"Generated {len(chunks)} chunks from {test_file.name}")

        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Type: {chunk.content_type}")
            print(f"  Hierarchy: {' > '.join(chunk.section_hierarchy)}")
            print(f"  Tokens: {chunk.token_count}")
            print(f"  Content preview: {chunk.content[:100]}...")
