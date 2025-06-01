#!/usr/bin/env python3
"""Helper script to properly fix SearchResult instances in test files."""

import re


def fix_search_result_simple():
    """Fix the broken SearchResult instances with a simple replacement."""

    files_to_fix = [
        'tests/unit/test_context_management.py',
        'tests/unit/test_citations.py',
        'tests/unit/test_prompt_templates.py',
        'tests/unit/test_conversation_flow.py',
        'tests/integration/test_chat_integration.py'
    ]

    for file_path in files_to_fix:
        print(f"Fixing {file_path}")

        with open(file_path) as f:
            content = f.read()

        # Fix the broken SearchResult pattern by properly structuring fields
        # Pattern: title="Test Document",\n        section_hierarchy=
        content = re.sub(
            r'title="Test Document",\s*\n\s*section_hierarchy=',
            'title="Test Document",\n                section_hierarchy=',
            content
        )

        # Fix the broken score and search_method placement
        content = re.sub(
            r',\s*\n\s*score=([0-9.]+),\s*\n\s*metadata=([^}]+})\s*,\s*\n\s*search_method="hybrid"\)',
            r',\n                score=\1,\n                search_method="hybrid",\n                metadata=\2)',
            content
        )

        # Fix chunk_index references that shouldn't be there
        content = re.sub(r'chunk_index=\d+,\s*\n\s*', '', content)

        # Fix similarity_score to score
        content = re.sub(r'similarity_score=', 'score=', content)

        # Fix specific broken patterns
        content = re.sub(
            r'SearchResult\(title="Test Document",\s*\n\s*content="([^"]+)",\s*score=([0-9.]+),\s*search_method="hybrid"\)',
            r'SearchResult(\n                content="\1",\n                chunk_id="test_chunk",\n                source_file="test.md",\n                title="Test Document",\n                section_hierarchy=["Test"],\n                content_type="text",\n                document_type="markdown",\n                score=\2,\n                search_method="hybrid",\n                metadata={}\n            )',
            content
        )

        with open(file_path, 'w') as f:
            f.write(content)

        print(f"Fixed {file_path}")

if __name__ == "__main__":
    fix_search_result_simple()
