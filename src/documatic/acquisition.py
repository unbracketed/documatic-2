"""Document acquisition module for Documatic.

Handles cloning/pulling AppPack documentation from GitHub and processing
markdown files with metadata extraction.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import frontmatter  # type: ignore[import-untyped]
from git import Repo

logger = logging.getLogger(__name__)


class DocumentAcquisition:
    """Handles acquisition and processing of AppPack documentation."""

    def __init__(self, data_dir: Path = Path("data")):
        """Initialize the document acquisition system.

        Args:
            data_dir: Base directory for storing data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.repo_dir = self.raw_dir / "apppack-docs"
        self.manifest_file = self.raw_dir / "manifest.json"

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def acquire_documents(
        self, repo_url: str = "https://github.com/apppackio/apppack-docs.git"
    ) -> dict[str, Any]:
        """Acquire documents from the AppPack documentation repository.

        Args:
            repo_url: URL of the git repository to clone/pull

        Returns:
            Dictionary with acquisition summary
        """
        logger.info(f"Acquiring documents from {repo_url}")

        try:
            # Clone or pull repository
            repo_info = self._clone_or_pull_repo(repo_url)

            # Process markdown files
            documents = self._process_markdown_files()

            # Create/update manifest
            self._create_manifest(repo_info, documents)

            logger.info(f"Successfully acquired {len(documents)} documents")
            return {
                "status": "success",
                "documents_count": len(documents),
                "repo_info": repo_info,
                "manifest_path": str(self.manifest_file)
            }

        except Exception as e:
            logger.error(f"Failed to acquire documents: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _clone_or_pull_repo(self, repo_url: str) -> dict[str, Any]:
        """Clone repository if it doesn't exist, otherwise pull latest changes.

        Args:
            repo_url: URL of the repository

        Returns:
            Dictionary with repository information
        """
        if self.repo_dir.exists():
            logger.info("Repository exists, pulling latest changes")
            repo = Repo(self.repo_dir)
            origin = repo.remotes.origin
            origin.pull()
            action = "pulled"
        else:
            logger.info("Cloning repository")
            repo = Repo.clone_from(repo_url, self.repo_dir)
            action = "cloned"

        # Get repository info
        latest_commit = repo.head.commit
        return {
            "url": repo_url,
            "action": action,
            "commit_hash": latest_commit.hexsha,
            "commit_message": latest_commit.message.strip(),
            "commit_date": latest_commit.committed_datetime.isoformat(),
            "timestamp": time.time()
        }

    def _process_markdown_files(self) -> list[dict[str, Any]]:
        """Process all markdown files in the repository.

        Returns:
            List of document dictionaries with metadata
        """
        documents = []

        # Find all markdown files
        md_files = list(self.repo_dir.rglob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files")

        for md_file in md_files:
            try:
                doc_info = self._process_single_file(md_file)
                if doc_info:
                    documents.append(doc_info)
            except Exception as e:
                logger.warning(f"Failed to process {md_file}: {e}")

        return documents

    def _process_single_file(self, file_path: Path) -> dict[str, Any] | None:
        """Process a single markdown file.

        Args:
            file_path: Path to the markdown file

        Returns:
            Document information dictionary or None if processing fails
        """
        try:
            # Read file content
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            # Parse frontmatter
            post = frontmatter.loads(content)

            # Calculate file hash
            file_hash = hashlib.md5(content.encode()).hexdigest()

            # Get relative path from repo root
            relative_path = file_path.relative_to(self.repo_dir)

            # Extract metadata
            metadata = post.metadata if hasattr(post, 'metadata') else {}

            return {
                "file_path": str(relative_path),
                "absolute_path": str(file_path),
                "size": file_path.stat().st_size,
                "hash": file_hash,
                "modified_time": file_path.stat().st_mtime,
                "frontmatter": metadata,
                "title": metadata.get("title", relative_path.stem),
                "content_preview": (
                    post.content[:200] + "..."
                    if len(post.content) > 200
                    else post.content
                ),
                "word_count": len(post.content.split()),
                "processed_time": time.time()
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def _create_manifest(
        self,
        repo_info: dict[str, Any],
        documents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Create or update the manifest file.

        Args:
            repo_info: Repository information
            documents: List of processed documents

        Returns:
            Manifest dictionary
        """
        manifest = {
            "acquisition_time": time.time(),
            "repo_info": repo_info,
            "documents": documents,
            "summary": {
                "total_documents": len(documents),
                "total_size": sum(doc["size"] for doc in documents),
                "total_words": sum(doc["word_count"] for doc in documents)
            }
        }

        # Save manifest to file
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info(f"Manifest saved to {self.manifest_file}")
        return manifest

    def get_manifest(self) -> dict[str, Any] | None:
        """Load and return the current manifest.

        Returns:
            Manifest dictionary or None if file doesn't exist
        """
        if self.manifest_file.exists():
            with open(self.manifest_file) as f:
                data: dict[str, Any] = json.load(f)
                return data
        return None

    def is_update_needed(self, check_interval: int = 3600) -> bool:
        """Check if an update is needed based on the last acquisition time.

        Args:
            check_interval: Minimum seconds between checks (default: 1 hour)

        Returns:
            True if update is needed
        """
        manifest = self.get_manifest()
        if not manifest:
            return True

        last_update: float = manifest.get("acquisition_time", 0)
        return (time.time() - last_update) > check_interval


def acquire_apppack_docs(data_dir: Path | None = None) -> dict[str, Any]:
    """Convenience function to acquire AppPack documentation.

    Args:
        data_dir: Directory to store data (defaults to ./data)

    Returns:
        Acquisition result dictionary
    """
    if data_dir is None:
        data_dir = Path("data")

    acquisition = DocumentAcquisition(data_dir)
    return acquisition.acquire_documents()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    result = acquire_apppack_docs()
    print(f"Acquisition result: {result}")

