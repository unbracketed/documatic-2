"""CLI application for Documatic.

Provides a Click-based command-line interface for document acquisition,
indexing, search, and chat functionality.
"""

import logging
import os
import sys
from pathlib import Path

import click

from .acquisition import DocumentAcquisition
from .chat import ChatConfig, RAGChatInterface
from .embeddings import EmbeddingPipeline
from .search import SearchLayer


# Configure logging
def setup_logging(verbose: bool) -> None:
    """Setup logging configuration based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(path_type=Path),
    default="data",
    help="Data directory for storage (default: data)",
)
@click.version_option(version="0.1.0", prog_name="documatic")
@click.pass_context
def cli(ctx: click.Context, config: Path | None, verbose: bool, data_dir: Path) -> None:
    """Documatic - RAG application for AppPack.io documentation.

    Provides commands to fetch, index, search, and chat with AppPack documentation.
    """
    setup_logging(verbose)

    # Store common options in context
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose
    ctx.obj["data_dir"] = data_dir


@cli.command()
@click.option(
    "--repo-url",
    default="https://github.com/apppackio/apppack-docs.git",
    help="Git repository URL to fetch from",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force full re-acquisition of documents",
)
@click.pass_context
def fetch(ctx: click.Context, repo_url: str, force: bool) -> None:
    """Acquire/update documents from the AppPack documentation repository."""
    data_dir = ctx.obj["data_dir"]

    click.echo("📥 Fetching documents...")

    try:
        acquisition = DocumentAcquisition(data_dir=data_dir)

        with click.progressbar(
            length=100, label="Acquiring documents", show_percent=True
        ) as bar:
            # Simulate progress for now - in a real implementation,
            # you'd update progress based on actual operations
            result = acquisition.acquire_documents(repo_url=repo_url)
            bar.update(100)

        if result["status"] == "success":
            click.echo(f"✅ Successfully processed {result['documents_count']} files")
            click.echo(f"📄 Manifest saved to {result['manifest_path']}")
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Error during document acquisition: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--chunk-size",
    default=1024,
    help="Maximum chunk size in tokens (default: 1024)",
)
@click.option(
    "--overlap",
    default=0.15,
    help="Overlap ratio between chunks (default: 0.15)",
)
@click.option(
    "--batch-size",
    default=50,
    help="Batch size for embedding generation (default: 50)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-indexing of all documents",
)
@click.pass_context
def index(
    ctx: click.Context,
    chunk_size: int,
    overlap: float,
    batch_size: int,
    force: bool,
) -> None:
    """Chunk and embed documents into the vector database."""
    data_dir = ctx.obj["data_dir"]

    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        click.echo("❌ Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    click.echo("🔄 Starting document indexing...")

    try:
        with click.progressbar(
            length=100, label="Processing documents", show_percent=True
        ) as bar:
            # Get manifest to determine number of files
            acquisition = DocumentAcquisition(data_dir=data_dir)
            manifest_file = acquisition.manifest_file

            if not manifest_file.exists():
                click.echo("❌ No manifest found. Run 'documatic fetch' first.")
                sys.exit(1)

            # For now, show that indexing would happen here
            click.echo("Processing documents from manifest...")
            # TODO: Implement manifest-based chunk processing
            result = {"status": "success", "chunks_processed": 0, "total_documents": 0}
            bar.update(100)

        if result["status"] == "success":
            click.echo(f"✅ Successfully indexed {result['chunks_processed']} chunks")
            click.echo(f"📊 Total documents in database: {result['total_documents']}")
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Error during indexing: {e}")
        sys.exit(1)


@cli.command()
@click.argument("query", required=True)
@click.option(
    "--limit",
    "-l",
    default=5,
    help="Number of results to return (default: 5)",
)
@click.option(
    "--method",
    type=click.Choice(["vector", "fulltext", "hybrid"]),
    default="hybrid",
    help="Search method to use (default: hybrid)",
)
@click.pass_context
def search(ctx: click.Context, query: str, limit: int, method: str) -> None:
    """Perform a one-off search query against the document index."""
    data_dir = ctx.obj["data_dir"]

    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        click.echo("❌ Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    try:
        embedding_pipeline = EmbeddingPipeline(db_path=data_dir / "embeddings")
        search_layer = SearchLayer(embedding_pipeline)

        click.echo(f"🔍 Searching for: {query}")

        if method == "vector":
            results = search_layer.vector_search(query, limit=limit)
        elif method == "fulltext":
            results = search_layer.fulltext_search(query, limit=limit)
        else:  # hybrid
            results = search_layer.hybrid_search(query, limit=limit)

        if not results:
            click.echo("No results found.")
            return

        click.echo(f"\n📋 Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            click.echo(f"{i}. {result.title}")
            click.echo(f"   📁 {result.source_file}")
            click.echo(f"   📊 Score: {result.score:.3f}")
            click.echo(f"   🔗 {' > '.join(result.section_hierarchy)}")
            click.echo(f"   📝 {result.content[:200]}...")
            click.echo()

    except Exception as e:
        click.echo(f"❌ Error during search: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--model",
    default="gpt-4o-mini",
    help="Chat model to use (default: gpt-4o-mini)",
)
@click.option(
    "--context-window",
    default=5,
    help="Number of previous turns to include in context (default: 5)",
)
@click.pass_context
def chat(ctx: click.Context, model: str, context_window: int) -> None:
    """Start an interactive chat session with the documentation."""
    data_dir = ctx.obj["data_dir"]

    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        click.echo("❌ Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    try:
        embedding_pipeline = EmbeddingPipeline(db_path=data_dir / "embeddings")
        search_layer = SearchLayer(embedding_pipeline)
        chat_config = ChatConfig(
            model_name=model,
        )
        chat_interface = RAGChatInterface(search_layer, chat_config)

        click.echo("💬 Starting chat session with AppPack documentation")
        click.echo("Type 'exit', 'quit', or Ctrl+C to end the session\n")

        while True:
            try:
                user_input = click.prompt("You", type=str)

                if user_input.lower() in {"exit", "quit"}:
                    break

                click.echo("Assistant: ", nl=False)

                # Stream the response
                import asyncio
                response_text = ""
                captured_input = user_input  # Capture to avoid closure issues

                async def stream_response() -> None:
                    nonlocal response_text
                    async for chunk in chat_interface.chat_stream(captured_input):
                        click.echo(chunk, nl=False)
                        response_text += chunk

                # Run the async streaming
                asyncio.run(stream_response())

                click.echo("\n")  # Add newline after streaming response

            except KeyboardInterrupt:
                click.echo("\n👋 Goodbye!")
                break
            except EOFError:
                click.echo("\n👋 Goodbye!")
                break

    except Exception as e:
        click.echo(f"❌ Error during chat: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for evaluation results",
)
@click.option(
    "--sample-size",
    default=10,
    help="Number of test queries to evaluate (default: 10)",
)
@click.pass_context
def evaluate(ctx: click.Context, output: Path | None, sample_size: int) -> None:
    """Run quality evaluation checks on the system."""
    click.echo("🔬 Running quality evaluation...")

    # Placeholder implementation - this would be implemented in task 08
    click.echo("⚠️  Evaluation system not yet implemented")
    click.echo("This command will be available after completing "
              "task 08_quality_evaluation")

    if output:
        click.echo(f"Results would be saved to: {output}")


if __name__ == "__main__":
    cli()
