"""Quality evaluation system for Documatic.

Implements comprehensive evaluation metrics including retrieval accuracy,
answer relevance scoring, citation accuracy, and automated benchmark testing
for validating RAG system performance.
"""

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from .chat import RAGChatInterface
from .search import SearchLayer, SearchResult


class EvaluationDataset(BaseModel):
    """Represents a dataset for evaluation."""

    name: str = Field(description="Dataset name")
    description: str = Field(description="Dataset description")
    questions: list["EvaluationQuestion"] = Field(description="List of evaluation questions")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationQuestion(BaseModel):
    """Represents a single evaluation question with expected answer."""

    question_id: str = Field(description="Unique question identifier")
    question: str = Field(description="Question text")
    question_type: Literal["factual", "procedural", "conceptual"] = Field(
        description="Type of question"
    )
    expected_answer: str = Field(description="Expected answer content")
    expected_sources: list[str] = Field(
        description="Expected source files that should be referenced"
    )
    context_keywords: list[str] = Field(
        description="Keywords that should appear in retrieved context"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="Question difficulty level"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Results from evaluating a single question."""

    question_id: str = Field(description="Question identifier")
    question: str = Field(description="Original question")
    generated_answer: str = Field(description="System-generated answer")
    retrieved_sources: list[SearchResult] = Field(description="Retrieved sources")

    # Retrieval metrics
    retrieval_precision: float = Field(description="Precision of retrieved sources")
    retrieval_recall: float = Field(description="Recall of retrieved sources")
    mrr_score: float = Field(description="Mean Reciprocal Rank score")

    # Answer quality metrics
    answer_relevance: float = Field(description="Answer relevance score (0-1)")
    answer_accuracy: float = Field(description="Answer accuracy score (0-1)")
    citation_accuracy: float = Field(description="Citation accuracy score (0-1)")

    # Overall metrics
    overall_score: float = Field(description="Overall evaluation score (0-1)")
    passed: bool = Field(description="Whether evaluation passed threshold")

    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report."""

    report_id: str = Field(description="Unique report identifier")
    dataset_name: str = Field(description="Dataset used for evaluation")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Aggregate metrics
    total_questions: int = Field(description="Total questions evaluated")
    passed_questions: int = Field(description="Questions that passed")
    pass_rate: float = Field(description="Overall pass rate")

    # Average scores
    avg_retrieval_precision: float = Field(description="Average retrieval precision")
    avg_retrieval_recall: float = Field(description="Average retrieval recall")
    avg_mrr_score: float = Field(description="Average MRR score")
    avg_answer_relevance: float = Field(description="Average answer relevance")
    avg_answer_accuracy: float = Field(description="Average answer accuracy")
    avg_citation_accuracy: float = Field(description="Average citation accuracy")
    avg_overall_score: float = Field(description="Average overall score")

    # Per-category breakdown
    results_by_type: dict[str, dict[str, float]] = Field(
        description="Results broken down by question type"
    )
    results_by_difficulty: dict[str, dict[str, float]] = Field(
        description="Results broken down by difficulty"
    )

    # Individual results
    individual_results: list[EvaluationResult] = Field(
        description="Detailed results for each question"
    )

    # Configuration
    pass_threshold: float = Field(description="Pass threshold used")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation system."""

    pass_threshold: float = Field(
        default=0.7, description="Minimum score to pass evaluation"
    )
    retrieval_k: int = Field(
        default=5, description="Number of documents to retrieve for evaluation"
    )
    evaluation_model: str = Field(
        default="gpt-4o-mini", description="Model for answer evaluation"
    )
    max_answer_length: int = Field(
        default=1000, description="Maximum answer length for evaluation"
    )

    # Weights for overall score calculation
    retrieval_weight: float = Field(default=0.3, description="Weight for retrieval metrics")
    answer_quality_weight: float = Field(default=0.5, description="Weight for answer quality")
    citation_weight: float = Field(default=0.2, description="Weight for citation accuracy")


class DatasetGenerator:
    """Generates evaluation datasets from documentation."""

    def __init__(self, search_layer: SearchLayer, model_name: str = "gpt-4o-mini"):
        """Initialize dataset generator.
        
        Args:
            search_layer: Search layer for document retrieval
            model_name: LLM model for question generation
        """
        self.search_layer = search_layer
        self.model = OpenAIModel(model_name)
        self.agent = Agent(self.model)

    async def generate_questions_from_document(
        self,
        document_content: str,
        document_title: str,
        count: int = 5
    ) -> list[EvaluationQuestion]:
        """Generate evaluation questions from a document.
        
        Args:
            document_content: Document content to generate questions from
            document_title: Document title for context
            count: Number of questions to generate
            
        Returns:
            List of generated evaluation questions
        """
        prompt = f"""
        Generate {count} high-quality evaluation questions from the following AppPack.io documentation:
        
        Document: {document_title}
        Content: {document_content[:2000]}...
        
        Create questions that test different aspects:
        - Factual questions: specific facts, settings, or configurations
        - Procedural questions: how-to steps and processes  
        - Conceptual questions: understanding of concepts and relationships
        
        For each question, also provide:
        - The expected answer based on the documentation
        - Key source files that should be referenced
        - Important keywords that should appear in retrieved context
        - Difficulty level (easy/medium/hard)
        
        Return the response as a JSON array with this structure:
        [
          {{
            "question": "How do you deploy a Flask application on AppPack?",
            "question_type": "procedural",
            "expected_answer": "To deploy a Flask application...",
            "expected_sources": ["flask-deployment.md"],
            "context_keywords": ["flask", "deployment", "wsgi"],
            "difficulty": "medium"
          }}
        ]
        """

        try:
            result = await self.agent.run(prompt)
            questions_data = json.loads(result.data)

            questions = []
            for i, q_data in enumerate(questions_data):
                question = EvaluationQuestion(
                    question_id=f"{document_title}_{i+1}",
                    question=q_data["question"],
                    question_type=q_data["question_type"],
                    expected_answer=q_data["expected_answer"],
                    expected_sources=q_data.get("expected_sources", []),
                    context_keywords=q_data.get("context_keywords", []),
                    difficulty=q_data.get("difficulty", "medium")
                )
                questions.append(question)

            return questions

        except Exception as e:
            print(f"Error generating questions: {e}")
            return []

    async def generate_dataset_from_corpus(
        self,
        questions_per_doc: int = 3,
        max_documents: int = 10
    ) -> EvaluationDataset:
        """Generate evaluation dataset from document corpus.
        
        Args:
            questions_per_doc: Questions to generate per document
            max_documents: Maximum documents to sample from
            
        Returns:
            Generated evaluation dataset
        """
        try:
            # Get document corpus statistics
            stats = self.search_layer.pipeline.get_document_stats()

            # Sample documents from the corpus
            df = self.search_layer.pipeline.table.to_pandas()
            unique_sources = df['source_file'].unique()

            if len(unique_sources) > max_documents:
                sampled_sources = random.sample(list(unique_sources), max_documents)
            else:
                sampled_sources = list(unique_sources)

            all_questions = []

            for source_file in sampled_sources:
                # Get representative chunk from this document
                doc_chunks = df[df['source_file'] == source_file]
                if len(doc_chunks) > 0:
                    # Use the first chunk as representative content
                    sample_chunk = doc_chunks.iloc[0]

                    questions = await self.generate_questions_from_document(
                        sample_chunk['content'],
                        sample_chunk['title'],
                        questions_per_doc
                    )
                    all_questions.extend(questions)

            dataset = EvaluationDataset(
                name=f"apppack_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=f"Generated from {len(sampled_sources)} AppPack.io documents",
                questions=all_questions,
                metadata={
                    "corpus_stats": stats,
                    "sampled_sources": sampled_sources,
                    "questions_per_doc": questions_per_doc
                }
            )

            return dataset

        except Exception as e:
            print(f"Error generating dataset: {e}")
            return EvaluationDataset(
                name="empty_dataset",
                description="Error occurred during generation",
                questions=[]
            )


class EvaluationMetrics:
    """Computes evaluation metrics for RAG system performance."""

    def __init__(self, config: EvaluationConfig):
        """Initialize metrics calculator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config

        # Initialize evaluation model
        self.model = OpenAIModel(config.evaluation_model)
        self.agent = Agent(self.model)

    def calculate_retrieval_metrics(
        self,
        retrieved_sources: list[SearchResult],
        expected_sources: list[str],
        context_keywords: list[str]
    ) -> dict[str, float]:
        """Calculate retrieval accuracy metrics.
        
        Args:
            retrieved_sources: Sources retrieved by the system
            expected_sources: Expected source files
            context_keywords: Keywords that should appear in context
            
        Returns:
            Dictionary with retrieval metrics
        """
        # Extract source filenames from retrieved results
        retrieved_files = set()
        for source in retrieved_sources:
            filename = Path(source.source_file).name
            retrieved_files.add(filename)

        # Convert expected sources to just filenames
        expected_files = set()
        for source_path in expected_sources:
            filename = Path(source_path).name
            expected_files.add(filename)

        # Calculate precision and recall
        if retrieved_files:
            precision = len(retrieved_files & expected_files) / len(retrieved_files)
        else:
            precision = 0.0

        if expected_files:
            recall = len(retrieved_files & expected_files) / len(expected_files)
        else:
            recall = 0.0  # No expected sources means 0 recall, not 1

        # Calculate MRR (Mean Reciprocal Rank)
        mrr_score = 0.0
        for rank, source in enumerate(retrieved_sources, 1):
            filename = Path(source.source_file).name
            if filename in expected_files:
                mrr_score = 1.0 / rank
                break

        # Check for context keywords in retrieved content
        all_content = " ".join(source.content.lower() for source in retrieved_sources)
        keyword_hits = sum(1 for keyword in context_keywords
                          if keyword.lower() in all_content)
        keyword_coverage = (keyword_hits / len(context_keywords)
                           if context_keywords else 1.0)

        return {
            "precision": precision,
            "recall": recall,
            "mrr_score": mrr_score,
            "keyword_coverage": keyword_coverage
        }

    async def evaluate_answer_quality(
        self,
        question: str,
        generated_answer: str,
        expected_answer: str,
        retrieved_sources: list[SearchResult]
    ) -> dict[str, Any]:
        """Evaluate answer quality using LLM.
        
        Args:
            question: Original question
            generated_answer: System-generated answer
            expected_answer: Expected answer
            retrieved_sources: Retrieved context sources
            
        Returns:
            Dictionary with answer quality metrics
        """
        # Build context from retrieved sources
        context = "\n".join(f"Source: {source.source_file}\n{source.content[:500]}..."
                           for source in retrieved_sources[:3])

        evaluation_prompt = f"""
        Evaluate the quality of this generated answer for an AppPack.io documentation question.
        
        Question: {question}
        
        Expected Answer: {expected_answer}
        
        Generated Answer: {generated_answer}
        
        Available Context: {context}
        
        Rate the generated answer on these criteria (0.0 to 1.0):
        
        1. Relevance: How well does the answer address the specific question?
        2. Accuracy: How factually correct is the answer based on the context?
        3. Completeness: Does the answer provide sufficient detail?
        4. Clarity: Is the answer clear and well-structured?
        
        Return your evaluation as JSON:
        {{
          "relevance": 0.8,
          "accuracy": 0.9,
          "completeness": 0.7,
          "clarity": 0.8,
          "reasoning": "Brief explanation of the scores"
        }}
        """

        try:
            result = await self.agent.run(evaluation_prompt)
            scores = json.loads(result.data)

            # Calculate overall answer quality
            answer_relevance = scores.get("relevance", 0.0)
            answer_accuracy = scores.get("accuracy", 0.0)

            return {
                "answer_relevance": answer_relevance,
                "answer_accuracy": answer_accuracy,
                "completeness": scores.get("completeness", 0.0),
                "clarity": scores.get("clarity", 0.0),
                "reasoning": scores.get("reasoning", "")
            }

        except Exception as e:
            print(f"Error evaluating answer quality: {e}")
            return {
                "answer_relevance": 0.0,
                "answer_accuracy": 0.0,
                "completeness": 0.0,
                "clarity": 0.0,
                "reasoning": f"Evaluation failed: {str(e)}"
            }

    def evaluate_citations(
        self,
        generated_answer: str,
        retrieved_sources: list[SearchResult]
    ) -> float:
        """Evaluate citation accuracy in the generated answer.
        
        Args:
            generated_answer: Answer text with citations
            retrieved_sources: Available sources for citation
            
        Returns:
            Citation accuracy score (0.0 to 1.0)
        """
        # Extract citation patterns from answer
        citation_pattern = r'\[Source:\s*([^\]]+)\]'
        citations = re.findall(citation_pattern, generated_answer, re.IGNORECASE)

        if not citations:
            # No citations found - check if sources were available
            return 1.0 if not retrieved_sources else 0.0

        # Check if each citation matches an available source
        available_sources = set()
        for source in retrieved_sources:
            filename = Path(source.source_file).name
            available_sources.add(filename.lower())
            available_sources.add(source.source_file.lower())

        valid_citations = 0
        for citation in citations:
            citation_clean = citation.strip().lower()
            if any(citation_clean in available or available in citation_clean
                   for available in available_sources):
                valid_citations += 1

        return valid_citations / len(citations) if citations else 1.0


class EvaluationRunner:
    """Runs comprehensive evaluation of the RAG system."""

    def __init__(
        self,
        chat_interface: RAGChatInterface,
        config: EvaluationConfig | None = None
    ):
        """Initialize evaluation runner.
        
        Args:
            chat_interface: RAG chat interface to evaluate
            config: Evaluation configuration
        """
        self.chat_interface = chat_interface
        self.config = config or EvaluationConfig()
        self.metrics = EvaluationMetrics(self.config)

    async def evaluate_question(self, question: EvaluationQuestion) -> EvaluationResult:
        """Evaluate system performance on a single question.
        
        Args:
            question: Question to evaluate
            
        Returns:
            Evaluation result
        """
        # Get system response
        generated_answer = await self.chat_interface.chat(question.question)

        # Get retrieved sources from the chat context
        last_turn = self.chat_interface.context.turns[-1] if self.chat_interface.context.turns else None
        retrieved_sources = last_turn.sources if last_turn else []

        # Calculate retrieval metrics
        retrieval_metrics = self.metrics.calculate_retrieval_metrics(
            retrieved_sources,
            question.expected_sources,
            question.context_keywords
        )

        # Evaluate answer quality
        answer_metrics = await self.metrics.evaluate_answer_quality(
            question.question,
            generated_answer,
            question.expected_answer,
            retrieved_sources
        )

        # Evaluate citations
        citation_accuracy = self.metrics.evaluate_citations(
            generated_answer,
            retrieved_sources
        )

        # Calculate overall score
        overall_score = (
            self.config.retrieval_weight * (
                retrieval_metrics["precision"] + retrieval_metrics["recall"]
            ) / 2 +
            self.config.answer_quality_weight * (
                answer_metrics["answer_relevance"] + answer_metrics["answer_accuracy"]
            ) / 2 +
            self.config.citation_weight * citation_accuracy
        )

        # Create result
        result = EvaluationResult(
            question_id=question.question_id,
            question=question.question,
            generated_answer=generated_answer,
            retrieved_sources=retrieved_sources,
            retrieval_precision=retrieval_metrics["precision"],
            retrieval_recall=retrieval_metrics["recall"],
            mrr_score=retrieval_metrics["mrr_score"],
            answer_relevance=answer_metrics["answer_relevance"],
            answer_accuracy=answer_metrics["answer_accuracy"],
            citation_accuracy=citation_accuracy,
            overall_score=overall_score,
            passed=overall_score >= self.config.pass_threshold,
            metadata={
                "keyword_coverage": retrieval_metrics["keyword_coverage"],
                "completeness": answer_metrics.get("completeness", 0.0),
                "clarity": answer_metrics.get("clarity", 0.0),
                "evaluation_reasoning": answer_metrics.get("reasoning", "")
            }
        )

        return result

    async def run_evaluation(self, dataset: EvaluationDataset) -> EvaluationReport:
        """Run evaluation on a complete dataset.
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            Comprehensive evaluation report
        """
        print(f"Running evaluation on {len(dataset.questions)} questions...")

        individual_results = []

        for i, question in enumerate(dataset.questions):
            print(f"Evaluating question {i+1}/{len(dataset.questions)}: {question.question[:50]}...")

            try:
                result = await self.evaluate_question(question)
                individual_results.append(result)

                # Clear conversation context between questions to avoid interference
                self.chat_interface.clear_conversation()

            except Exception as e:
                print(f"Error evaluating question {question.question_id}: {e}")
                # Create failed result
                failed_result = EvaluationResult(
                    question_id=question.question_id,
                    question=question.question,
                    generated_answer=f"Evaluation failed: {e}",
                    retrieved_sources=[],
                    retrieval_precision=0.0,
                    retrieval_recall=0.0,
                    mrr_score=0.0,
                    answer_relevance=0.0,
                    answer_accuracy=0.0,
                    citation_accuracy=0.0,
                    overall_score=0.0,
                    passed=False,
                    metadata={"error": str(e)}
                )
                individual_results.append(failed_result)

        # Calculate aggregate metrics
        total_questions = len(individual_results)
        passed_questions = sum(1 for r in individual_results if r.passed)
        pass_rate = passed_questions / total_questions if total_questions > 0 else 0.0

        # Calculate averages
        if total_questions > 0:
            avg_metrics = {
                "retrieval_precision": sum(r.retrieval_precision for r in individual_results) / total_questions,
                "retrieval_recall": sum(r.retrieval_recall for r in individual_results) / total_questions,
                "mrr_score": sum(r.mrr_score for r in individual_results) / total_questions,
                "answer_relevance": sum(r.answer_relevance for r in individual_results) / total_questions,
                "answer_accuracy": sum(r.answer_accuracy for r in individual_results) / total_questions,
                "citation_accuracy": sum(r.citation_accuracy for r in individual_results) / total_questions,
                "overall_score": sum(r.overall_score for r in individual_results) / total_questions,
            }
        else:
            avg_metrics = {
                "retrieval_precision": 0.0,
                "retrieval_recall": 0.0,
                "mrr_score": 0.0,
                "answer_relevance": 0.0,
                "answer_accuracy": 0.0,
                "citation_accuracy": 0.0,
                "overall_score": 0.0,
            }

        # Calculate breakdown by type and difficulty
        results_by_type = self._calculate_breakdown(individual_results, dataset.questions, "question_type")
        results_by_difficulty = self._calculate_breakdown(individual_results, dataset.questions, "difficulty")

        # Create report
        report = EvaluationReport(
            report_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataset_name=dataset.name,
            total_questions=total_questions,
            passed_questions=passed_questions,
            pass_rate=pass_rate,
            avg_retrieval_precision=avg_metrics["retrieval_precision"],
            avg_retrieval_recall=avg_metrics["retrieval_recall"],
            avg_mrr_score=avg_metrics["mrr_score"],
            avg_answer_relevance=avg_metrics["answer_relevance"],
            avg_answer_accuracy=avg_metrics["answer_accuracy"],
            avg_citation_accuracy=avg_metrics["citation_accuracy"],
            avg_overall_score=avg_metrics["overall_score"],
            results_by_type=results_by_type,
            results_by_difficulty=results_by_difficulty,
            individual_results=individual_results,
            pass_threshold=self.config.pass_threshold,
            metadata={
                "dataset_metadata": dataset.metadata,
                "evaluation_config": self.config.model_dump()
            }
        )

        return report

    def _calculate_breakdown(
        self,
        results: list[EvaluationResult],
        questions: list[EvaluationQuestion],
        field: str
    ) -> dict[str, dict[str, float]]:
        """Calculate metrics breakdown by a specific field."""
        breakdown = {}

        # Group by field value
        groups: dict[str, list[EvaluationResult]] = {}
        for result, question in zip(results, questions, strict=False):
            field_value = getattr(question, field)
            if field_value not in groups:
                groups[field_value] = []
            groups[field_value].append(result)

        # Calculate metrics for each group
        for group_name, group_results in groups.items():
            if group_results:
                breakdown[group_name] = {
                    "count": float(len(group_results)),
                    "pass_rate": sum(1 for r in group_results if r.passed) / len(group_results),
                    "avg_overall_score": sum(r.overall_score for r in group_results) / len(group_results),
                    "avg_retrieval_precision": sum(r.retrieval_precision for r in group_results) / len(group_results),
                    "avg_answer_relevance": sum(r.answer_relevance for r in group_results) / len(group_results),
                }

        return breakdown


# Convenience functions
async def generate_evaluation_dataset(
    search_layer: SearchLayer,
    questions_per_doc: int = 3,
    max_documents: int = 10
) -> EvaluationDataset:
    """Generate evaluation dataset from document corpus.
    
    Args:
        search_layer: Search layer for document access
        questions_per_doc: Questions per document
        max_documents: Maximum documents to sample
        
    Returns:
        Generated evaluation dataset
    """
    generator = DatasetGenerator(search_layer)
    return await generator.generate_dataset_from_corpus(questions_per_doc, max_documents)


async def run_quality_evaluation(
    chat_interface: RAGChatInterface,
    dataset: EvaluationDataset,
    config: EvaluationConfig | None = None
) -> EvaluationReport:
    """Run quality evaluation on RAG system.
    
    Args:
        chat_interface: RAG chat interface to evaluate
        dataset: Evaluation dataset
        config: Evaluation configuration
        
    Returns:
        Evaluation report
    """
    runner = EvaluationRunner(chat_interface, config)
    return await runner.run_evaluation(dataset)


def save_evaluation_report(report: EvaluationReport, file_path: Path) -> None:
    """Save evaluation report to file.
    
    Args:
        report: Evaluation report to save
        file_path: Path to save report
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        print(f"Evaluation report saved to {file_path}")
    except Exception as e:
        print(f"Error saving report: {e}")


def load_evaluation_dataset(file_path: Path) -> EvaluationDataset:
    """Load evaluation dataset from file.
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        Loaded evaluation dataset
    """
    try:
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
        return EvaluationDataset.model_validate(data)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return EvaluationDataset(
            name="error_dataset",
            description=f"Failed to load: {e}",
            questions=[]
        )


if __name__ == "__main__":
    from pathlib import Path

    from .chat import create_chat_interface
    from .embeddings import EmbeddingPipeline
    from .search import create_search_layer

    async def main() -> None:
        """Example usage of the evaluation system."""

        # Initialize components
        db_path = Path("data/embeddings")
        if not db_path.exists():
            print(f"Vector database not found at {db_path}")
            print("Run the embedding pipeline first to create the database.")
            return

        try:
            # Setup pipeline, search, and chat
            pipeline = EmbeddingPipeline(db_path=db_path)
            search_layer = create_search_layer(pipeline)
            chat_interface = create_chat_interface(search_layer)

            print("Generating evaluation dataset...")
            dataset = await generate_evaluation_dataset(
                search_layer,
                questions_per_doc=2,
                max_documents=5
            )

            print(f"Generated {len(dataset.questions)} evaluation questions")

            # Run evaluation
            print("Running quality evaluation...")
            config = EvaluationConfig(pass_threshold=0.6)
            report = await run_quality_evaluation(chat_interface, dataset, config)

            # Display results
            print("\nEvaluation Results:")
            print(f"Pass Rate: {report.pass_rate:.1%}")
            print(f"Average Overall Score: {report.avg_overall_score:.3f}")
            print(f"Average Retrieval Precision: {report.avg_retrieval_precision:.3f}")
            print(f"Average Answer Relevance: {report.avg_answer_relevance:.3f}")

            # Save report
            report_path = Path("data") / f"evaluation_report_{report.report_id}.json"
            save_evaluation_report(report, report_path)

        except Exception as e:
            print(f"Error running evaluation: {e}")

    # Note: Requires vector database and API key
    print("Evaluation module loaded. Set OPENAI_API_KEY and run embedding pipeline first.")
    # asyncio.run(main())
