"""Unit tests for the evaluation module.

Tests dataset generation, metrics calculation, and evaluation reporting
functionality with comprehensive mock scenarios.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.models.openai import OpenAIModel

from src.documatic.evaluation import (
    DatasetGenerator,
    EvaluationConfig,
    EvaluationDataset,
    EvaluationMetrics,
    EvaluationQuestion,
    EvaluationResult,
    EvaluationRunner,
)
from src.documatic.search import SearchResult


class TestEvaluationConfig:
    """Test evaluation configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        assert config.pass_threshold == 0.7
        assert config.retrieval_k == 5
        assert config.evaluation_model == "gpt-4o-mini"
        assert config.retrieval_weight == 0.3
        assert config.answer_quality_weight == 0.5
        assert config.citation_weight == 0.2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvaluationConfig(
            pass_threshold=0.8,
            retrieval_k=10,
            evaluation_model="gpt-4",
            retrieval_weight=0.4,
            answer_quality_weight=0.4,
            citation_weight=0.2
        )
        assert config.pass_threshold == 0.8
        assert config.retrieval_k == 10
        assert config.evaluation_model == "gpt-4"
        assert config.retrieval_weight == 0.4


class TestEvaluationQuestion:
    """Test evaluation question model."""

    def test_question_creation(self):
        """Test creating evaluation questions."""
        question = EvaluationQuestion(
            question_id="test_1",
            question="How do you deploy an app?",
            question_type="procedural",
            expected_answer="To deploy an app, use the deploy command.",
            expected_sources=["deployment.md"],
            context_keywords=["deploy", "app", "command"]
        )

        assert question.question_id == "test_1"
        assert question.question_type == "procedural"
        assert question.difficulty == "medium"  # default
        assert "deploy" in question.context_keywords

    def test_question_validation(self):
        """Test question type validation."""
        # Valid question types
        for q_type in ["factual", "procedural", "conceptual"]:
            question = EvaluationQuestion(
                question_id="test",
                question="Test?",
                question_type=q_type,
                expected_answer="Test answer",
                expected_sources=[],
                context_keywords=[]
            )
            assert question.question_type == q_type


class TestEvaluationDataset:
    """Test evaluation dataset model."""

    def test_dataset_creation(self):
        """Test creating evaluation dataset."""
        questions = [
            EvaluationQuestion(
                question_id="q1",
                question="What is AppPack?",
                question_type="factual",
                expected_answer="AppPack is a deployment platform.",
                expected_sources=["overview.md"],
                context_keywords=["apppack", "platform"]
            )
        ]

        dataset = EvaluationDataset(
            name="test_dataset",
            description="Test questions",
            questions=questions
        )

        assert dataset.name == "test_dataset"
        assert len(dataset.questions) == 1
        assert isinstance(dataset.created_at, datetime)

    def test_empty_dataset(self):
        """Test empty dataset creation."""
        dataset = EvaluationDataset(
            name="empty",
            description="No questions",
            questions=[]
        )

        assert len(dataset.questions) == 0


class TestDatasetGenerator:
    """Test dataset generation functionality."""

    @pytest.fixture
    def mock_search_layer(self):
        """Create mock search layer."""
        search_layer = MagicMock()
        search_layer.pipeline = MagicMock()
        return search_layer

    @pytest.fixture
    def dataset_generator(self, mock_search_layer):
        """Create dataset generator with mocked dependencies."""
        with patch.object(OpenAIModel, '__init__', return_value=None):
            generator = DatasetGenerator(mock_search_layer)
            generator.agent = AsyncMock()
            return generator

    @pytest.mark.asyncio
    async def test_generate_questions_from_document(self, dataset_generator):
        """Test generating questions from document content."""
        # Mock agent response
        mock_response = MagicMock()
        mock_response.data = json.dumps([
            {
                "question": "How do you deploy a Flask app?",
                "question_type": "procedural",
                "expected_answer": "Use the deploy command with Flask configuration.",
                "expected_sources": ["flask.md"],
                "context_keywords": ["flask", "deploy"],
                "difficulty": "medium"
            }
        ])
        dataset_generator.agent.run.return_value = mock_response

        questions = await dataset_generator.generate_questions_from_document(
            "Flask deployment documentation content...",
            "Flask Deployment",
            count=1
        )

        assert len(questions) == 1
        question = questions[0]
        assert question.question == "How do you deploy a Flask app?"
        assert question.question_type == "procedural"
        assert "flask" in question.context_keywords

    @pytest.mark.asyncio
    async def test_generate_questions_error_handling(self, dataset_generator):
        """Test error handling in question generation."""
        # Mock agent to raise exception
        dataset_generator.agent.run.side_effect = Exception("API Error")

        questions = await dataset_generator.generate_questions_from_document(
            "Test content",
            "Test Doc",
            count=1
        )

        assert len(questions) == 0

    @pytest.mark.asyncio
    async def test_generate_dataset_from_corpus(self, dataset_generator):
        """Test generating dataset from document corpus."""
        # Mock corpus data
        import pandas as pd
        mock_df = pd.DataFrame({
            'source_file': ['doc1.md', 'doc2.md'],
            'content': ['Content 1', 'Content 2'],
            'title': ['Doc 1', 'Doc 2']
        })

        dataset_generator.search_layer.pipeline.get_document_stats.return_value = {
            "total_documents": 2
        }
        dataset_generator.search_layer.pipeline.table.to_pandas.return_value = mock_df

        # Mock question generation
        mock_response = MagicMock()
        mock_response.data = json.dumps([
            {
                "question": "Test question?",
                "question_type": "factual",
                "expected_answer": "Test answer",
                "expected_sources": ["doc1.md"],
                "context_keywords": ["test"],
                "difficulty": "easy"
            }
        ])
        dataset_generator.agent.run.return_value = mock_response

        dataset = await dataset_generator.generate_dataset_from_corpus(
            questions_per_doc=1,
            max_documents=2
        )

        assert dataset.name.startswith("apppack_eval_")
        assert len(dataset.questions) == 2  # 1 question per 2 documents


class TestEvaluationMetrics:
    """Test evaluation metrics calculation."""

    @pytest.fixture
    def evaluation_config(self):
        """Create evaluation configuration."""
        return EvaluationConfig()

    @pytest.fixture
    def evaluation_metrics(self, evaluation_config):
        """Create evaluation metrics calculator."""
        with patch.object(OpenAIModel, '__init__', return_value=None):
            metrics = EvaluationMetrics(evaluation_config)
            metrics.agent = AsyncMock()
            return metrics

    def test_calculate_retrieval_metrics(self, evaluation_metrics):
        """Test retrieval metrics calculation."""
        # Create test data
        retrieved_sources = [
            SearchResult(
                chunk_id="1",
                source_file="docs/deployment.md",
                title="Deployment",
                section_hierarchy=["Deployment", "Getting Started"],
                content="How to deploy applications with AppPack",
                content_type="text",
                document_type="guide",
                score=0.9,
                search_method="vector"
            ),
            SearchResult(
                chunk_id="2",
                source_file="docs/config.md",
                title="Configuration",
                section_hierarchy=["Configuration"],
                content="Configuration settings for AppPack",
                content_type="text",
                document_type="reference",
                score=0.8,
                search_method="vector"
            )
        ]

        expected_sources = ["deployment.md", "other.md"]
        context_keywords = ["deploy", "configuration", "missing"]

        metrics = evaluation_metrics.calculate_retrieval_metrics(
            retrieved_sources,
            expected_sources,
            context_keywords
        )

        # Check precision: 1 out of 2 retrieved sources is expected
        assert metrics["precision"] == 0.5

        # Check recall: 1 out of 2 expected sources is retrieved
        assert metrics["recall"] == 0.5

        # Check MRR: first relevant result is at rank 1
        assert metrics["mrr_score"] == 1.0

        # Check keyword coverage: 2 out of 3 keywords found
        assert metrics["keyword_coverage"] == pytest.approx(2/3, rel=1e-2)

    def test_retrieval_metrics_no_results(self, evaluation_metrics):
        """Test retrieval metrics with no results."""
        metrics = evaluation_metrics.calculate_retrieval_metrics(
            [],  # no retrieved sources
            ["expected.md"],
            ["keyword"]
        )

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["mrr_score"] == 0.0
        assert metrics["keyword_coverage"] == 0.0

    def test_retrieval_metrics_no_expected(self, evaluation_metrics):
        """Test retrieval metrics with no expected sources."""
        retrieved_sources = [
            SearchResult(
                chunk_id="1",
                source_file="docs/test.md",
                title="Test",
                section_hierarchy=[],
                content="test content",
                content_type="text",
                document_type="guide",
                score=0.9,
                search_method="vector"
            )
        ]

        metrics = evaluation_metrics.calculate_retrieval_metrics(
            retrieved_sources,
            [],  # no expected sources
            []   # no expected keywords
        )

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0  # recall is 0 when no expected sources
        assert metrics["keyword_coverage"] == 1.0  # coverage is 1 when no expected keywords

    @pytest.mark.asyncio
    async def test_evaluate_answer_quality(self, evaluation_metrics):
        """Test answer quality evaluation."""
        # Mock agent response
        mock_response = MagicMock()
        mock_response.data = json.dumps({
            "relevance": 0.8,
            "accuracy": 0.9,
            "completeness": 0.7,
            "clarity": 0.8,
            "reasoning": "Answer is accurate and relevant."
        })
        evaluation_metrics.agent.run.return_value = mock_response

        retrieved_sources = [
            SearchResult(
                chunk_id="1",
                source_file="test.md",
                title="Test",
                section_hierarchy=[],
                content="Test content for evaluation",
                content_type="text",
                document_type="guide",
                score=0.9,
                search_method="vector"
            )
        ]

        metrics = await evaluation_metrics.evaluate_answer_quality(
            "How do you test?",
            "You test by running tests.",
            "To test, you should run the test command.",
            retrieved_sources
        )

        assert metrics["answer_relevance"] == 0.8
        assert metrics["answer_accuracy"] == 0.9
        assert metrics["completeness"] == 0.7
        assert metrics["clarity"] == 0.8
        assert "accurate" in metrics["reasoning"]

    @pytest.mark.asyncio
    async def test_evaluate_answer_quality_error(self, evaluation_metrics):
        """Test answer quality evaluation with error."""
        # Mock agent to raise exception
        evaluation_metrics.agent.run.side_effect = Exception("API Error")

        metrics = await evaluation_metrics.evaluate_answer_quality(
            "Test question?",
            "Test answer",
            "Expected answer",
            []
        )

        assert metrics["answer_relevance"] == 0.0
        assert metrics["answer_accuracy"] == 0.0
        assert "Evaluation failed" in metrics["reasoning"]

    def test_evaluate_citations(self, evaluation_metrics):
        """Test citation accuracy evaluation."""
        retrieved_sources = [
            SearchResult(
                chunk_id="1",
                source_file="docs/deployment.md",
                title="Deployment",
                section_hierarchy=[],
                content="deployment content",
                content_type="text",
                document_type="guide",
                score=0.9,
                search_method="vector"
            ),
            SearchResult(
                chunk_id="2",
                source_file="docs/config.md",
                title="Configuration",
                section_hierarchy=[],
                content="config content",
                content_type="text",
                document_type="guide",
                score=0.8,
                search_method="vector"
            )
        ]

        # Test with correct citations
        answer_with_citations = (
            "To deploy, see [Source: deployment.md]. "
            "For configuration, check [Source: config.md]."
        )

        accuracy = evaluation_metrics.evaluate_citations(
            answer_with_citations,
            retrieved_sources
        )
        assert accuracy == 1.0

        # Test with partial citations
        answer_partial = "To deploy, see [Source: deployment.md]. Also check [Source: missing.md]."
        accuracy = evaluation_metrics.evaluate_citations(answer_partial, retrieved_sources)
        assert accuracy == 0.5

        # Test with no citations
        answer_no_citations = "To deploy, follow the instructions."
        accuracy = evaluation_metrics.evaluate_citations(answer_no_citations, retrieved_sources)
        assert accuracy == 0.0  # Sources available but not cited

        # Test with no sources and no citations
        accuracy = evaluation_metrics.evaluate_citations(answer_no_citations, [])
        assert accuracy == 1.0  # No sources available, no citations expected


class TestEvaluationRunner:
    """Test evaluation runner functionality."""

    @pytest.fixture
    def mock_chat_interface(self):
        """Create mock chat interface."""
        chat_interface = MagicMock()
        chat_interface.chat = AsyncMock()
        chat_interface.context = MagicMock()
        chat_interface.clear_conversation = MagicMock()
        return chat_interface

    @pytest.fixture
    def evaluation_runner(self, mock_chat_interface):
        """Create evaluation runner with mocked dependencies."""
        config = EvaluationConfig(pass_threshold=0.7)
        runner = EvaluationRunner(mock_chat_interface, config)

        # Mock the metrics calculator
        runner.metrics = MagicMock()
        return runner

    @pytest.mark.asyncio
    async def test_evaluate_question(self, evaluation_runner, mock_chat_interface):
        """Test evaluating a single question."""
        # Setup test data
        question = EvaluationQuestion(
            question_id="test_1",
            question="How do you deploy?",
            question_type="procedural",
            expected_answer="Use the deploy command.",
            expected_sources=["deploy.md"],
            context_keywords=["deploy"]
        )

        # Mock chat response
        mock_chat_interface.chat.return_value = "To deploy, use the deploy command. [Source: deploy.md]"

        # Mock conversation context
        mock_turn = MagicMock()
        mock_turn.sources = [
            SearchResult(
                chunk_id="1",
                source_file="deploy.md",
                title="Deployment",
                section_hierarchy=[],
                content="deployment instructions",
                content_type="text",
                document_type="guide",
                score=0.9,
                search_method="vector"
            )
        ]
        mock_chat_interface.context.turns = [mock_turn]

        # Mock metrics calculations
        evaluation_runner.metrics.calculate_retrieval_metrics.return_value = {
            "precision": 1.0,
            "recall": 1.0,
            "mrr_score": 1.0,
            "keyword_coverage": 1.0
        }

        evaluation_runner.metrics.evaluate_answer_quality = AsyncMock(return_value={
            "answer_relevance": 0.9,
            "answer_accuracy": 0.9,
            "completeness": 0.8,
            "clarity": 0.8,
            "reasoning": "Good answer"
        })

        evaluation_runner.metrics.evaluate_citations.return_value = 1.0

        # Run evaluation
        result = await evaluation_runner.evaluate_question(question)

        # Check results
        assert result.question_id == "test_1"
        assert result.passed == True  # Should pass with high scores
        assert result.overall_score > 0.7
        assert result.retrieval_precision == 1.0
        assert result.answer_relevance == 0.9

    @pytest.mark.asyncio
    async def test_run_evaluation(self, evaluation_runner, mock_chat_interface):
        """Test running evaluation on a dataset."""
        # Create test dataset
        questions = [
            EvaluationQuestion(
                question_id="q1",
                question="Question 1?",
                question_type="factual",
                expected_answer="Answer 1",
                expected_sources=["doc1.md"],
                context_keywords=["test"],
                difficulty="easy"
            ),
            EvaluationQuestion(
                question_id="q2",
                question="Question 2?",
                question_type="procedural",
                expected_answer="Answer 2",
                expected_sources=["doc2.md"],
                context_keywords=["test"],
                difficulty="hard"
            )
        ]

        dataset = EvaluationDataset(
            name="test_dataset",
            description="Test questions",
            questions=questions
        )

        # Mock question evaluation
        async def mock_evaluate_question(question):
            return EvaluationResult(
                question_id=question.question_id,
                question=question.question,
                generated_answer="Generated answer",
                retrieved_sources=[],
                retrieval_precision=0.8,
                retrieval_recall=0.7,
                mrr_score=0.9,
                answer_relevance=0.8,
                answer_accuracy=0.9,
                citation_accuracy=0.7,
                overall_score=0.8,
                passed=True
            )

        evaluation_runner.evaluate_question = mock_evaluate_question

        # Run evaluation
        report = await evaluation_runner.run_evaluation(dataset)

        # Check report
        assert report.dataset_name == "test_dataset"
        assert report.total_questions == 2
        assert report.passed_questions == 2
        assert report.pass_rate == 1.0
        assert len(report.individual_results) == 2

        # Check breakdowns exist
        assert "factual" in report.results_by_type
        assert "procedural" in report.results_by_type
        assert "easy" in report.results_by_difficulty
        assert "hard" in report.results_by_difficulty

    @pytest.mark.asyncio
    async def test_run_evaluation_with_failures(self, evaluation_runner):
        """Test evaluation with some failing questions."""
        # Create test dataset
        question = EvaluationQuestion(
            question_id="fail_q",
            question="Failing question?",
            question_type="factual",
            expected_answer="Answer",
            expected_sources=["doc.md"],
            context_keywords=["test"]
        )

        dataset = EvaluationDataset(
            name="fail_dataset",
            description="Failing test",
            questions=[question]
        )

        # Mock evaluation to raise exception
        async def mock_failing_evaluate_question(question):
            raise Exception("Evaluation failed")

        evaluation_runner.evaluate_question = mock_failing_evaluate_question

        # Run evaluation
        report = await evaluation_runner.run_evaluation(dataset)

        # Check that failure was handled
        assert report.total_questions == 1
        assert report.passed_questions == 0
        assert report.pass_rate == 0.0

        result = report.individual_results[0]
        assert result.passed == False
        assert "Evaluation failed" in result.generated_answer
