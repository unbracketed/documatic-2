"""Integration tests for the evaluation system.

Tests end-to-end evaluation functionality with real components
but mocked external dependencies.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.documatic.evaluation import (
    EvaluationConfig,
    EvaluationDataset,
    EvaluationQuestion,
    generate_evaluation_dataset,
    load_evaluation_dataset,
    run_quality_evaluation,
    save_evaluation_report,
)


class TestEvaluationIntegration:
    """Integration tests for evaluation system."""

    @pytest.fixture
    def mock_search_layer(self):
        """Create comprehensive mock search layer."""
        search_layer = MagicMock()

        # Mock pipeline
        pipeline = MagicMock()

        # Mock document statistics
        pipeline.get_document_stats.return_value = {
            "total_documents": 3,
            "source_files": 3
        }

        # Mock pandas dataframe with sample documents
        import pandas as pd
        mock_df = pd.DataFrame({
            'source_file': [
                'docs/deployment.md',
                'docs/configuration.md',
                'docs/troubleshooting.md'
            ],
            'content': [
                'AppPack deployment allows you to deploy applications easily. Use the deploy command to start deployment.',
                'Configure your AppPack application using environment variables and config files.',
                'Common issues and their solutions when using AppPack platform.'
            ],
            'title': [
                'Deployment Guide',
                'Configuration Reference',
                'Troubleshooting'
            ]
        })

        pipeline.table.to_pandas.return_value = mock_df
        search_layer.pipeline = pipeline

        return search_layer

    @pytest.fixture
    def mock_chat_interface(self):
        """Create comprehensive mock chat interface."""
        chat_interface = MagicMock()

        # Mock chat method to return realistic responses
        async def mock_chat(question: str) -> str:
            if "deploy" in question.lower():
                return (
                    "To deploy an application with AppPack, use the deploy command. "
                    "This will package and deploy your application to the platform. "
                    "[Source: deployment.md]"
                )
            elif "config" in question.lower():
                return (
                    "Configuration in AppPack is done through environment variables "
                    "and configuration files. [Source: configuration.md]"
                )
            else:
                return "I don't have information about that topic."

        chat_interface.chat = mock_chat

        # Mock conversation context
        from src.documatic.search import SearchResult
        mock_turn = MagicMock()
        mock_turn.sources = [
            SearchResult(
                chunk_id="1",
                source_file="deployment.md",
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
                source_file="configuration.md",
                title="Configuration",
                section_hierarchy=[],
                content="config content",
                content_type="text",
                document_type="guide",
                score=0.8,
                search_method="vector"
            )
        ]
        chat_interface.context = MagicMock()
        chat_interface.context.turns = [mock_turn]
        chat_interface.clear_conversation = MagicMock()

        return chat_interface

    @pytest.mark.asyncio
    async def test_generate_evaluation_dataset_integration(self, mock_search_layer):
        """Test end-to-end dataset generation."""
        # Mock the DatasetGenerator's agent
        mock_questions_data = [
            {
                "question": "How do you deploy an application?",
                "question_type": "procedural",
                "expected_answer": "Use the deploy command to deploy applications.",
                "expected_sources": ["deployment.md"],
                "context_keywords": ["deploy", "application"],
                "difficulty": "medium"
            },
            {
                "question": "What is AppPack?",
                "question_type": "factual",
                "expected_answer": "AppPack is a deployment platform.",
                "expected_sources": ["deployment.md"],
                "context_keywords": ["apppack", "platform"],
                "difficulty": "easy"
            }
        ]

        with patch('src.documatic.evaluation.DatasetGenerator') as MockGenerator:
            # Create mock instance
            mock_generator = MockGenerator.return_value

            # Mock the generate_questions_from_document method
            async def mock_generate_questions(content, title, count):
                return [
                    EvaluationQuestion(
                        question_id=f"{title}_{i+1}",
                        question=q["question"],
                        question_type=q["question_type"],
                        expected_answer=q["expected_answer"],
                        expected_sources=q["expected_sources"],
                        context_keywords=q["context_keywords"],
                        difficulty=q["difficulty"]
                    )
                    for i, q in enumerate(mock_questions_data[:count])
                ]

            mock_generator.generate_questions_from_document = mock_generate_questions

            # Mock the generate_dataset_from_corpus method
            async def mock_generate_dataset(questions_per_doc, max_documents):
                all_questions = []
                docs = ["deployment", "configuration", "troubleshooting"][:max_documents]

                for doc in docs:
                    questions = await mock_generate_questions("content", doc, questions_per_doc)
                    all_questions.extend(questions)

                return EvaluationDataset(
                    name="test_dataset",
                    description=f"Generated from {len(docs)} documents",
                    questions=all_questions
                )

            mock_generator.generate_dataset_from_corpus = mock_generate_dataset

            # Test the function
            dataset = await generate_evaluation_dataset(
                mock_search_layer,
                questions_per_doc=2,
                max_documents=2
            )

            # Verify results
            assert dataset.name == "test_dataset"
            assert len(dataset.questions) == 4  # 2 docs × 2 questions
            assert all(isinstance(q, EvaluationQuestion) for q in dataset.questions)

    @pytest.mark.asyncio
    async def test_run_quality_evaluation_integration(self, mock_chat_interface):
        """Test end-to-end evaluation execution."""
        # Create test dataset
        questions = [
            EvaluationQuestion(
                question_id="deploy_1",
                question="How do you deploy an application?",
                question_type="procedural",
                expected_answer="Use the deploy command.",
                expected_sources=["deployment.md"],
                context_keywords=["deploy", "application"],
                difficulty="medium"
            ),
            EvaluationQuestion(
                question_id="config_1",
                question="How do you configure AppPack?",
                question_type="procedural",
                expected_answer="Use environment variables and config files.",
                expected_sources=["configuration.md"],
                context_keywords=["config", "environment"],
                difficulty="easy"
            )
        ]

        dataset = EvaluationDataset(
            name="integration_test",
            description="Integration test dataset",
            questions=questions
        )

        # Mock the evaluation metrics
        with patch('src.documatic.evaluation.EvaluationMetrics') as MockMetrics:
            mock_metrics = MockMetrics.return_value

            # Mock retrieval metrics
            mock_metrics.calculate_retrieval_metrics.return_value = {
                "precision": 0.8,
                "recall": 0.9,
                "mrr_score": 1.0,
                "keyword_coverage": 0.8
            }

            # Mock answer quality evaluation
            async def mock_evaluate_answer_quality(*args):
                return {
                    "answer_relevance": 0.9,
                    "answer_accuracy": 0.8,
                    "completeness": 0.7,
                    "clarity": 0.8,
                    "reasoning": "Good quality answer"
                }

            mock_metrics.evaluate_answer_quality = mock_evaluate_answer_quality

            # Mock citation evaluation
            mock_metrics.evaluate_citations.return_value = 0.9

            # Create evaluation config
            config = EvaluationConfig(
                pass_threshold=0.7,
                evaluation_model="gpt-4o-mini"
            )

            # Run evaluation
            report = await run_quality_evaluation(mock_chat_interface, dataset, config)

            # Verify report structure
            assert report.dataset_name == "integration_test"
            assert report.total_questions == 2
            assert report.pass_threshold == 0.7
            assert len(report.individual_results) == 2

            # Check that all questions were evaluated
            question_ids = {result.question_id for result in report.individual_results}
            assert question_ids == {"deploy_1", "config_1"}

            # Check aggregated metrics
            assert 0.0 <= report.pass_rate <= 1.0
            assert 0.0 <= report.avg_overall_score <= 1.0
            assert 0.0 <= report.avg_retrieval_precision <= 1.0

            # Check breakdowns
            assert "procedural" in report.results_by_type
            assert "easy" in report.results_by_difficulty
            assert "medium" in report.results_by_difficulty

    def test_save_and_load_evaluation_report(self):
        """Test saving and loading evaluation reports."""
        # Create a test report
        from src.documatic.evaluation import EvaluationReport, EvaluationResult

        individual_results = [
            EvaluationResult(
                question_id="test_1",
                question="Test question?",
                generated_answer="Test answer",
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
        ]

        report = EvaluationReport(
            report_id="test_report",
            dataset_name="test_dataset",
            total_questions=1,
            passed_questions=1,
            pass_rate=1.0,
            avg_retrieval_precision=0.8,
            avg_retrieval_recall=0.7,
            avg_mrr_score=0.9,
            avg_answer_relevance=0.8,
            avg_answer_accuracy=0.9,
            avg_citation_accuracy=0.7,
            avg_overall_score=0.8,
            results_by_type={"factual": {"pass_rate": 1.0}},
            results_by_difficulty={"easy": {"pass_rate": 1.0}},
            individual_results=individual_results,
            pass_threshold=0.7
        )

        # Test saving and loading
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.json"

            # Save report
            save_evaluation_report(report, report_path)
            assert report_path.exists()

            # Load and verify content
            with open(report_path) as f:
                saved_data = json.load(f)

            assert saved_data["report_id"] == "test_report"
            assert saved_data["pass_rate"] == 1.0
            assert len(saved_data["individual_results"]) == 1

    def test_save_and_load_evaluation_dataset(self):
        """Test saving and loading evaluation datasets."""
        # Create test dataset
        questions = [
            EvaluationQuestion(
                question_id="test_1",
                question="Test question?",
                question_type="factual",
                expected_answer="Test answer",
                expected_sources=["test.md"],
                context_keywords=["test"]
            )
        ]

        dataset = EvaluationDataset(
            name="test_dataset",
            description="Test dataset",
            questions=questions
        )

        # Test saving and loading
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset.json"

            # Save dataset manually
            with open(dataset_path, 'w') as f:
                json.dump(dataset.model_dump(), f, indent=2, default=str)

            # Load dataset
            loaded_dataset = load_evaluation_dataset(dataset_path)

            assert loaded_dataset.name == "test_dataset"
            assert len(loaded_dataset.questions) == 1
            assert loaded_dataset.questions[0].question == "Test question?"

    def test_load_nonexistent_dataset(self):
        """Test loading dataset that doesn't exist."""
        nonexistent_path = Path("/tmp/nonexistent_dataset.json")

        dataset = load_evaluation_dataset(nonexistent_path)

        # Should return error dataset
        assert dataset.name == "error_dataset"
        assert "Failed to load" in dataset.description
        assert len(dataset.questions) == 0

    @pytest.mark.asyncio
    async def test_evaluation_with_different_question_types(self, mock_chat_interface):
        """Test evaluation with different question types and difficulties."""
        # Create diverse test dataset
        questions = [
            EvaluationQuestion(
                question_id="factual_easy",
                question="What is AppPack?",
                question_type="factual",
                expected_answer="AppPack is a platform.",
                expected_sources=["overview.md"],
                context_keywords=["apppack"],
                difficulty="easy"
            ),
            EvaluationQuestion(
                question_id="procedural_medium",
                question="How do you deploy?",
                question_type="procedural",
                expected_answer="Use deploy command.",
                expected_sources=["deployment.md"],
                context_keywords=["deploy"],
                difficulty="medium"
            ),
            EvaluationQuestion(
                question_id="conceptual_hard",
                question="Why use microservices?",
                question_type="conceptual",
                expected_answer="For scalability and modularity.",
                expected_sources=["architecture.md"],
                context_keywords=["microservices"],
                difficulty="hard"
            )
        ]

        dataset = EvaluationDataset(
            name="diverse_test",
            description="Diverse question types",
            questions=questions
        )

        # Mock evaluation components
        with patch('src.documatic.evaluation.EvaluationMetrics') as MockMetrics:
            mock_metrics = MockMetrics.return_value

            # Mock different scores for different difficulties
            def mock_retrieval_metrics(*args):
                return {
                    "precision": 0.8,
                    "recall": 0.8,
                    "mrr_score": 0.8,
                    "keyword_coverage": 0.8
                }

            async def mock_answer_quality(*args):
                question = args[0]
                if "easy" in question:
                    return {"answer_relevance": 0.9, "answer_accuracy": 0.9, "completeness": 0.8, "clarity": 0.9, "reasoning": "Easy"}
                elif "medium" in question:
                    return {"answer_relevance": 0.8, "answer_accuracy": 0.8, "completeness": 0.7, "clarity": 0.8, "reasoning": "Medium"}
                else:
                    return {"answer_relevance": 0.7, "answer_accuracy": 0.7, "completeness": 0.6, "clarity": 0.7, "reasoning": "Hard"}

            mock_metrics.calculate_retrieval_metrics = mock_retrieval_metrics
            mock_metrics.evaluate_answer_quality = mock_answer_quality
            mock_metrics.evaluate_citations.return_value = 0.8

            # Run evaluation
            config = EvaluationConfig(pass_threshold=0.6)
            report = await run_quality_evaluation(mock_chat_interface, dataset, config)

            # Verify breakdowns
            assert len(report.results_by_type) == 3
            assert "factual" in report.results_by_type
            assert "procedural" in report.results_by_type
            assert "conceptual" in report.results_by_type

            assert len(report.results_by_difficulty) == 3
            assert "easy" in report.results_by_difficulty
            assert "medium" in report.results_by_difficulty
            assert "hard" in report.results_by_difficulty

            # Check that different difficulties have different performance
            easy_metrics = report.results_by_difficulty["easy"]
            hard_metrics = report.results_by_difficulty["hard"]

            # Easy questions should generally perform better
            assert easy_metrics["avg_overall_score"] >= hard_metrics["avg_overall_score"]

    @pytest.mark.asyncio
    async def test_evaluation_error_handling(self, mock_search_layer):
        """Test evaluation system error handling."""
        # Test with invalid dataset
        empty_dataset = EvaluationDataset(
            name="empty",
            description="Empty dataset",
            questions=[]
        )

        mock_chat_interface = MagicMock()
        config = EvaluationConfig()

        report = await run_quality_evaluation(mock_chat_interface, empty_dataset, config)

        # Should handle empty dataset gracefully
        assert report.total_questions == 0
        assert report.pass_rate == 0.0  # No questions to evaluate
        assert len(report.individual_results) == 0

    @pytest.mark.asyncio
    async def test_evaluation_performance_tracking(self, mock_chat_interface):
        """Test that evaluation tracks performance correctly."""
        # Create dataset with known expected outcomes
        questions = [
            EvaluationQuestion(
                question_id="high_score",
                question="Easy question",
                question_type="factual",
                expected_answer="Easy answer",
                expected_sources=["test.md"],
                context_keywords=["easy"],
                difficulty="easy"
            ),
            EvaluationQuestion(
                question_id="low_score",
                question="Hard question",
                question_type="conceptual",
                expected_answer="Complex answer",
                expected_sources=["complex.md"],
                context_keywords=["complex"],
                difficulty="hard"
            )
        ]

        dataset = EvaluationDataset(
            name="performance_test",
            description="Performance tracking test",
            questions=questions
        )

        # Mock metrics to return predictable scores
        with patch('src.documatic.evaluation.EvaluationMetrics') as MockMetrics:
            mock_metrics = MockMetrics.return_value

            def mock_retrieval_metrics(retrieved, expected, keywords):
                # Return different scores based on question
                if "easy" in str(expected):
                    return {"precision": 0.9, "recall": 0.9, "mrr_score": 1.0, "keyword_coverage": 1.0}
                else:
                    return {"precision": 0.5, "recall": 0.6, "mrr_score": 0.5, "keyword_coverage": 0.5}

            async def mock_answer_quality(question, *args):
                if "easy" in question.lower():
                    return {"answer_relevance": 0.9, "answer_accuracy": 0.9, "completeness": 0.9, "clarity": 0.9, "reasoning": "High quality"}
                else:
                    return {"answer_relevance": 0.5, "answer_accuracy": 0.6, "completeness": 0.5, "clarity": 0.6, "reasoning": "Lower quality"}

            mock_metrics.calculate_retrieval_metrics = mock_retrieval_metrics
            mock_metrics.evaluate_answer_quality = mock_answer_quality
            mock_metrics.evaluate_citations.return_value = 0.8

            # Run evaluation
            config = EvaluationConfig(pass_threshold=0.7)
            report = await run_quality_evaluation(mock_chat_interface, dataset, config)

            # Verify performance tracking
            assert report.total_questions == 2

            # Find results by question ID
            high_score_result = next(r for r in report.individual_results if r.question_id == "high_score")
            low_score_result = next(r for r in report.individual_results if r.question_id == "low_score")

            # High score question should pass, low score should fail
            assert high_score_result.passed == True
            assert low_score_result.passed == False
            assert high_score_result.overall_score > low_score_result.overall_score

            # Check pass rate
            assert report.passed_questions == 1
            assert report.pass_rate == 0.5
