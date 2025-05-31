# Test Specification: Quality Evaluation System

## Overview
This test specification covers the evaluation module (`documatic/evaluation.py`) which implements automated quality assessment for the RAG system through synthetic Q&A generation and metric calculation.

## Unit Tests

### 1. Q&A Extraction Tests (`test_qa_extraction.py`)
- **Test documentation parsing**
  - Extract factual statements
  - Identify procedural steps
  - Find conceptual explanations
  - Detect code examples
  
- **Test question generation**
  - Factual questions from statements
  - How-to questions from procedures
  - Why/What questions from concepts
  - Code-related questions
  
- **Test answer extraction**
  - Direct answer identification
  - Multi-sentence answers
  - Code snippet answers
  - Table/list answers

### 2. Synthetic Question Generation Tests (`test_synthetic_generation.py`)
- **Test LLM integration**
  - Prompt template validation
  - Response parsing
  - Error handling
  - Rate limiting
  
- **Test question diversity**
  - Question type distribution
  - Complexity levels
  - Topic coverage
  - Language variety
  
- **Test question quality**
  - Grammatical correctness
  - Answerability
  - Specificity
  - Relevance to content

### 3. Retrieval Metric Tests (`test_retrieval_metrics.py`)
- **Test MRR calculation**
  - Single query MRR
  - Batch MRR averaging
  - Edge cases (no results)
  - Tie handling
  
- **Test Recall@K**
  - Different K values
  - Multiple relevant docs
  - Partial matches
  - Score normalization
  
- **Test precision metrics**
  - Precision@K
  - F1 scores
  - Confusion matrix
  - ROC curves

### 4. Answer Evaluation Tests (`test_answer_evaluation.py`)
- **Test relevance scoring**
  - Semantic similarity
  - Keyword matching
  - Length appropriateness
  - Completeness checking
  
- **Test factual accuracy**
  - Fact extraction
  - Contradiction detection
  - Hallucination identification
  - Source verification
  
- **Test citation validation**
  - Citation presence
  - Citation accuracy
  - Source relevance
  - Coverage assessment

### 5. Benchmark Management Tests (`test_benchmark_management.py`)
- **Test dataset creation**
  - Question-answer pairing
  - Metadata assignment
  - Version control
  - Dataset splitting
  
- **Test baseline establishment**
  - Initial benchmarks
  - Performance thresholds
  - Statistical significance
  - Confidence intervals
  
- **Test regression detection**
  - Performance comparison
  - Degradation alerts
  - Improvement tracking
  - Trend analysis

## Integration Tests

### 1. End-to-End Evaluation Tests (`test_evaluation_integration.py`)
- **Test complete evaluation pipeline**
  - Dataset generation
  - System evaluation
  - Report creation
  - Pass/fail determination
  
- **Test different question types**
  - Factual Q&A evaluation
  - Procedural Q&A evaluation
  - Troubleshooting Q&A
  - Mixed question sets
  
- **Test evaluation consistency**
  - Reproducible results
  - Stable metrics
  - Version compatibility

### 2. Report Generation Tests (`test_report_generation.py`)
- **Test report formats**
  - HTML reports
  - JSON data export
  - CSV metrics
  - Markdown summaries
  
- **Test visualizations**
  - Metric charts
  - Confusion matrices
  - Distribution plots
  - Trend graphs
  
- **Test report content**
  - Executive summary
  - Detailed metrics
  - Failed examples
  - Recommendations

### 3. Regression Testing Tests (`test_regression_system.py`)
- **Test change detection**
  - Code changes impact
  - Data changes impact
  - Model changes impact
  - Configuration changes
  
- **Test historical tracking**
  - Performance history
  - Metric evolution
  - Change correlation
  - Root cause analysis

## Edge Cases and Error Handling

### 1. Dataset Edge Cases (`test_dataset_edge_cases.py`)
- **Test sparse documentation**
  - Few extractable Q&As
  - Limited content variety
  - Missing sections
  - Incomplete docs
  
- **Test problematic content**
  - Ambiguous statements
  - Contradictory information
  - Technical jargon
  - Format variations
  
- **Test generation failures**
  - LLM unavailable
  - Invalid responses
  - Timeout handling
  - Partial generation

### 2. Evaluation Error Tests (`test_evaluation_errors.py`)
- **Test system failures**
  - Search system down
  - Chat system errors
  - Partial results
  - Timeout scenarios
  
- **Test metric edge cases**
  - Division by zero
  - Empty result sets
  - Infinite scores
  - NaN handling
  
- **Test recovery mechanisms**
  - Partial evaluation
  - Error reporting
  - Graceful degradation
  - Manual intervention

## Mock/Stub Requirements

### 1. LLM Mock for Question Generation
```python
class MockQuestionGenerator:
    def __init__(self):
        self.question_templates = {
            "factual": [
                "What is {topic}?",
                "Which {feature} does {product} support?",
                "When should you use {concept}?"
            ],
            "procedural": [
                "How do you {action}?",
                "What are the steps to {task}?",
                "How can I {goal}?"
            ],
            "conceptual": [
                "Why is {concept} important?",
                "What's the difference between {a} and {b}?",
                "When would you choose {option}?"
            ]
        }
    
    def generate_questions(self, content: str, question_type: str, count: int):
        # Generate deterministic questions based on content
        questions = []
        # Implementation...
        return questions
```

### 2. RAG System Mock
```python
class MockRAGSystem:
    def __init__(self):
        self.retrieval_results = {}
        self.chat_responses = {}
    
    def search(self, query: str, k: int = 5):
        # Return mock search results
        return self.retrieval_results.get(query, [])
    
    def chat(self, question: str):
        # Return mock chat response
        return self.chat_responses.get(
            question,
            {"answer": "Default response", "citations": []}
        )
```

### 3. Metric Calculator Mock
```python
class MockMetricCalculator:
    def __init__(self):
        self.scores = {
            "mrr": 0.85,
            "recall@5": 0.92,
            "precision@5": 0.88,
            "f1": 0.90
        }
    
    def calculate_metrics(self, predictions, ground_truth):
        # Return mock metrics with slight variations
        import random
        return {
            metric: score + random.uniform(-0.05, 0.05)
            for metric, score in self.scores.items()
        }
```

## Test Data Requirements

### 1. Sample Documentation (`tests/fixtures/eval_docs/`)
```markdown
# AppPack Deployment Guide

## Overview
AppPack is a platform that simplifies application deployment...

## Getting Started
1. Install the AppPack CLI
2. Configure your application
3. Deploy to the cloud

## Configuration
Applications are configured using apppack.yml files...
```

### 2. Expected Q&A Pairs (`tests/fixtures/expected_qa/`)
```json
{
  "factual": [
    {
      "question": "What is AppPack?",
      "answer": "AppPack is a platform that simplifies application deployment",
      "source": "deployment_guide.md#overview"
    }
  ],
  "procedural": [
    {
      "question": "How do you deploy an application with AppPack?",
      "answer": "1. Install the AppPack CLI\n2. Configure your application\n3. Deploy to the cloud",
      "source": "deployment_guide.md#getting-started"
    }
  ]
}
```

### 3. Evaluation Configurations (`tests/fixtures/eval_configs/`)
```python
# Evaluation thresholds and settings
EVAL_CONFIG = {
    "metrics": {
        "mrr": {"threshold": 0.8, "weight": 0.3},
        "recall@5": {"threshold": 0.85, "weight": 0.3},
        "answer_relevance": {"threshold": 0.8, "weight": 0.2},
        "citation_accuracy": {"threshold": 0.9, "weight": 0.2}
    },
    "question_generation": {
        "types": ["factual", "procedural", "conceptual"],
        "count_per_type": 20,
        "complexity_levels": ["basic", "intermediate", "advanced"]
    },
    "pass_criteria": {
        "overall_threshold": 0.85,
        "required_metrics": ["mrr", "recall@5"],
        "max_regressions": 2
    }
}
```

### 4. Historical Baselines (`tests/fixtures/baselines/`)
```json
{
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:00:00Z",
  "metrics": {
    "mrr": 0.87,
    "recall@5": 0.91,
    "precision@5": 0.85,
    "answer_relevance": 0.83,
    "citation_accuracy": 0.92
  },
  "question_performance": {
    "factual": {"success_rate": 0.90},
    "procedural": {"success_rate": 0.85},
    "conceptual": {"success_rate": 0.80}
  }
}
```

## Test Execution Strategy

1. **Unit testing**: Component validation
2. **Dataset generation**: Create test Q&A pairs
3. **Metric validation**: Verify calculations
4. **Integration testing**: Full evaluation runs
5. **Regression testing**: Compare with baselines

## Quality Metrics

### Evaluation System Metrics
- **Dataset quality**: Question answerability rate
- **Metric reliability**: Consistency across runs
- **Coverage**: Documentation coverage percentage
- **Discrimination**: Ability to detect regressions

### Performance Requirements
- **Dataset generation**: <5 minutes for 100 Q&As
- **Evaluation run**: <10 minutes full suite
- **Report generation**: <30 seconds
- **Memory usage**: <1GB for evaluation

### Report Quality
- **Clarity**: Easy to understand results
- **Actionability**: Clear improvement suggestions
- **Completeness**: All metrics included
- **Traceability**: Failed examples included

## Success Criteria

- Unit tests achieve >95% code coverage
- Dataset generation produces diverse questions
- Metrics accurately reflect system performance
- Reports clearly indicate pass/fail status
- Regression detection catches degradations
- Evaluation is reproducible
- Performance targets are met
- Edge cases handled gracefully
- Integration with CI/CD possible
- Historical tracking functional
- Visualization are informative
- System helps improve RAG quality