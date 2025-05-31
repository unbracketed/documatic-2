# Test Specification: Testing Infrastructure

## Overview
This test specification covers the comprehensive testing infrastructure for the Documatic project, including pytest configuration, test organization, fixtures, mocks, and CI/CD integration.

## Unit Tests

### 1. Test Framework Setup (`test_framework_setup.py`)
- **Test pytest configuration**
  - pytest.ini settings
  - Plugin configuration
  - Test discovery patterns
  - Marker definitions
  
- **Test directory structure**
  - Test organization
  - Module mapping
  - Naming conventions
  - Import paths
  
- **Test utilities**
  - Helper functions
  - Common assertions
  - Test decorators
  - Retry mechanisms

### 2. Fixture Management Tests (`test_fixtures.py`)
- **Test fixture creation**
  - Scope management (function, module, session)
  - Dependency injection
  - Parameterization
  - Cleanup/teardown
  
- **Test data fixtures**
  - Sample documents
  - Mock embeddings
  - Test databases
  - Configuration sets
  
- **Test fixture composition**
  - Fixture dependencies
  - Fixture factories
  - Dynamic fixtures
  - Fixture caching

### 3. Mock Infrastructure Tests (`test_mocks.py`)
- **Test mock creation**
  - External API mocks
  - Database mocks
  - File system mocks
  - Network mocks
  
- **Test mock behavior**
  - Response simulation
  - Error injection
  - Latency simulation
  - State management
  
- **Test mock verification**
  - Call tracking
  - Argument validation
  - Order verification
  - Reset functionality

### 4. Coverage Analysis Tests (`test_coverage.py`)
- **Test coverage configuration**
  - Coverage targets
  - Exclusion patterns
  - Branch coverage
  - Report formats
  
- **Test coverage tracking**
  - Per-module coverage
  - Line coverage
  - Branch coverage
  - Missing coverage identification
  
- **Test coverage reporting**
  - HTML reports
  - Console output
  - CI integration
  - Trend tracking

### 5. Performance Testing Framework (`test_performance_framework.py`)
- **Test benchmark setup**
  - pytest-benchmark config
  - Performance fixtures
  - Baseline establishment
  - Result storage
  
- **Test performance metrics**
  - Execution time
  - Memory usage
  - Resource utilization
  - Scalability tests
  
- **Test regression detection**
  - Performance comparison
  - Threshold violations
  - Trend analysis
  - Alert generation

## Integration Tests

### 1. Module Integration Tests (`test_module_integration.py`)
- **Test module interactions**
  - Data flow validation
  - Interface contracts
  - Error propagation
  - State consistency
  
- **Test pipeline stages**
  - Stage transitions
  - Data transformation
  - Error handling
  - Progress tracking
  
- **Test system coherence**
  - End-to-end flows
  - Component compatibility
  - Version alignment
  - Configuration propagation

### 2. External Service Tests (`test_external_services.py`)
- **Test GitHub integration**
  - Repository cloning
  - File fetching
  - Rate limiting
  - Authentication
  
- **Test LLM API integration**
  - API communication
  - Response handling
  - Error recovery
  - Fallback behavior
  
- **Test database operations**
  - Connection management
  - Transaction handling
  - Concurrent access
  - Data integrity

### 3. CLI Integration Tests (`test_cli_integration.py`)
- **Test command execution**
  - Command parsing
  - Option handling
  - Output formatting
  - Exit codes
  
- **Test command workflows**
  - Multi-command sequences
  - State preservation
  - Error recovery
  - User interaction
  
- **Test system integration**
  - File system operations
  - Environment variables
  - Configuration loading
  - Process management

## Test Organization

### 1. Test Categories
```python
# pytest markers for test categorization
import pytest

# Test speed markers
pytest.mark.unit          # Fast unit tests (<1s)
pytest.mark.integration   # Integration tests (<10s)
pytest.mark.slow         # Slow tests (>10s)
pytest.mark.benchmark    # Performance tests

# Test type markers
pytest.mark.smoke        # Critical path tests
pytest.mark.regression   # Regression tests
pytest.mark.edge_case    # Edge case tests
pytest.mark.flaky       # Known flaky tests

# Environment markers
pytest.mark.requires_api  # Needs external API
pytest.mark.requires_db   # Needs database
pytest.mark.requires_net  # Needs network
pytest.mark.offline      # Can run offline
```

### 2. Test Structure
```
tests/
├── unit/                    # Unit tests
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_search.py
│   ├── test_chat.py
│   └── test_config.py
├── integration/            # Integration tests
│   ├── test_pipeline.py
│   ├── test_cli_commands.py
│   └── test_external_apis.py
├── e2e/                   # End-to-end tests
│   ├── test_full_workflow.py
│   └── test_user_scenarios.py
├── benchmarks/            # Performance tests
│   ├── test_chunking_perf.py
│   ├── test_search_perf.py
│   └── test_embedding_perf.py
├── fixtures/              # Test data and fixtures
│   ├── documents/
│   ├── embeddings/
│   ├── configs/
│   └── responses/
└── conftest.py           # Shared fixtures and config
```

## Mock Requirements

### 1. GitHub API Mock
```python
@pytest.fixture
def mock_github_api():
    """Mock GitHub API for document fetching"""
    with patch('documatic.fetcher.github_client') as mock:
        mock.get_repository.return_value = {
            "name": "apppack-docs",
            "default_branch": "main"
        }
        mock.get_contents.return_value = [
            {"path": "index.md", "type": "file", "content": "# AppPack"},
            {"path": "guides/", "type": "dir"}
        ]
        yield mock
```

### 2. LLM API Mock
```python
@pytest.fixture
def mock_llm_api():
    """Mock LLM API for embeddings and chat"""
    with patch('documatic.llm.client') as mock:
        # Embedding mock
        mock.embed.return_value = [[0.1] * 1536]
        
        # Chat mock
        mock.chat.return_value = {
            "content": "Test response",
            "usage": {"total_tokens": 100}
        }
        yield mock
```

### 3. Database Mock
```python
@pytest.fixture
def mock_lancedb(tmp_path):
    """Mock LanceDB instance"""
    import lancedb
    db = lancedb.connect(tmp_path / "test.db")
    
    # Create test table
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 1536)),
        pa.field("content", pa.string()),
        pa.field("metadata", pa.string())
    ])
    
    table = db.create_table("documents", schema=schema)
    yield table
    
    # Cleanup
    db.close()
```

## Fixture Requirements

### 1. Document Fixtures
```python
@pytest.fixture
def sample_markdown():
    """Sample markdown documents"""
    return {
        "simple": "# Title\n\nThis is a paragraph.",
        "complex": """
# Main Title

## Section 1
Content with **bold** and *italic*.

### Subsection
- List item 1
- List item 2

```python
def example():
    return "code"
```
        """,
        "edge_case": "# " + "A" * 1000  # Very long title
    }
```

### 2. Embedding Fixtures
```python
@pytest.fixture
def sample_embeddings():
    """Pre-computed embeddings for testing"""
    return {
        "doc1": np.random.rand(1536).tolist(),
        "doc2": np.random.rand(1536).tolist(),
        "query": np.random.rand(1536).tolist()
    }
```

### 3. Configuration Fixtures
```python
@pytest.fixture
def test_config(tmp_path):
    """Test configuration"""
    config = {
        "llm": {"model": "gpt-3.5-turbo", "temperature": 0.0},
        "embedding": {"model": "ada-002", "dimension": 1536},
        "database": {"path": str(tmp_path / "test.db")}
    }
    return Config(**config)
```

## CI/CD Configuration

### 1. GitHub Actions Workflow
```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh
    
    - name: Install dependencies
      run: |
        uv venv
        uv pip install -r requirements.txt
        uv pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        uv run pytest -v --cov=documatic --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. Performance Regression Check
```yaml
name: Performance Tests

on:
  schedule:
    - cron: '0 0 * * *'  # Daily
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - name: Run benchmarks
      run: |
        uv run pytest tests/benchmarks/ --benchmark-json=output.json
    
    - name: Compare with baseline
      run: |
        uv run pytest-benchmark compare output.json baseline.json --fail-on-regression
    
    - name: Store results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: output.json
```

## Test Execution Strategy

1. **Local development**: Fast unit tests on save
2. **Pre-commit**: Unit + smoke tests
3. **PR validation**: Full test suite
4. **Nightly**: Performance + integration
5. **Release**: Complete validation

## Quality Metrics

### Coverage Targets
- **Overall**: >80% coverage
- **Core modules**: >90% coverage
- **Critical paths**: 100% coverage
- **Branch coverage**: >75%

### Performance Baselines
- **Unit tests**: <1s each
- **Integration tests**: <10s each
- **E2E tests**: <60s each
- **Total suite**: <10 minutes

### Test Quality Metrics
- **Flaky test rate**: <1%
- **Test maintenance**: <10% churn/month
- **Mock accuracy**: Validated quarterly
- **Fixture reuse**: >50%

## Success Criteria

- All tests pass on supported Python versions
- Coverage targets achieved
- Performance baselines maintained
- CI/CD pipeline reliable
- Test execution fast
- Mocks accurately simulate services
- Fixtures cover common scenarios
- Tests catch real bugs
- Documentation comprehensive
- Easy to add new tests
- Clear test failure messages
- Minimal test maintenance burden