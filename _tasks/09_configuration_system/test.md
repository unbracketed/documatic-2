# Test Specification: Configuration Management

## Overview
This test specification covers the configuration module (`documatic/config.py`) which provides centralized configuration management using Pydantic settings with multiple source support and validation.

## Unit Tests

### 1. Pydantic Model Tests (`test_config_models.py`)
- **Test model definition**
  - Field types and validation
  - Default values
  - Required vs optional fields
  - Nested configuration objects
  
- **Test value validation**
  - Type checking
  - Range constraints
  - Enum values
  - Custom validators
  
- **Test serialization**
  - JSON export/import
  - YAML compatibility
  - Environment variable naming
  - Secrets masking

### 2. Environment Variable Tests (`test_env_config.py`)
- **Test env var loading**
  - Simple value types
  - Complex types (lists, dicts)
  - Nested object mapping
  - Prefix handling (DOCUMATIC_)
  
- **Test type conversion**
  - String to int/float
  - String to boolean
  - JSON string parsing
  - List parsing (comma-separated)
  
- **Test precedence**
  - Env vars override defaults
  - Explicit values override env
  - Case sensitivity handling

### 3. File Configuration Tests (`test_file_config.py`)
- **Test .env file loading**
  - Basic key-value pairs
  - Comments and empty lines
  - Quote handling
  - Variable expansion
  
- **Test YAML loading**
  - Nested structures
  - Lists and mappings
  - Anchors and references
  - Multiple documents
  
- **Test TOML loading**
  - Table syntax
  - Array syntax
  - Inline tables
  - Type preservation

### 4. CLI Override Tests (`test_cli_overrides.py`)
- **Test argument mapping**
  - CLI args to config fields
  - Nested field access (dot notation)
  - List/dict arguments
  - Type conversion
  
- **Test precedence order**
  - CLI overrides all sources
  - Partial overrides
  - Deep merge behavior
  - Conflict resolution

### 5. Configuration Sections Tests (`test_config_sections.py`)
- **Test LLM configuration**
  - Model selection validation
  - Temperature bounds (0-2)
  - Token limits
  - API key presence
  
- **Test embedding configuration**
  - Model compatibility
  - Dimension validation
  - Batch size limits
  - API configuration
  
- **Test search configuration**
  - Strategy validation
  - Weight constraints (sum to 1)
  - K value bounds
  - Index settings

## Integration Tests

### 1. Multi-Source Loading Tests (`test_config_integration.py`)
- **Test source combination**
  - Default + env + file + CLI
  - Proper precedence order
  - Deep merging behavior
  - Source tracking
  
- **Test real-world scenarios**
  - Development setup
  - Production deployment
  - Testing configuration
  - Local overrides
  
- **Test configuration reload**
  - Hot reload capability
  - File watching
  - Validation on reload
  - Error recovery

### 2. Application Integration Tests (`test_config_app_integration.py`)
- **Test module initialization**
  - Config injection
  - Lazy loading
  - Singleton pattern
  - Thread safety
  
- **Test configuration usage**
  - LLM client setup
  - Database connections
  - API configurations
  - Feature flags
  
- **Test configuration updates**
  - Runtime updates
  - Propagation to modules
  - Cache invalidation
  - State consistency

### 3. Validation Integration Tests (`test_config_validation_integration.py`)
- **Test cross-field validation**
  - Dependent fields
  - Mutual exclusivity
  - Conditional requirements
  - Business rules
  
- **Test configuration profiles**
  - Profile selection
  - Profile inheritance
  - Profile overrides
  - Default profiles

## Edge Cases and Error Handling

### 1. Invalid Configuration Tests (`test_config_errors.py`)
- **Test type errors**
  - Wrong type values
  - Unparseable strings
  - Invalid JSON/YAML
  - Type coercion failures
  
- **Test validation errors**
  - Out of range values
  - Missing required fields
  - Invalid combinations
  - Constraint violations
  
- **Test file errors**
  - Missing files
  - Permission errors
  - Corrupted files
  - Encoding issues

### 2. Environment Edge Cases (`test_config_edge_cases.py`)
- **Test special values**
  - Empty strings
  - Null/None handling
  - Unicode in config
  - Very long values
  
- **Test system limits**
  - Environment size limits
  - File path limits
  - Memory constraints
  - Circular references
  
- **Test security concerns**
  - Secret exposure
  - Path traversal
  - Command injection
  - Sensitive defaults

## Mock/Stub Requirements

### 1. Environment Mock
```python
class MockEnvironment:
    def __init__(self, env_vars: dict = None):
        self.env_vars = env_vars or {}
        self._original = {}
    
    def __enter__(self):
        self._original = os.environ.copy()
        os.environ.clear()
        os.environ.update(self.env_vars)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.environ.clear()
        os.environ.update(self._original)
```

### 2. File System Mock
```python
class MockConfigFile:
    def __init__(self, path: str, content: str):
        self.path = path
        self.content = content
    
    @contextmanager
    def create(self):
        # Create temporary file with content
        with tempfile.NamedTemporaryFile(mode='w', suffix=Path(self.path).suffix, delete=False) as f:
            f.write(self.content)
            temp_path = f.name
        
        try:
            yield temp_path
        finally:
            os.unlink(temp_path)
```

### 3. Configuration Factory
```python
class ConfigFactory:
    @staticmethod
    def create_test_config(**overrides):
        """Create config with test defaults and overrides"""
        defaults = {
            "llm": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 500
            },
            "embedding": {
                "model": "text-embedding-ada-002",
                "dimension": 1536
            },
            "search": {
                "strategy": "hybrid",
                "vector_weight": 0.7,
                "fulltext_weight": 0.3
            }
        }
        # Deep merge overrides
        return Config(**deep_merge(defaults, overrides))
```

## Test Data Requirements

### 1. Configuration Files (`tests/fixtures/configs/`)

**.env.test**
```bash
# Test environment file
DOCUMATIC_LLM_MODEL=gpt-4
DOCUMATIC_LLM_TEMPERATURE=0.5
DOCUMATIC_EMBEDDING_MODEL=text-embedding-ada-002
DOCUMATIC_DATABASE_PATH=/tmp/test.db
DOCUMATIC_API_KEY=test-key-123
```

**config.yaml**
```yaml
llm:
  model: gpt-3.5-turbo
  temperature: 0.7
  max_tokens: 1000
  api_key: ${OPENAI_API_KEY}

embedding:
  model: text-embedding-ada-002
  dimension: 1536
  batch_size: 100

database:
  path: ./data/lancedb
  table_name: documents

chunking:
  min_size: 512
  max_size: 1024
  overlap: 0.15

search:
  strategy: hybrid
  vector_weight: 0.7
  fulltext_weight: 0.3
  rerank_top_k: 20
  final_top_k: 5
```

**config.toml**
```toml
[llm]
model = "gpt-3.5-turbo"
temperature = 0.7
max_tokens = 1000

[embedding]
model = "text-embedding-ada-002"
dimension = 1536

[search]
strategy = "hybrid"
weights = { vector = 0.7, fulltext = 0.3 }
```

### 2. Invalid Configurations (`tests/fixtures/invalid_configs/`)
- Missing required fields
- Type mismatches
- Out of range values
- Circular references
- Malformed syntax

### 3. Test Profiles (`tests/fixtures/profiles/`)
```python
# Development profile
DEV_PROFILE = {
    "debug": True,
    "llm": {"model": "gpt-3.5-turbo", "temperature": 0.0},
    "database": {"path": "./dev_data"}
}

# Production profile
PROD_PROFILE = {
    "debug": False,
    "llm": {"model": "gpt-4", "temperature": 0.3},
    "database": {"path": "/var/lib/documatic/data"}
}

# Test profile
TEST_PROFILE = {
    "debug": True,
    "llm": {"model": "mock", "temperature": 0.5},
    "database": {"path": ":memory:"}
}
```

### 4. CLI Override Examples
```python
CLI_OVERRIDE_TESTS = [
    # Simple overrides
    (["--llm.model", "gpt-4"], {"llm": {"model": "gpt-4"}}),
    
    # Nested overrides
    (["--search.weights.vector", "0.8"], {"search": {"weights": {"vector": 0.8}}}),
    
    # List overrides
    (["--chunking.sizes", "256,512,1024"], {"chunking": {"sizes": [256, 512, 1024]}}),
    
    # Boolean flags
    (["--debug"], {"debug": True}),
    (["--no-debug"], {"debug": False}),
]
```

## Test Execution Strategy

1. **Unit tests**: Individual component testing
2. **Integration tests**: Multi-source configuration
3. **Validation tests**: Error handling and constraints
4. **Performance tests**: Loading speed and memory
5. **Security tests**: Secret handling and injection

## Performance Requirements

### Loading Performance
- **Initial load**: <100ms for all sources
- **File parsing**: <50ms per file
- **Validation**: <10ms for full config
- **Hot reload**: <200ms with validation

### Memory Usage
- **Base footprint**: <10MB
- **With caching**: <50MB
- **Large configs**: <100MB
- **No memory leaks on reload

## Security Requirements

### Secret Management
- **API keys**: Never logged or printed
- **Passwords**: Masked in serialization
- **Env isolation**: No cross-contamination
- **File permissions**: Respect umask

### Input Validation
- **Path validation**: No directory traversal
- **Command injection**: Safe parsing
- **Size limits**: Prevent DoS
- **Type safety**: No eval() usage

## Success Criteria

- All unit tests pass with 100% coverage
- Configuration loads from all sources correctly
- Precedence order is maintained
- Validation catches all invalid configs
- Performance meets requirements
- No security vulnerabilities
- Hot reload works reliably
- Error messages are helpful
- Documentation is comprehensive
- Integration is seamless
- Defaults are sensible
- Extension is straightforward