# Test Specification: CLI Application

## Overview
This test specification covers the CLI module (`documatic/cli.py`) which provides the command-line interface for all Documatic functionality using the Click framework.

## Unit Tests

### 1. Command Structure Tests (`test_cli_structure.py`)
- **Test command registration**
  - All commands registered properly
  - Command names and aliases
  - Command groups organization
  - Help text availability
  
- **Test global options**
  - Config file option parsing
  - Verbosity levels (--verbose, -v)
  - Debug mode (--debug)
  - Version display (--version)
  
- **Test option inheritance**
  - Global options available to subcommands
  - Option precedence
  - Default value handling

### 2. Fetch Command Tests (`test_cli_fetch.py`)
- **Test command execution**
  - Basic fetch operation
  - Force update option (--force)
  - Specific file patterns (--pattern)
  - Dry run mode (--dry-run)
  
- **Test progress reporting**
  - Progress bar display
  - File count updates
  - Download speed indication
  - Error reporting
  
- **Test output formats**
  - Normal output
  - Verbose output
  - JSON output (--json)
  - Quiet mode (-q)

### 3. Index Command Tests (`test_cli_index.py`)
- **Test indexing options**
  - Full reindex (--full)
  - Incremental update (default)
  - Specific documents (--documents)
  - Batch size (--batch-size)
  
- **Test embedding configuration**
  - Model selection (--embedding-model)
  - API key handling
  - Dimension validation
  
- **Test progress tracking**
  - Document processing progress
  - Embedding generation progress
  - Index building progress
  - Time estimates

### 4. Chat Command Tests (`test_cli_chat.py`)
- **Test interactive mode**
  - REPL initialization
  - Input handling
  - Output formatting
  - Exit commands
  
- **Test chat options**
  - Model selection (--model)
  - Temperature (--temperature)
  - Max tokens (--max-tokens)
  - System prompt (--system-prompt)
  
- **Test session management**
  - Session persistence (--save-session)
  - Session loading (--load-session)
  - History display
  - Session export

### 5. Search Command Tests (`test_cli_search.py`)
- **Test search execution**
  - Query parsing
  - Result formatting
  - Number of results (-n, --num-results)
  - Search type (--type vector|fulltext|hybrid)
  
- **Test output formats**
  - Human-readable (default)
  - JSON output (--json)
  - CSV output (--csv)
  - Markdown output (--markdown)
  
- **Test filtering options**
  - Document type filter (--doc-type)
  - Date range (--after, --before)
  - Score threshold (--min-score)

### 6. Eval Command Tests (`test_cli_eval.py`)
- **Test evaluation types**
  - Search quality (--eval-search)
  - Chat quality (--eval-chat)
  - Full evaluation (default)
  - Custom test sets (--test-set)
  
- **Test reporting**
  - Summary statistics
  - Detailed reports (--detailed)
  - Export results (--export)
  - Comparison mode (--compare-to)

## Integration Tests

### 1. End-to-End CLI Tests (`test_cli_integration.py`)
- **Test complete workflows**
  - fetch → index → chat
  - fetch → index → search
  - Incremental updates
  - Error recovery
  
- **Test configuration**
  - Config file loading
  - Environment variables
  - Command-line overrides
  - Default behaviors
  
- **Test error handling**
  - Missing API keys
  - Network failures
  - Invalid inputs
  - Graceful exits

### 2. Interactive Mode Tests (`test_cli_interactive.py`)
- **Test REPL features**
  - Command history
  - Tab completion
  - Multi-line input
  - Special commands
  
- **Test user experience**
  - Response formatting
  - Color output
  - Unicode handling
  - Terminal compatibility
  
- **Test keyboard handling**
  - Ctrl+C interruption
  - Ctrl+D EOF
  - Arrow key navigation
  - Input editing

### 3. Progress Display Tests (`test_cli_progress.py`)
- **Test progress bars**
  - Accurate progress tracking
  - Time estimates
  - Speed calculations
  - Nested progress bars
  
- **Test terminal handling**
  - Width detection
  - Non-TTY mode
  - Color support
  - Animation smoothness

## Edge Cases and Error Handling

### 1. Input Validation Tests (`test_cli_validation.py`)
- **Test invalid arguments**
  - Unknown commands
  - Invalid options
  - Type mismatches
  - Out-of-range values
  
- **Test path handling**
  - Invalid file paths
  - Missing directories
  - Permission errors
  - Symbolic links
  
- **Test conflicting options**
  - Mutually exclusive flags
  - Required combinations
  - Dependency validation

### 2. Environment Tests (`test_cli_environment.py`)
- **Test API key handling**
  - Environment variables
  - Config file keys
  - Key validation
  - Missing keys
  
- **Test system compatibility**
  - Different OS platforms
  - Terminal types
  - Locale settings
  - File system types
  
- **Test resource limits**
  - Memory constraints
  - Disk space
  - Network issues
  - Process limits

## Mock/Stub Requirements

### 1. Service Mocks
```python
class MockDocumentFetcher:
    def __init__(self):
        self.fetch_called = False
        self.files_fetched = []
    
    def fetch(self, force=False, pattern=None):
        self.fetch_called = True
        # Simulate file fetching
        for i in range(10):
            yield f"file_{i}.md"
            self.files_fetched.append(f"file_{i}.md")

class MockIndexer:
    def __init__(self):
        self.documents_indexed = 0
    
    def index(self, documents, batch_size=100):
        for i, doc in enumerate(documents):
            self.documents_indexed += 1
            yield {"progress": i + 1, "total": len(documents)}

class MockChatInterface:
    def __init__(self):
        self.messages = []
    
    def chat(self, message, session_id=None):
        self.messages.append(message)
        return f"Response to: {message}"
```

### 2. Click Testing Utilities
```python
from click.testing import CliRunner

def create_test_runner():
    runner = CliRunner()
    return runner

def run_command(command_args, input_text=None, env=None):
    runner = create_test_runner()
    result = runner.invoke(
        cli,
        command_args,
        input=input_text,
        env=env
    )
    return result
```

### 3. Progress Bar Mock
```python
class MockProgressBar:
    def __init__(self, total=100):
        self.total = total
        self.current = 0
        self.updates = []
    
    def update(self, n=1):
        self.current += n
        self.updates.append(self.current)
    
    def close(self):
        pass
```

## Test Data Requirements

### 1. Configuration Files (`tests/fixtures/configs/`)
```yaml
# default_config.yaml
embedding:
  model: "text-embedding-ada-002"
  dimension: 1536

search:
  default_type: "hybrid"
  num_results: 5

chat:
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 500
```

### 2. Command Test Cases (`tests/fixtures/commands/`)
```python
# Command invocation test cases
TEST_COMMANDS = [
    # Basic commands
    ["fetch"],
    ["index"],
    ["chat"],
    ["search", "how to deploy"],
    ["eval"],
    
    # With options
    ["fetch", "--force", "--pattern", "*.md"],
    ["index", "--full", "--batch-size", "50"],
    ["chat", "--model", "gpt-4", "--save-session", "test"],
    ["search", "deployment", "-n", "10", "--json"],
    
    # Global options
    ["--config", "custom.yaml", "fetch"],
    ["-v", "index"],
    ["--debug", "chat"],
]
```

### 3. Interactive Session Scripts (`tests/fixtures/sessions/`)
```python
# Chat session test script
CHAT_SESSION = """
Hello
How do I deploy an app?
What about Python apps?
exit
"""

# Multi-line input test
MULTILINE_SESSION = """
Can you explain how to configure
a multi-stage Docker build
for a Python application?
exit
"""
```

### 4. Expected Outputs (`tests/fixtures/expected_outputs/`)
- Command help texts
- Error messages
- Success messages
- Progress bar formats

## Test Execution Strategy

1. **Unit tests**: Individual command testing
2. **Integration tests**: Complete workflows
3. **Interactive tests**: REPL functionality
4. **Performance tests**: Command execution speed
5. **Compatibility tests**: Cross-platform testing

## Performance Requirements

### Command Execution Times
- **fetch**: Start within 1s, progress updates every 100ms
- **index**: Start within 2s, batch processing feedback
- **chat**: First response <2s, streaming updates
- **search**: Results within 500ms
- **eval**: Progress reporting throughout

### Resource Usage
- **Memory**: <200MB for normal operations
- **CPU**: Efficient progress bar updates
- **I/O**: Non-blocking for user input
- **Network**: Timeout handling

## User Experience Requirements

### Help System
- **Command help**: Clear descriptions and examples
- **Option documentation**: Purpose and defaults
- **Error messages**: Actionable guidance
- **Examples**: Common use cases

### Output Formatting
- **Colors**: Semantic color usage (errors=red, success=green)
- **Tables**: Aligned columns for search results
- **Progress**: Clear indication of long operations
- **Verbosity**: Appropriate detail levels

### Interactive Features
- **Tab completion**: Commands and options
- **History**: Previous commands recall
- **Shortcuts**: Common operations
- **Interruption**: Clean Ctrl+C handling

## Success Criteria

- All commands execute without errors
- Help text is comprehensive and accurate
- Progress bars show accurate information
- Error messages are helpful and actionable
- Configuration system works correctly
- Environment variables are respected
- Interactive mode is responsive
- Output formats are consistent
- Cross-platform compatibility confirmed
- Performance targets met
- No memory leaks in long sessions
- Graceful handling of all error conditions