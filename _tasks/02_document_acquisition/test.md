# Document Acquisition Module - Test Specification

## Test Categories

### 1. Repository Operations
- **Test Git Clone**: Verify successful cloning of AppPack docs repository
  - Assert repository is cloned to correct location
  - Verify .git directory exists
  - Check all markdown files are present
  
- **Test Git Pull**: Verify incremental updates work correctly
  - Test pull on existing repository
  - Verify only changed files are updated
  - Assert commit hash changes after pull
  
- **Test Clone vs Pull Logic**: Verify correct operation selection
  - Test clones when directory doesn't exist
  - Test pulls when directory exists with .git
  - Test handles corrupted/incomplete repositories

### 2. File System Operations
- **Test Directory Creation**: Verify data/raw/ directory structure
  - Assert directory is created if not exists
  - Test permissions are correct
  - Handle existing directory gracefully
  
- **Test File Discovery**: Verify markdown file discovery
  - Find all .md files recursively
  - Ignore non-markdown files
  - Handle symlinks appropriately
  - Test with nested directory structures

### 3. Metadata Extraction
- **Test Frontmatter Parsing**: Verify frontmatter extraction
  - Parse YAML frontmatter correctly
  - Handle missing frontmatter
  - Test malformed YAML
  - Extract all metadata fields (title, description, tags, etc.)
  
- **Test Content Separation**: Verify content parsing
  - Separate frontmatter from content
  - Preserve content formatting
  - Handle edge cases (no frontmatter, empty content)

### 4. Manifest Generation
- **Test Manifest Creation**: Verify manifest file generation
  - Include source file paths
  - Record timestamps (modified, acquired)
  - Calculate and store file hashes (SHA-256)
  - Store in JSON format
  
- **Test Manifest Updates**: Verify incremental manifest updates
  - Add new files to existing manifest
  - Update changed files
  - Remove deleted files
  - Preserve unchanged entries

### 5. Error Handling
- **Test Network Errors**: Verify graceful handling
  - Simulate connection timeout
  - Handle DNS resolution failures
  - Test with invalid repository URL
  - Implement retry logic with backoff
  
- **Test Rate Limiting**: Verify rate limit handling
  - Mock GitHub API rate limit responses
  - Implement exponential backoff
  - Log rate limit encounters
  
- **Test File System Errors**: Verify error handling
  - Handle permission denied
  - Test disk full scenarios
  - Handle corrupted files

### 6. Integration Tests
- **Test End-to-End Workflow**: Full acquisition pipeline
  - Clone repository
  - Parse all documents
  - Generate complete manifest
  - Verify data integrity
  
- **Test Incremental Updates**: Verify update workflow
  - Make changes to test repository
  - Run acquisition again
  - Verify only changed files processed
  - Check manifest reflects changes

## Test Data Requirements

### Mock Data
- Sample markdown files with various frontmatter formats
- Repository with known structure for testing
- Network error simulation responses
- Rate limit response mocks

### Test Fixtures
- Pre-cloned repository state
- Sample manifest files
- Corrupted repository states
- Various frontmatter examples

## Performance Tests
- **Test Large Repository**: Handle repositories with many files
  - Measure cloning time
  - Test memory usage during parsing
  - Verify efficient file processing
  
- **Test Large Files**: Handle large markdown documents
  - Test parsing performance
  - Memory efficiency
  - Hash calculation speed

## Security Tests
- **Test Path Traversal**: Prevent directory traversal attacks
  - Validate all file paths
  - Test with malicious path inputs
  - Ensure files stay within data/raw/
  
- **Test Repository Validation**: Verify repository source
  - Only accept GitHub URLs
  - Validate repository structure
  - Check for malicious content

## Configuration Tests
- **Test Custom Paths**: Allow configurable directories
  - Test with different data directories
  - Handle relative and absolute paths
  - Environment variable support

## Logging and Monitoring Tests
- **Test Logging Output**: Verify appropriate logging
  - Log acquisition progress
  - Record errors with context
  - Test different log levels
  - Structured logging format

## Test Implementation Notes
- Use pytest fixtures for repository setup/teardown
- Mock git operations for faster tests
- Use temporary directories for file system tests
- Implement test data generators for various scenarios
- Ensure tests are idempotent and isolated