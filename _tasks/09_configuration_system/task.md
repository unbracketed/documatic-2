# Task: Configuration Management

## Technical Requirements
- Create `documatic/config.py` module
- Use Pydantic settings for configuration validation
- Support configuration sources:
  - Environment variables
  - .env files
  - YAML/TOML config files
  - CLI arguments (override)
- Configure:
  - LLM settings (model, temperature, tokens)
  - Embedding model selection
  - LanceDB connection
  - Chunking parameters
  - Search parameters
- Implement sensible defaults for all settings

## Addresses Requirement
"using configuration, schemas, and embeddings that will work well for general Q&A type questions" - provides flexible configuration for optimization.