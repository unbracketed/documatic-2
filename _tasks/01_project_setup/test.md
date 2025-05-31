# Test Specification: Project Setup and Dependencies

## Overview
This test specification covers the verification of project initialization, structure, configuration, and toolchain setup for the Documatic project.

## Test Categories

### 1. Project Initialization Tests

#### 1.1 UV Package Manager Setup
- **Test**: Verify `uv` command is available and functional
- **Expected**: `uv --version` returns valid version information
- **Validation**: Project initialized with `uv init` creates expected files

#### 1.2 Python Environment
- **Test**: Verify Python version compatibility
- **Expected**: Python 3.11+ is available
- **Validation**: `uv run python --version` executes successfully

### 2. Project Structure Tests

#### 2.1 Directory Structure
- **Test**: Verify all required directories exist
- **Expected**: 
  - `src/documatic/` directory exists
  - `tests/` directory exists
  - `docs/` directory exists
- **Validation**: Path checking for each directory

#### 2.2 Module Structure
- **Test**: Verify documatic is importable as a Python module
- **Expected**: `from documatic import __version__` succeeds
- **Validation**: No import errors when accessing the module

### 3. Configuration File Tests

#### 3.1 pyproject.toml Validation
- **Test**: Verify pyproject.toml exists and is valid
- **Expected**:
  - File exists at project root
  - Contains valid TOML syntax
  - Has required sections: [project], [project.dependencies], [project.optional-dependencies]
- **Validation**: Parse TOML and check structure

#### 3.2 Project Metadata
- **Test**: Verify project metadata is properly configured
- **Expected**:
  - Project name is "documatic"
  - Version follows semantic versioning
  - Description is present
  - Author information is included
- **Validation**: Extract and verify each metadata field

#### 3.3 Core Dependencies
- **Test**: Verify all core dependencies are specified
- **Expected** dependencies present:
  - pydantic-ai
  - lancedb
  - click
  - httpx
- **Validation**: Check each dependency in pyproject.toml

#### 3.4 Development Dependencies
- **Test**: Verify development dependencies are specified
- **Expected** dev dependencies present:
  - pytest
  - pytest-cov
  - ruff
  - mypy
- **Validation**: Check optional-dependencies.dev section

### 4. Dependency Installation Tests

#### 4.1 Core Dependency Installation
- **Test**: Verify core dependencies can be installed
- **Expected**: `uv pip install -e .` completes successfully
- **Validation**: Import each core dependency without errors

#### 4.2 Development Dependency Installation
- **Test**: Verify dev dependencies can be installed
- **Expected**: `uv pip install -e ".[dev]"` completes successfully
- **Validation**: Commands available: pytest, ruff, mypy

### 5. Development Tool Tests

#### 5.1 Pytest Configuration
- **Test**: Verify pytest is properly configured
- **Expected**:
  - `uv run pytest` executes without configuration errors
  - Test discovery works in tests/ directory
- **Validation**: Run pytest with --collect-only

#### 5.2 Ruff Linting
- **Test**: Verify ruff is configured and functional
- **Expected**:
  - `uv run ruff check .` executes
  - ruff.toml or pyproject.toml contains ruff configuration
- **Validation**: Run ruff on sample Python file

#### 5.3 Mypy Type Checking
- **Test**: Verify mypy is configured
- **Expected**:
  - `uv run mypy documatic/` executes
  - mypy.ini or pyproject.toml contains mypy configuration
- **Validation**: Run mypy on documatic module

### 6. Git Configuration Tests

#### 6.1 Gitignore File
- **Test**: Verify .gitignore exists with Python patterns
- **Expected** patterns included:
  - `__pycache__/`
  - `*.pyc`
  - `.pytest_cache/`
  - `.coverage`
  - `.mypy_cache/`
  - `venv/`
  - `.env`
- **Validation**: Check each pattern exists in .gitignore

#### 6.2 Pre-commit Hooks
- **Test**: Verify pre-commit configuration exists
- **Expected**:
  - `.pre-commit-config.yaml` file exists
  - Contains hooks for ruff, mypy, and other tools
- **Validation**: Parse YAML and verify hook configuration

### 7. CLI Entry Point Tests

#### 7.1 Click CLI Setup
- **Test**: Verify CLI entry point is configured
- **Expected**:
  - Entry point defined in pyproject.toml
  - `uv run documatic` or similar command works
- **Validation**: Execute CLI with --help

#### 7.2 Module Execution
- **Test**: Verify module can be run directly
- **Expected**: `uv run python -m documatic` executes
- **Validation**: Check for proper main module setup

## Test Execution Plan

1. **Setup Verification**: Run all initialization and structure tests first
2. **Configuration Tests**: Validate all configuration files
3. **Installation Tests**: Test dependency installation in clean environment
4. **Tool Tests**: Verify each development tool works correctly
5. **Integration**: Ensure all components work together

## Success Criteria

- All directories and files are created as specified
- Dependencies can be installed without conflicts
- Development tools execute without errors
- Project can be imported as a Python module
- CLI entry point is accessible
- Code quality tools are properly configured

## Edge Cases to Test

1. Running setup in directory with existing Python project
2. Missing or incompatible Python version
3. Network issues during dependency installation
4. Permission issues creating directories
5. Conflicting dependency versions

## Automation

These tests should be automated as part of CI/CD:
- Create GitHub Action to verify project setup
- Run tests on multiple Python versions (3.11, 3.12)
- Test on different OS platforms (Linux, macOS, Windows)