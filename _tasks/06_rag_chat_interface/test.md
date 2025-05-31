# Test Specification: RAG Chat Interface

## Overview
This test specification covers the chat module (`documatic/chat.py`) which implements the RAG-powered conversational interface using Pydantic AI.

## Unit Tests

### 1. Pydantic AI Integration Tests (`test_pydantic_ai.py`)
- **Test model initialization**
  - Pydantic AI client setup
  - Model configuration
  - API key validation
  - Connection testing
  
- **Test model responses**
  - Response parsing
  - Type validation
  - Error handling
  - Retry logic
  
- **Test streaming**
  - Stream initialization
  - Chunk processing
  - Stream completion
  - Error recovery

### 2. Context Management Tests (`test_context_management.py`)
- **Test conversation memory**
  - Message history storage
  - Context window limits
  - Memory truncation strategies
  - Thread safety
  
- **Test context formatting**
  - System prompt construction
  - Retrieved document formatting
  - Conversation history formatting
  - Token counting
  
- **Test context relevance**
  - Relevant context selection
  - Context ranking
  - Duplicate removal
  - Context summarization

### 3. Prompt Template Tests (`test_prompt_templates.py`)
- **Test system prompt**
  - Documentation context injection
  - Role definition
  - Behavior constraints
  - Citation requirements
  
- **Test query reformulation**
  - Conversation-aware reformulation
  - Intent preservation
  - Context incorporation
  - Multiple reformulations
  
- **Test answer generation**
  - Citation formatting
  - Code block handling
  - List/table generation
  - Error message formatting

### 4. Citation Management Tests (`test_citations.py`)
- **Test source attribution**
  - Citation extraction
  - Source linking
  - Reference formatting
  - Uniqueness checking
  
- **Test citation accuracy**
  - Content-citation matching
  - Citation validation
  - Missing citation detection
  - Over-citation prevention
  
- **Test citation formats**
  - Inline citations [1]
  - Footnote style
  - Source sections
  - URL generation

### 5. Conversation Flow Tests (`test_conversation_flow.py`)
- **Test single-turn Q&A**
  - Question → Answer flow
  - Context retrieval
  - Response generation
  - Citation inclusion
  
- **Test multi-turn conversations**
  - Context preservation
  - Follow-up questions
  - Clarification requests
  - Topic switching
  
- **Test conversation patterns**
  - Greeting handling
  - Goodbye detection
  - Command recognition
  - Meta-questions

## Integration Tests

### 1. End-to-End Chat Tests (`test_chat_integration.py`)
- **Test complete chat flow**
  - User input → Final response
  - Search integration
  - LLM interaction
  - Response formatting
  
- **Test different question types**
  - How-to questions
  - Troubleshooting
  - Conceptual queries
  - Code examples
  - Configuration help
  
- **Test conversation quality**
  - Answer accuracy
  - Response coherence
  - Citation relevance
  - Helpfulness scoring

### 2. Streaming Response Tests (`test_streaming.py`)
- **Test stream initialization**
  - First token latency
  - Stream setup time
  - Error handling
  
- **Test chunk processing**
  - Partial response handling
  - Buffer management
  - Unicode handling
  - Markdown preservation
  
- **Test stream completion**
  - Final response assembly
  - Citation appendix
  - Cleanup operations
  - Connection closing

### 3. Persistence Tests (`test_persistence.py`)
- **Test history storage**
  - Conversation saving
  - Session management
  - User identification
  - Privacy controls
  
- **Test history retrieval**
  - Session restoration
  - Context rebuilding
  - Timestamp handling
  - Search in history
  
- **Test cleanup**
  - Old session removal
  - Storage limits
  - Data retention
  - Export functionality

## Edge Cases and Error Handling

### 1. Input Edge Cases (`test_input_edge_cases.py`)
- **Test unusual inputs**
  - Empty messages
  - Very long questions
  - Code as questions
  - Multiple questions
  - Non-questions
  
- **Test problematic content**
  - Injection attempts
  - Harmful requests
  - Off-topic queries
  - Nonsensical input
  
- **Test encoding issues**
  - Unicode questions
  - Mixed encodings
  - Special characters
  - Emoji handling

### 2. System Error Tests (`test_chat_errors.py`)
- **Test LLM failures**
  - API timeouts
  - Rate limiting
  - Model errors
  - Fallback responses
  
- **Test search failures**
  - No results found
  - Search timeout
  - Database errors
  - Graceful degradation
  
- **Test resource limits**
  - Context too large
  - Memory constraints
  - Response size limits
  - Concurrent sessions

## Mock/Stub Requirements

### 1. LLM Mock
```python
class MockLLM:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.call_history = []
        self.response_delay = 0.1
    
    async def complete(self, messages: List[dict], stream: bool = False):
        self.call_history.append(messages)
        
        if stream:
            return self._stream_response()
        else:
            return self._complete_response()
    
    async def _stream_response(self):
        response = "Based on the documentation, here's the answer"
        for chunk in response.split():
            await asyncio.sleep(self.response_delay)
            yield chunk + " "
    
    def _complete_response(self):
        return {
            "content": "Based on the documentation, AppPack is...",
            "citations": ["deployment.md", "configuration.md"]
        }
```

### 2. Search Mock
```python
class MockSearcher:
    def __init__(self):
        self.mock_results = {
            "deployment": [
                {
                    "content": "AppPack deployment process...",
                    "metadata": {"source": "deployment.md", "section": "Getting Started"},
                    "score": 0.95
                }
            ],
            "default": [
                {
                    "content": "General AppPack information...",
                    "metadata": {"source": "overview.md", "section": "Introduction"},
                    "score": 0.7
                }
            ]
        }
    
    def search(self, query: str, k: int = 5) -> List[dict]:
        # Return relevant mock results based on query
        for keyword, results in self.mock_results.items():
            if keyword in query.lower():
                return results[:k]
        return self.mock_results["default"][:k]
```

### 3. Conversation Store Mock
```python
class MockConversationStore:
    def __init__(self):
        self.conversations = {}
    
    def save_message(self, session_id: str, role: str, content: str):
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
    
    def get_history(self, session_id: str, limit: int = 10):
        return self.conversations.get(session_id, [])[-limit:]
```

## Test Data Requirements

### 1. Question Test Sets (`tests/fixtures/questions/`)
```json
{
  "basic_questions": [
    "What is AppPack?",
    "How do I deploy an application?",
    "What are environment variables?"
  ],
  "technical_questions": [
    "How do I configure a custom domain with SSL?",
    "What's the difference between staging and production environments?",
    "How can I debug a failing deployment?"
  ],
  "troubleshooting": [
    "My app won't start, what should I check?",
    "I'm getting a 502 error, how do I fix it?",
    "The deployment is stuck, what can I do?"
  ],
  "multi_turn_conversations": [
    {
      "conversation": [
        "How do I deploy a Python app?",
        "What version of Python is supported?",
        "Can I use a requirements.txt file?",
        "What about Poetry?"
      ]
    }
  ]
}
```

### 2. Expected Responses (`tests/fixtures/expected_responses/`)
- Model responses for quality comparison
- Required elements (citations, formatting)
- Unacceptable response patterns

### 3. Context Documents (`tests/fixtures/context_docs/`)
```python
# Sample retrieved documents for testing
SAMPLE_CONTEXTS = [
    {
        "content": "AppPack supports Python applications using pip, poetry, or pipenv...",
        "metadata": {
            "source": "languages/python.md",
            "section": "Package Management",
            "url": "https://docs.apppack.io/languages/python#package-management"
        }
    },
    # More sample contexts...
]
```

### 4. Prompt Templates (`tests/fixtures/prompts/`)
```python
SYSTEM_PROMPT = """You are a helpful assistant for AppPack documentation.
You have access to the following documentation context:
{context}

Guidelines:
- Always cite your sources using [source] notation
- Provide specific, actionable answers
- Include code examples when relevant
- Admit if something is not in the documentation
"""

QUERY_REFORMULATION_PROMPT = """Given the conversation history:
{history}

And the user's question: {question}

Reformulate this into a search query that captures the user's intent.
"""
```

## Test Execution Strategy

1. **Unit testing**: All components in isolation
2. **Integration testing**: Complete chat flows
3. **User simulation**: Realistic conversations
4. **Performance testing**: Response times
5. **Quality assessment**: Manual review sampling

## Quality Metrics

### Response Quality Metrics
- **Accuracy**: Correct information percentage
- **Completeness**: All aspects addressed
- **Relevance**: On-topic responses
- **Clarity**: Readability scores
- **Citation accuracy**: Correct source attribution

### Performance Metrics
- **Time to first token**: <1s (streaming)
- **Complete response time**: <5s
- **Context retrieval**: <200ms
- **Memory usage**: <500MB per session
- **Concurrent sessions**: >100

### User Experience Metrics
- **Conversation coherence**: Context preservation
- **Follow-up handling**: Understanding rate
- **Error recovery**: Graceful failure rate
- **Helpfulness**: User satisfaction proxy

## Success Criteria

- All unit tests pass with >95% coverage
- Integration tests complete successfully
- Response quality metrics:
  - Accuracy > 90%
  - Citation rate > 95%
  - Relevance > 85%
- Performance targets met
- Streaming works smoothly
- Multi-turn conversations maintain context
- Error handling prevents crashes
- Citations link to correct sources