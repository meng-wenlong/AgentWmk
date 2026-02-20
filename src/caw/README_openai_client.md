# Unified OpenAI Client Usage

## Overview
The CAW project now uses a unified OpenAI client to manage all OpenAI API interactions.

## Configuration

### Environment Variables
Set these environment variables to configure the client globally:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_DEFAULT_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
```

### Programmatic Configuration
```python
from caw import set_config, OpenAIConfig

# Set custom configuration
config = OpenAIConfig(
    api_key="custom-key",
    base_url="http://custom-server:8000/v1",
    default_model="gpt-4"
)
set_config(config)
```

## Usage Examples

### In Utility Functions
```python
from caw import get_openai_client

def my_utility_function(prompt: str):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### In Watermark Classes
Watermark classes automatically use the unified client:
```python
from caw.watermarks.abstract_watermark import AbstractWatermark

class MyWatermark(AbstractWatermark):
    def inject(self, messages):
        # self.client is automatically available
        response = self.client.chat.completions.create(...)
```

## Benefits
1. **Single configuration point** - Configure once, use everywhere
2. **Resource efficient** - One client instance shared across the application
3. **Consistent behavior** - Same client configuration for all components
4. **Easy testing** - Mock or reset the client for tests using `reset_client()`