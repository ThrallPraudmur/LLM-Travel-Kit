## Как обернуть open-source LLM в классы для LangChain

### OpenRouter

```python
import requests
import json

OPENROUTER_API_KEY = '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'

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    # "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    # "model": "qwen/qwen-2.5-7b-instruct:free",
    "model": "qwen/qwen2.5-vl-72b-instruct:free",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ],
    'max_tokens': 12,
    'temperature': 0.1,

  })
)

response.json()['choices'][0]['message']['content']
```

```python
class OpenRouterLLM(BaseChatModel):
    model_name: str = "qwen/qwen-2.5-7b-instruct:free"
    # model_name: str = "qwen/qwen2.5-vl-72b-instruct:free"
    # model_name: str = "mistralai/mistral-nemo:free"
    # model_name: str = "google/gemini-2.5-pro-exp-03-25"
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key: str = "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"

    def _llm_type(self) -> str:
        return "openrouter-custom"

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        payload = {
            "model": self.model_name,
            "messages": [{"role": msg.type, "content": msg.content} for msg in messages],
            "temperature": 0.1,
            # "max_tokens": 1024,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )
```

###

```python

```

### Together AI

