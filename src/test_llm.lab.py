import os
from openai import OpenAI

client_llm = OpenAI(
    api_key=os.environ["OLLAMA_API_KEY"],
    base_url=os.environ["OLLAMA_URL"]
)

response = client_llm.embeddings.create(
    model="qwen3-embedding:8b",
    input="chaine à embedder"
)

response = client_llm.embeddings.create(
    model="gpt-oss:120b",
    input="Salut toi"
)

result = client_llm.chat.completions.create(
    model="gpt-oss:120b",
    messages=[
        {"role": "user", "content": "Salut toi"}
    ],
    temperature=0.1,
    max_tokens=256,
    response_format={"type": "json_object"}
)
