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


from openai import OpenAI

OPENAI_API_BASE_URL="http://projet-budget-famille-vllm-gpt-oss:8000/v1"
OPENAI_API_BASE_URL="http://projet-models-hf-vllm-gpt-oss:8000/v1"

client = OpenAI(
    base_url=OPENAI_API_BASE_URL,
    api_key="EMPTY"  # souvent ignoré par vLLM
)

models = client.models.list()

response = client.chat.completions.create(
    model="oss-120B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explique le théorème de Bayes."}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
