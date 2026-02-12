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



URL_VLLM_GENERATION = "http://projet-models-hf-vllm-embed.projet-budget-famille:8000/v1"
URL_VLLM_EMBEDDING = "http://projet-models-hf-vllm-embed-bdf.projet-budget-famille:8000/v1"

from openai import OpenAI

client_gen = OpenAI(
    base_url=URL_VLLM_GENERATION,
    api_key=""  # vLLM ignore généralement la clé
)

client_emb = OpenAI(
    base_url=URL_VLLM_EMBEDDING,
    api_key="EMPTY"  # vLLM ignore généralement la clé
)


client_gen.models.list()
emb_model_name = client_emb.models.list().data[0].id

response = client_emb.embeddings.create(
        model=emb_model_name,
        input="Ceci est une phrase de test pour vérifier l'embedding."
    )


response = client_gen.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Mara des bois"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
