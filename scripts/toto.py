from openai import OpenAI

client_vllm_emb = OpenAI(
    base_url="https://vllm-embed2.user.lab.sspcloud.fr/v1",
    api_key=""
)

client_vllm_emb.models.list()


print(client_vllm_emb.base_url)

response = client_vllm_emb.embeddings.create(
            model='Qwen/Qwen3-Embedding-8B',
            input="Pâté de pangolin"
        )


client_vllm_gen = OpenAI(
    base_url="https://vllm-gpt-oss120-gen2.user.lab.sspcloud.fr/v1",
    api_key=""
)

client_vllm_gen.models.list()



print(client_vllm_gen.base_url)

response = client_vllm_emb.embeddings.create(
            model='Qwen/Qwen3-Embedding-8B',
            input="Pâté de pangolin"
        )