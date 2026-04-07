from openai import OpenAI

# client_vllm_emb = OpenAI(
#     base_url="https://vllm-embed2.user.lab.sspcloud.fr/v1",
#     api_key=""
# )

# client_vllm_emb.models.list()


# print(client_vllm_emb.base_url)

# response = client_vllm_emb.embeddings.create(
#             model='Qwen/Qwen3-Embedding-8B',
#             input="Pâté de pangolin"
#         )


client_vllm_gen = OpenAI(
    base_url= "https://vllm-generation3.user.lab.sspcloud.fr/v1",  #"https://vllm-gpt-oss120-gen2.user.lab.sspcloud.fr/v1",
    api_key=""
)

client_vllm_gen = OpenAI(
    base_url="https://vllm-gpt-oss120-gen2.user.lab.sspcloud.fr/v1",
    api_key=""
)

client_vllm_gen.models.list()



print(client_vllm_gen.base_url)

# ── Test chat completion via client_vllm_gen ─────────────────────────────────
model_name = client_vllm_gen.models.list().data[0].id

response_gen = client_vllm_gen.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": "Réponds en un mot : quelle est la capitale de la France ?"}
    ],
    temperature=0.1,
    max_tokens=2048,
)

print("Model  :", model_name)
print("Reply  :", response_gen.choices[0].message.content)
print("Tokens :", response_gen.usage)


response = client_vllm_gen.responses.input_tokens.count(
    model="openai/gpt-oss-120b",
    input="Tell me a joke."
)
print(response.input_tokens)



client.chat.completions.create(
                model=config["llm"]["model_name"],
                messages=message,
                temperature=config["llm"]["temperature"],
                max_tokens=config["llm"]["max_tokens"],
                extra_body=extra_body,
            )


# ── Test guided JSON decoding (structured output) ─────────────────────────────
from pydantic import BaseModel
from typing import Optional

class ReponseFormat(BaseModel):
    codable: bool
    code_predict: Optional[str] = None
    confidence: float
    reasons: str

for backend in ["xgrammar", "outlines", None]:
    extra_body = {"guided_json": ReponseFormat.model_json_schema()}
    if backend:
        extra_body["guided_decoding_backend"] = backend
    try:
        r = client_vllm_gen.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Produit : lait entier. Code COICOP prédit ?"}],
            temperature=0.1,
            max_tokens=128,
            extra_body=extra_body,
        )
        print(f"guided_decoding_backend={backend!r:12} → OK  : {r.choices[0].message.content[:80]}")
    except Exception as e:
        print(f"guided_decoding_backend={backend!r:12} → ERR : {e}")

# ── Test avec prompt long (simulation pipeline) ───────────────────────────────
# Simule un prompt avec 10 codes COICOP + descriptions (~3000 tokens)
fake_code_block = """
## 01.1.1.1 — Lait frais entier
Le lait entier frais pasteurisé, conditionné en bouteilles ou briques.

## 01.1.1.2 — Lait demi-écrémé
Le lait demi-écrémé frais pasteurisé, conditionné en briques.

## 01.1.2.1 — Yaourt nature
Yaourts natures non sucrés, en pot individuel ou multipack.

## 01.1.2.2 — Yaourt aux fruits
Yaourts aromatisés aux fruits, sucrés, en pot ou multipack.

## 01.1.3.1 — Fromage à pâte molle
Camembert, brie, munster et autres fromages à pâte molle et croûte fleurie.

## 01.1.3.2 — Fromage à pâte pressée
Emmental, comté, gruyère et autres fromages à pâte pressée cuite ou non.

## 01.1.4.1 — Beurre
Beurre doux ou demi-sel, en plaquette de 250g ou 500g.

## 01.1.4.2 — Crème fraîche
Crème fraîche épaisse ou liquide, entière ou allégée.

## 01.1.5.1 — Œufs
Œufs de poule, en boîte de 6 ou 12, labels divers (bio, plein air, cage).

## 01.1.6.1 — Lait concentré et en poudre
Lait concentré sucré ou non, lait en poudre entier ou écrémé.
""" * 4  # répété pour atteindre ~3000 tokens

long_prompt = f"""Tu es un expert en classification COICOP.
Voici des exemples de codes et leurs descriptions :
{fake_code_block}
Produit à classer : lait entier bio demi-écrémé 1L.
Réponds en JSON avec les champs codable, code_predict, confidence, reasons."""

try:
    r = client_vllm_gen.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": long_prompt}],
        temperature=0.1,
        max_tokens=2048,
        extra_body={"guided_json": ReponseFormat.model_json_schema()},
    )
    print(f"\nLong prompt → OK  ({r.usage.prompt_tokens} prompt tokens, "
          f"{r.usage.completion_tokens} completion tokens)")
    print(r.choices[0].message.content[:200])
except Exception as e:
    print(f"\nLong prompt → ERR : {e}")



import requests                                                                                                                                                                                        
r = requests.get("https://vllm-generation3.user.lab.sspcloud.fr/metrics")                                                                                                                              
print(r.text) 