import os
# os.chdir("coicop-rag")
import yaml
import uuid
from tqdm import tqdm
import duckdb
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

sample_size = config["annotations"]["sample_size"]

con = duckdb.connect(database=":memory:")

# Qdrant config
client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"], 
    api_key=os.environ["QDRANT_API_KEY"],
    port=os.environ["QDRANT_API_PORT"]
)

# llm config
client_llm = OpenAI(
    api_key=os.environ["OLLAMA_API_KEY"],
    base_url=os.environ["OLLAMA_URL"]
)

# Import searched products 

query_definition = f"SELECT * FROM read_parquet('{config["annotations"]["s3_path"]}')"
annotations = con.sql(query_definition).to_df()

searched_products = (
    annotations.loc[
        annotations["manual_from_books"],  # Only hand-written spendings
        ["product", "code", "coicop", "enseigne", "budget"]
    ]
    .assign(id=lambda x: [str(uuid.uuid4()) for _ in range(len(x))])
    .to_dict(orient="records")
)

if sample_size:
    import random
    random.seed(42)
    searched_products = random.sample(searched_products, sample_size)

print(f"Number of spendings to code: {len(searched_products)}")

print("Starting spendings embeddings")

search_embeddings = []
for searched_product in tqdm(searched_products, desc="Generating embeddings"):
    response = client_llm.embeddings.create(
        model=config["embedding"]["model_name"],
        input=searched_product['product']
    )
    search_embeddings.append(response.data[0].embedding)

print(f"Embedding dimension : {len(search_embeddings[0])}")

# Search one by one (to batch !)

qdrant_results = []
for search_embedding in tqdm(search_embeddings, desc="Vectorial search"):
    points = client_qdrant.query_points(
        collection_name=config["qdrant"]["collection_name"],
        query=search_embedding,
        limit=config["retrieval"]["size"],
    )
    qdrant_results.append(
        [point["payload"]["text"] for point in points.model_dump()["points"]]
    )

print(f"Number of vectorial searches done : {len(qdrant_results)}")
print(f"Number of points returns per search : {len(qdrant_results[0])}")


example_num = 0
choices = qdrant_results[example_num]
searched_product = searched_products[example_num]

system_prompt = """
    Tu es un expert en classification COICOP (Classification of Individual Consumption According to Purpose).
    Ton rôle est d'aider à classifier des produits selon la nomenclature COICOP en te basant sur les codes pertinents fournis.

    Instructions:
    1. Analyse le produit demandé
    2. Compare avec les codes COICOP fournis
    3. Recommande le code le plus approprié
    4. Justifie ton choix en expliquant pourquoi ce code correspond
    5. Si plusieurs codes sont possibles, explique les nuances
    6. Réponds en français de manière claire et concise
"""

query = f"""
Trouve à quel code de la classification correspond le produit suivant : "{searched_product["product"]}"{f" (acheté dans l'enseigne {searched_product['enseigne']}" if searched_product["enseigne"] is not None else ""})
"""

user_prompt = f"""
Question: {query}\n\nListe des possibilités:\n{choices}
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

llm_model: str = "gpt-oss:120b"
temperature = 0.1
max_tokens = 256


response = client_llm.chat.completions.create(
    model=llm_model,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens
)

answer = response.choices[0].message.content
print(answer)

