
import os
# os.chdir("coicop-rag")
import duckdb
import yaml
import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
from openai import OpenAI
from src.data.coicop_document import CoicopDocument

load_dotenv()

# DuckDB config
con = duckdb.connect(database=":memory:")

# Params
qdrant_url = os.environ["QDRANT_URL"]
qdrant_api_key = os.environ["QDRANT_API_KEY"]

with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set clients
client = OpenAI(
    api_key=os.environ["OLLAMA_API_KEY"],
    base_url=os.environ["OLLAMA_URL"]
)

# Qdrant config
client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    port=os.environ["QDRANT_API_PORT"]
)

# Import coicop notices

query = f"""
    SELECT
        * 
    FROM read_csv('{config["coicop"]["path"]}');
"""
notices_raw = duckdb.sql(query).to_df()

columns_to_keep = [
    col for col in notices_raw.columns 
    if 'column' not in col.lower() and not col.endswith('_en')
]

notices_raw = notices_raw[columns_to_keep]
notices_raw = notices_raw.to_dict(orient="records")


# Create documents to embed and upload to vectorial database
documents = []
for notice in notices_raw:
    doc = CoicopDocument(
        code=str(notice['code']),
        label_fr=str(notice['label_fr']),
        note_generale_fr=notice.get('note_generale_fr'),
        contenu_central_fr=notice.get('contenu_central_fr'),
        contenu_additionnel_fr=notice.get('contenu_additionnel_fr'),
        note_exclusion_fr=notice.get('note_exclusion_fr')
    )
    chunk = doc.to_text_chunks()
    documents.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk["text"],
                        "metadata": {
                            "code": doc.code,
                            "label_fr": doc.label_fr,
                            "strategy": chunk["type"],
                        }
                    })

# print(documents[6])



#client_qdrant.get_collections()

client_qdrant.recreate_collection(
    collection_name=config["qdrant"]["collection_name"],
    vectors_config=VectorParams(
        size=config["embedding"]["model_len"],
        distance=Distance.COSINE
    )
)

embeddings = []
for document in documents:
    response = client.embeddings.create(
        model=config["embedding"]["model_name"],
        input=document["text"]
    )
    embeddings.append(response.data[0].embedding)

print(len(embeddings))
print(len(embeddings[0]))

 

# Créer les points pour Qdrant
points = []
for i, (document, embedding) in enumerate(zip(documents, embeddings)):
    points.append(
        PointStruct(
            id=document["id"],
            vector=embedding,
            payload={
                "text": document["text"],
                **document["metadata"]
            }
        )
    )

# Upload par batchs
print(f"Uploading {len(points)} points to Qdrant...")
batch_size = 5
for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    client_qdrant.upsert(
        collection_name=config["qdrant"]["collection_name"],
        points=batch
    )
    print(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

print("✓ Upload terminé!")

