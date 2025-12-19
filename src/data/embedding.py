# %%
import os
import duckdb
import pandas
import uuid
from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Any, Optional
from typing import Optional, List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# DuckDB config
con = duckdb.connect(database=":memory:")

# Qdrant config
client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"], 
    api_key=os.environ["QDRANT_API_KEY"],
    port="443"
)

# Params
collection_name = "coicop_test"

# Import coicop notices

s3_path = "s3://projet-budget-famille/data/coicop-2018_envoi_rmes_20251022.csv"
query = f"""
    SELECT
        * 
    FROM read_csv('{s3_path}');
"""
notices_raw = duckdb.sql(query).to_df()

columns_to_keep = [
    col for col in notices_raw.columns 
    if 'column' not in col.lower() and not col.endswith('_en')
]

notices_raw = notices_raw[columns_to_keep]

# notices_raw
# notices_raw.loc[notices_raw["type"] == "Poste"].isna().sum()

# Define a CoicopDocument object

@dataclass
class CoicopDocument:
    """Structure of a COICOP document to embed"""
    code: str
    label_fr: str
    note_generale_fr: Optional[str] = None
    contenu_central_fr: Optional[str] = None
    contenu_additionnel_fr: Optional[str] = None
    note_exclusion_fr: Optional[str] = None
    
    @property
    def inclusions(self) -> Optional[str]:
        """Concatenate central and additional content"""
        parts = []
        if self.contenu_central_fr:
            parts.append(self.contenu_central_fr.strip())
        if self.contenu_additionnel_fr:
            parts.append(self.contenu_additionnel_fr.strip())
        return ". ".join(parts) if parts else None
        
    def to_single_text(self, strategy: str = "all_info") -> str:
        """
        Convert to a single Markdown-formatted text according to strategy.
        Each section is separated by a clear line break for embedding.
        """
        lines = [f"**{self.code}: {self.label_fr}**"]  # Bold title
        
        if strategy != "code_only":
            if self.note_generale_fr:
                lines.append(f"**General note:** {self.note_generale_fr}")
            
            inclusions = self.inclusions
            if inclusions:
                lines.append(f"**Inclusions:** {inclusions}")
            
            if strategy == "all_info" and self.note_exclusion_fr:
                lines.append(f"**Exclusions:** {self.note_exclusion_fr}")
        
        # Join sections with two line breaks for clear separation in embeddings
        return "\n\n".join(lines)
    
    def to_text_chunks(self, strategy: str = "all_info") -> List[Dict[str, str]]:
        """
        Convert to text chunks for embedding according to strategy.
        """
        chunk = {
            "type": strategy,
            "text": self.to_single_text(strategy),
            "code": self.code
        }
        return chunk  

# %%
# df = notices_raw[:10]
# doc = []
# for _, row in df.iterrows():
#         doc.append(
#             CoicopDocument(
#                 code=str(row['code']),
#                 label_fr=str(row['label_fr']),
#                 note_generale_fr=row.get('note_generale_fr'),
#                 contenu_central_fr=row.get('contenu_central_fr'),
#                 contenu_additionnel_fr=row.get('contenu_additionnel_fr'),
#                 note_exclusion_fr=row.get('note_exclusion_fr')
#             )
#         )


# print(doc[6].to_text_chunks(strategy="code_only")["text"])
# print(doc[6].to_text_chunks(strategy="without_exclusions")["text"])
# print(doc[6].to_text_chunks()["text"])
# print(doc[6].to_single_text())


# %%

# Create documents to embed and upload to vectorial databse

df = notices_raw[:30]
documents = []
for _, row in df.iterrows():
    doc = CoicopDocument(
        code=str(row['code']),
        label_fr=str(row['label_fr']),
        note_generale_fr=row.get('note_generale_fr'),
        contenu_central_fr=row.get('contenu_central_fr'),
        contenu_additionnel_fr=row.get('contenu_additionnel_fr'),
        note_exclusion_fr=row.get('note_exclusion_fr')
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

print(documents[6])

# %%
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OLLAMA_API_KEY"],
    base_url="https://llm.lab.sspcloud.fr/api"
)
embedding_model_name = "qwen3-embedding:8b"
embedding_model_len = 4096


# %%

#client_qdrant.get_collections()

client_qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=embedding_model_len,
        distance=Distance.COSINE
    )
)

embeddings = []
for document in documents:
    response = client.embeddings.create(
        model=embedding_model_name,
        input=document["text"]
    )
    embeddings.append(response.data[0].embedding)

print(len(embeddings))
print(len(embeddings[0]))

# %% 

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
        collection_name=collection_name,
        points=batch
    )
    print(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

print("✓ Upload terminé!")




# %% 

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')

# sentences = ["Le chat dort.", "Un félin repose.", "Paris est en France."]
# embeddings = model.encode(sentences)
# embeddings.shape

texts = [doc["text"] for doc in documents]

# Générer les embeddings
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=56
)




# %%

# Search 

search_texts = [
    "Food and non-alcoholic beverages",
    "Household appliances and maintenance",
    "Clothing and footwear",
    "Housing and household services",
    "Health",
    "Transport",
    "Communication",
    "Recreation and culture",
    "Education",
    "Restaurants and hotels",
    "Financial services",
    "Personal care and effects",
    "Miscellaneous goods and services",
    "Environmental protection",
    "Other services"
]

search_embeddings = []
for text in search_texts:
    response = client.embeddings.create(
        model=embedding_model_name,
        input=text
    )
    search_embeddings.append(response.data[0].embedding)

print(len(search_embeddings))
print(len(search_embeddings[0]))

# Search one by one (to batch !)

nb_points_retreived = 5

# search_embedding = search_embeddings[6]

found_texts = []
for search_embedding in search_embeddings:
    points = client_qdrant.query_points(
        collection_name=collection_name,
        query=search_embedding,
        limit=nb_points_retreived,
    )
    found_texts.append([point["payload"]["text"] for point in points.model_dump()["points"]])

len(found_texts)
len(found_texts[5])

# %%

example_num = 6
choices = found_texts[example_num]
searched_product = search_texts[example_num]

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
Trouve à quel code de la classification correspond le produit suivant : {searched_product}
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
max_tokens = 512


response = client.chat.completions.create(
    model=llm_model,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens
)

answer = response.choices[0].message.content
print(answer)
# %%
