# %%
import os
import duckdb
import pandas
import uuid

con = duckdb.connect(database=":memory:")

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


notices_raw
notices_raw.loc[notices_raw["type"] == "Poste"].isna().sum()


from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Any, Optional
from typing import Optional, List, Dict

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
    
    def to_text_chunks(self, strategy: str = "all_info") -> List[Dict[str, str]]:
        """
        Convert to text chunks for embedding according to strategy.
        strategy: "code_only", "all_info", "all_info_no_exclusions"
        """
        chunk = {
            "type": strategy,
            "text": self.to_single_text(strategy),
            "code": self.code
        }
        return chunk  
        
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

# %%
df = notices_raw[:10]
doc = []
for _, row in df.iterrows():
        doc.append(
            CoicopDocument(
                code=str(row['code']),
                label_fr=str(row['label_fr']),
                note_generale_fr=row.get('note_generale_fr'),
                contenu_central_fr=row.get('contenu_central_fr'),
                contenu_additionnel_fr=row.get('contenu_additionnel_fr'),
                note_exclusion_fr=row.get('note_exclusion_fr')
            )
        )

# %%
print(doc[6].to_text_chunks(strategy="code_only")["text"])
print(doc[6].to_text_chunks(strategy="without_exclusions")["text"])
print(doc[6].to_text_chunks()["text"])
print(doc[6].to_single_text())


# %%
df = notices_raw[:10]
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

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

collection_name = "coicop_test"

client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"], 
    api_key=os.environ["QDRANT_API_KEY"],
    port="443"
)

client_qdrant.get_collections()
client_qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384, # Comme les embeddings de test
        distance=Distance.COSINE
    )
)

# %% 

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


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

# Créer les points pour Qdrant
points = []
for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
    points.append(
        PointStruct(
            id=doc["id"],
            vector=embedding.tolist(),
            payload={
                "text": doc["text"],
                **doc["metadata"]
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
