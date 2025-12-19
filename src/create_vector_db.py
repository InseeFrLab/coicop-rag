#TODO : use different embedding flavours (example : without exclusions) and 
# strategies (ex: hierachical or flat)

import os
# os.chdir("coicop-rag")
import duckdb
import yaml
import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from data.coicop_document import CoicopDocument

# Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coicop_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set clients

con = duckdb.connect(database=":memory:")

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

logger.info("Starting data import process")

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

logger.info(f"Loaded {len(notices_raw)} notices from CSV")

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

logger.info(f"Created {len(documents)} document chunks")

client_qdrant.recreate_collection(
    collection_name=config["qdrant"]["collection_name"],
    vectors_config=VectorParams(
        size=config["embedding"]["model_len"],
        distance=Distance.COSINE
    )
)

logger.info(f"Recreated Qdrant collection: {config['qdrant']['collection_name']}")

embeddings = []
for i, document in enumerate(documents):
    try:
        response = client.embeddings.create(
            model=config["embedding"]["model_name"],
            input=document["text"]
        )
        embeddings.append(response.data[0].embedding)
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(documents)} embeddings")
    except Exception as e:
        logger.error(f"Failed to generate embedding for document {i}: {str(e)}")
        continue

logger.info(f"Generated {len(embeddings)} embeddings")

# Create points for qdrant

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

logger.info(f"Prepared {len(points)} points for upload")

upload_batch_size = config["qdrant"]["upload_batch_size"]

logger.info(f"Starting upload using batches of size {upload_batch_size}")

for i in range(0, len(points), upload_batch_size):
    batch = points[i:i + upload_batch_size]
    try:
        client_qdrant.upsert(
            collection_name=config["qdrant"]["collection_name"],
            points=batch
        )
        logger.info(f"Uploaded batch {i//upload_batch_size + 1}/{(len(points)-1)//upload_batch_size + 1}")
    except Exception as e:
        logger.error(f"Failed to upload batch {i//upload_batch_size + 1}: {str(e)}")
        continue

logger.info("✓ Upload completed successfully")

