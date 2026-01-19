import os
# os.chdir("coicop-rag")
import yaml
import datetime
import uuid
from tqdm import tqdm
import duckdb
import pandas as pd
from qdrant_client import QdrantClient
from openai import OpenAI
from langfuse import Langfuse
from data.parsing import extract_json_from_response
from src.utils import (
    merge_eval_and_retreived, 
    apply_rules
)
from src.eval.metrics import (
  compute_hierarchical_metrics,
  flatten_metrics,
  print_metrics_report
)

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

# Import prompt template
prompt_template = Langfuse().get_prompt(
    config["llm"]["prompt_name"], 
    label=config["llm"]["prompt_version"]
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

qdrant_results_texts = []
qdrant_results_codes = []

for search_embedding in tqdm(search_embeddings, desc="Vectorial search"):
    points = client_qdrant.query_points(
        collection_name=config["qdrant"]["collection_name"],
        query=search_embedding,
        limit=config["retrieval"]["size"],
    )

    qdrant_results_texts.append(
        [point["payload"]["text"] for point in points.model_dump()["points"]]
    )
    qdrant_results_codes.append(
        [point["payload"]["code"] for point in points.model_dump()["points"]]
    )

print(f"Number of vectorial searches done : {len(qdrant_results_texts)}")
print(f"Number of points returns per search : {len(qdrant_results_texts[0])}")


# get prompts ----------------------

messages = []
for i, searched_product in enumerate(searched_products):
    if searched_product["enseigne"]:
        enseigne_bloc = f"# Pour information, ce produit a été acheté dans cette enseigne : {searched_product["enseigne"]}"
    else:
        enseigne_bloc = None
    
    messages.append(
        prompt_template.compile(
            product=searched_product["product"],
            enseigne_bloc=enseigne_bloc,
            proposed_codes=qdrant_results_texts[i],
            list_proposed_codes=qdrant_results_codes[i]
        )
    )
# print(messages[0][1]["content"])

# for message in messages:
#     print(message[1]["content"])

print("Starting generation")

llm_responses = []
for message in tqdm(messages, desc="LLM generation"):
    llm_responses.append(
        client_llm.chat.completions.create(
            model=config["llm"]["model_name"],
            messages=message,
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"],
            response_format={"type": "json_object"}
        )
    )

print("Parsing LLM responses")

llm_responses_parsed = []
for llm_response in llm_responses:
    content = llm_response.choices[0].message.content
    llm_responses_parsed.append(
        extract_json_from_response(content)
    )

# Evaluation (must be same order !) ------------------

print("Create an evaluation df")

rows = []
for i in range(len(llm_responses_parsed)):
    pred = llm_responses_parsed[i]
    annotation = searched_products[i]
    row = pred | annotation
    row["good_pred"] = (row["code"] == row["coicop_pred"])
    rows.append(row)

print("Export predictions")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

df_eval = pd.DataFrame(rows)
df_retrieved_codes = pd.DataFrame(qdrant_results_codes)
df_retrieved_codes["id"] = df_eval["id"]


con.sql(f"""
    COPY df_eval 
    TO '{config['predictions']['s3_path'].format(timestamp=timestamp)}'
    (FORMAT PARQUET)
""")

con.sql(f"""
    COPY df_retrieved_codes 
    TO '{config['predictions']['s3_path_retrieved_codes'].format(timestamp=timestamp)}'
    (FORMAT PARQUET)
""")


print("All done !")

# Metrics -------------------------------------------

records = merge_eval_and_retreived(
    df_eval=df_eval,
    retrieved_codes=df_retrieved_codes,
    retrieval_size=config["retrieval"]["size"],
)

records = apply_rules(
    records=records,
    path_rules=config["eval"]["rules_path"]
)

records_rag = [record for record in records if record["coding_tool"] == "rag"]
records_regex = [record for record in records if record["coding_tool"] == "regex"]

len(records_rag)
len(records_regex)

metrics = compute_hierarchical_metrics(
  records=records_rag,
  threshold=config["eval"]["threshold_confidence"]
)

metrics_mlflow = flatten_metrics(metrics)

print_metrics_report(metrics)

