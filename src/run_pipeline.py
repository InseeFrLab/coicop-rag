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


# Environment --------------------------------------

import os
os.chdir("..")
# os.chdir("coicop-rag")
import re
import duckdb
import pandas as pd
from src.eval.metrics import (
  truncate_code, 
  compute_hierarchical_metrics, 
  calculate_accuracy_at_level, 
  print_metrics_report,
  analyze_error_sources,
  print_error_analysis,
  export_metrics_to_list
)
pd.reset_option("display.max_colwidth")
pd.set_option('display.max_rows', None)
retrieval_size = 5
threshold_confidence = 0.7

con = duckdb.connect(database=":memory:")

s3_path_predictions = "s3://projet-budget-famille/data/rag/predictions_20260107_180045.parquet"
s3_path_retrieved_codes = "s3://projet-budget-famille/data/rag/retrieved_codes_20260107_180045.parquet"
query_definition = f"SELECT * FROM read_parquet('{s3_path_predictions}')"
df_eval = con.sql(query_definition).to_df()

retrieved_codes = con.sql(f"SELECT * FROM read_parquet('{s3_path_retrieved_codes}')").to_df()

# Preprocessing --------------------------------------

# Preprocess rag records
cols = [str(i) for i in range(retrieval_size)]
retrieved_codes["list_retrieved_codes"] = retrieved_codes[cols].values.tolist()
retrieved_codes = retrieved_codes.drop(cols, axis=1)

df_eval = df_eval.merge(retrieved_codes, how="left", on="id")

df_eval["in_retrieved"] = df_eval.apply(
    lambda row: row["code"] in row["list_retrieved_codes"],
    axis=1
)

records = df_eval.to_dict('records')

len(records)
## Filtre des cas à gérer a priori -------------

# s3_path_duplicated_annotations = "s3://projet-budget-famille/data/output-annotation-consolidated-2026-01-05/annotations_with_multiple_codes_hors_copain.parquet"
# df_duplicated = con.sql(f"SELECT * FROM read_parquet('{s3_path_duplicated_annotations}')").to_df()

# df_product_counts = (
#     df_duplicated["product"]
#     .value_counts(ascending=True)
#     .reset_index()
#     .rename(columns={"index": "product", "product": "count"})
# )
# df_product_counts

# df_duplicated[df_duplicated["product"] == "marche"]

pattern_code_pairs = [
    (r"fruits? et l[eé]gumes?", "01.1"),
    (r"^l[eéèêë]gum[eéèêë]s?$", "01.1.7"),
    (r"^fruits?$", "01.1.6"),
    (r"\b(divers\s+)?courses?\b", "98.1"),
    (r"^\s*boulangerie\s*$", "01.1.1.3"),
    (r"^\s*billeterie\s*$", "09.4"),
    (r"^\s*restaurant\s*$", "11.1.1"),
    (r"^\s*resto$", "11.1.1"),
    (r"^carte bancaire$", "98.3"),
    (r"^alimentation?$", "98.1.1"),
    (r"^alimentaire$", "98.1.1"),
    (r"^courses alimentaires$", "98.1.1"),
    (r"^courses?$", "98.1"),
    (r"^reductions?.*", "98.5"),
    (r"^remises?.*", "98.5"), # Go reprise
    (r"^nourriture$", "98.1"), # Go reprise
    (r"^boissons?$", "98.1"), # Go reprise
    (r"^prelevement$", "98.4"), # Go reprise
    (r"^-10 % abonnement*", "98.5"), # Go reprise
    (r"^divers$", "98.2"), # Go reprise
    (r"^epicerie$", "98.1.1"), # Go reprise
    (r"^avantage carte 1028$", "99"), # Go reprise
    (r"^bon immediat$", "98.5"), # Go reprise
    (r"^rabais 30 %$", "98.5"), # Go reprise
    (r"^illisible$", "98.4"), # Go reprise
    (r"^[^a-zA-Z]*$", "98.4"), # Go reprise
    (r"^cantine$", "11.1.2.1"), # Go reprise
    (r"^cb$", "98"), # Go reprise
    (r"^marche$", "98.1.1"), # Go reprise
    (r"^surgeles?$", "98.1.1"), # Go reprise
    (r"^retrait$", "99.2"), # Go reprise
    (r"^boucher$", "01.1.2.2"), # Go reprise
]

# patterns = [p for p, _ in pattern_code_pairs]
# combined_pattern = "|".join(patterns)

pattern_code_pairs = [(re.compile(pattern, re.IGNORECASE), code) for pattern, code in pattern_code_pairs]

for entry in records:
    product = entry["product"]
    entry["coding_tool"] = "rag"
    for pattern, code in pattern_code_pairs:
        if pattern.fullmatch(product):
            entry["coding_tool"] = "regex"
            entry["code_predict"] = code
            break  # On arrête dès qu'un pattern correspond

len(records)

# Eval --------------------------------------

records_rag = [record for record in records if record["coding_tool"] == "rag"]
records_regex = [record for record in records if record["coding_tool"] == "regex"]

len(records_rag)
len(records_regex)


metrics = compute_hierarchical_metrics(
  records=records_rag,
  threshold=threshold_confidence
)

print_metrics_report(metrics)

error_analysis = analyze_error_sources(metrics)
print_error_analysis(error_analysis)

metrics_list = export_metrics_to_list(metrics)
metrics_df = pd.DataFrame(metrics_list)

print("\n" + "=" * 100)
print("METRICS SUMMARY TABLE")
print("=" * 100)
print(metrics_df.to_string(index=False))



# ----------------------------------------------
# Error analyses at level 4  

(
    overall_accuracy,
    result_list,
    retrieval_accuracy,
    generation_accuracy_when_retrieved,
    label_in_retrieved_list
) = calculate_accuracy_at_level(
    records=records_rag,
    predicted_col="coicop_pred",
    label_col="code",
    level=4,
    retrieved_col='list_retrieved_codes'
)

errors_list = [x for x, m in zip(records_rag, result_list) if not m]
print(f"Number of errors : {len(errors_list)} (on a total of {len(records_rag)})")

errors_list_high_confidence = [x for x in errors_list if x["confidence"] > threshold_confidence]
print(f"""
  Number of errors despite high confidence (>{threshold_confidence}) : {len(errors_list_high_confidence)})
  (on a total of {len(errors_list)} errors)
""")

errors_special_codes = [x for x in errors_list if (x["code"][:2] in ("98","99"))]
n_errors = len(errors_list)
n_errors_special_codes = len(errors_special_codes)
n_errors_special_codes/n_errors
print(f"""
  Number of errors due to special BDF codes (98, 99) : {n_errors_special_codes})
  (on a total of {len(errors_list)} errors ==> proprtion = {round(100 * n_errors_special_codes/n_errors, 1)}%)
""")

errors_normal_codes = [x for x in errors_list if (x["code"][:2] not in ("98", "99"))]
errors_normal_codes_too_precise = [
  x for x in errors_normal_codes
  if (x["coicop_pred"] and x["coicop_pred"].startswith(x["code"]))
]
n_errors_normal_codes = len(errors_normal_codes)
n_errors_normal_codes_too_precise = len(errors_normal_codes_too_precise)

print(f"""
  Number of errors due to overprecise predictions : {n_errors_normal_codes_too_precise} among normal codes (total of {n_errors_normal_codes}))
  proportion = {round(100 * n_errors_normal_codes_too_precise/n_errors_normal_codes, 1)}%)
""")


