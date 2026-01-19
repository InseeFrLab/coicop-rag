# %%

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

import yaml
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

threshold_confidence = config["eval"]["threshold_confidence"]
retrieval_size = config["retrieval"]["size"]


con = duckdb.connect(database=":memory:")

s3_path_predictions = "s3://projet-budget-famille/data/rag/predictions_20260107_180045.parquet"
s3_path_retrieved_codes = "s3://projet-budget-famille/data/rag/retrieved_codes_20260107_180045.parquet"
query_definition = f"SELECT * FROM read_parquet('{s3_path_predictions}')"
df_eval = con.sql(query_definition).to_df()

retrieved_codes = con.sql(f"SELECT * FROM read_parquet('{s3_path_retrieved_codes}')").to_df()

# Preprocessing --------------------------------------

# Preprocess rag records

from src.utils import merge_eval_and_retreived
records = merge_eval_and_retreived(
    df_eval=df_eval,
    retrieved_codes=retrieved_codes,
    retrieval_size=retrieval_size,
)

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



from src.utils import apply_rules

records = apply_rules(
    records=records,
    path_rules=config["eval"]["rules_path"]
)

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



pd.DataFrame(errors_list)[
  ["product", "enseigne", "code", "coicop_pred","confidence", "in_retrieved", "list_retrieved_codes"]
].sample(5)

[m for m in errors_list if m["product"]=="cadre"]


# ----------------------------------------------

df_eval.columns
df_eval["good_pred"].mean()
df_eval["parsed"].value_counts()
df_eval["parsed"].dtype
df_eval["codable"].dtype
df_eval["codable"].value_counts()
df_eval["coicop_pred"]
df_eval["code"].isna().sum()
df_eval["coicop_pred"].isna().sum()
df_eval["good_pred"].isna().sum()
df_eval.loc[df_eval["coicop_pred"].isna()]

truncate_code("01.2.3.0.7.000", level=5)
truncate_code(None, level=5)

accuracy, results = calculate_accuracy_at_level(
    df_eval.to_dict('records'),
    "coicop_pred",
    "code",
    4
)

accuracy, results = calculate_accuracy_at_level(
    df_eval[df_eval["confidence"]>0.7].to_dict('records'),
    "coicop_pred",
    "code",
    4
)


df_eval["result"] = results

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_eval, x='confidence', hue="result", common_norm=False, fill=True, alpha=0.3)
plt.title("Distribution de l'indice de confiance par résultat de prédiction")
plt.xlabel("Indice de confiance (confidence_0)")
plt.ylabel("Densité")
plt.legend(title='Résultat', labels=['Faux (False)', 'Vrai (True)'])
plt.grid(True, alpha=0.3)
plt.show()
output_path = "distribution_confidence_par_resultat.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")


df_eval.groupby("result")["confidence"].mean()

print(
        df_eval
          .loc[
            ~df_eval["result"], 
            ["product", "enseigne", "code", "coicop_pred","confidence"]
            ]
          .sort_values(by="confidence", ascending=False)
          .head(20)
)

print(
        df_eval
          .loc[
            ~df_eval["result"], 
            ["product", "enseigne", "code", "coicop_pred","confidence"]
            ]
          .sample(20)
)
pd.reset_option("display.max_colwidth")
str(df_eval.loc[df_eval["product"] == "billets avion", "reasons"].to_string(index=False))


# %%
