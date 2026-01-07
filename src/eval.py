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
retrieval_size = 5

con = duckdb.connect(database=":memory:")

s3_path_predictions = "s3://projet-budget-famille/data/rag/predictions_20260106_155655.parquet"
s3_path_retrieved_codes = "s3://projet-budget-famille/data/rag/retrieved_codes_20260106_155655.parquet"
query_definition = f"SELECT * FROM read_parquet('{s3_path_predictions}')"
df_eval = con.sql(query_definition).to_df()

retrieved_codes = con.sql(f"SELECT * FROM read_parquet('{s3_path_retrieved_codes}')").to_df()

predictions = df_eval.to_dict('records')

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


# Eval --------------------------------------

records_rag = [record for record in records if record["coding_tool"] == "rag"]
records_regex = [record for record in records if record["coding_tool"] == "regex"]

len(records_rag)
len(records_regex)



calculate_accuracy_at_level(
    records_rag,
    "coicop_pred",
    "code",
    4,
    "list_retrieved_codes"
)


metrics = compute_hierarchical_metrics(records_rag)

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


# Compute metrics
metrics = compute_hierarchical_metrics(df_eval)
metrics["all_raw"]["level_5"]

# Print formatted report
print_metrics_report(metrics)

# Export to DataFrame for further analysis
metrics_df = export_metrics_to_dataframe(metrics)
print("\nMetrics as DataFrame:")
print(metrics_df)

# Access specific metrics programmatically
print("\n" + "="*80)
print("SPECIFIC METRIC ACCESS EXAMPLES")
print("="*80)
print(f"All parsed - Level 3 accuracy: {metrics['all_parsed']['level_3']:.2%}")
print(f"Codable only - Level 5 accuracy: {metrics['codable_only']['level_5']:.2%}")
print(f"Parsed & codable - Level 1 accuracy: {metrics['parsed_and_codable']['level_1']:.2%}")
# %%
