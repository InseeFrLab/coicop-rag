# %%
import os
os.chdir("..")
# os.chdir("coicop-rag")
import duckdb
import pandas as pd
from src.eval.metrics import truncate_code, compute_hierarchical_metrics, calculate_accuracy_at_level, print_metrics_report

con = duckdb.connect(database=":memory:")

s3_path = "s3://projet-budget-famille/data/rag/eval_test.parquet_20251220_154926.parquet"
query_definition = f"SELECT * FROM read_parquet('{s3_path}')"
df_eval = con.sql(query_definition).to_df()




pattern_code_pairs = [
    (r"fruits? et l[eé]gumes?", "01.1"),
    (r"^l[eéèêë]gum[eéèêë]s?$", "01.1.7"),
    (r"^fruits?$", "01.1.6"),
    (r"\b(divers\s+)?courses?\b", "98.1"),
    (r"^\s*boulangerie\s*$", "01.1.1.3"),
    (r"^\s*billeterie\s*$", "09.4"),
    (r"^\s*restaurant\s*$", "11.1.1")
]

import re

# Extraire les patterns regex
patterns = [p for p, _ in pattern_code_pairs]

# Combiner les patterns en une seule regex
combined_pattern = "|".join(patterns)

# Retirer les lignes dont 'product' match un des patterns
df_eval = df_eval[
    ~df_eval["product"].str.contains(
        combined_pattern,
        regex=True,
        case=False,
        na=False
    )
]




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
