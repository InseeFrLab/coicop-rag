# %%

# Environment --------------------------------------

import os
os.chdir("..")
# os.chdir("coicop-rag")
import re
import duckdb
import pandas as pd
from src.eval.metrics import truncate_code, compute_hierarchical_metrics, calculate_accuracy_at_level, print_metrics_report
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



# dev retrieval eval 

from typing import Dict, List, Optional, Tuple


def check_label_in_retrieved(
    label_code: str,
    retrieved_codes: List[str],
    level: int
) -> bool:
    """
    Check if the label code is present in the retrieved codes list at a given level
    
    Args:
        label_code: Ground truth code
        retrieved_codes: List of retrieved codes from RAG
        level: Hierarchical level (1-5) to check
    
    Returns:
        True if label is in retrieved codes at this level, False otherwise
    """
    if label_code is None or retrieved_codes is None:
        return False
    
    # Truncate label to specified level
    label_truncated = truncate_code(label_code, level)
    if label_truncated is None:
        return False
    
    # Check if any retrieved code matches at this level
    for retrieved_code in retrieved_codes:
        retrieved_truncated = truncate_code(retrieved_code, level)
        if retrieved_truncated == label_truncated:
            return True
    
    return False



def calculate_accuracy_at_level(
    records: List[Dict],
    predicted_col: str,
    label_col: str,
    level: int,
    retrieved_col: str = 'list_retrieved_codes'
) -> Tuple[float, List[bool], float, float, List[bool]]:
    """
    Calculate accuracy at a specific hierarchical level with retrieval analysis
    
    Args:
        records: List of dictionaries with predictions and labels
        predicted_col: Key name for predicted code
        label_col: Key name for labeled code
        level: Hierarchical level (1-5)
        retrieved_col: Key name for list of retrieved codes
    
    Returns:
        Tuple containing:
        - overall_accuracy: Overall accuracy (0.0 to 1.0)
        - result_list: List of bool indicating if each prediction is correct
        - retrieval_accuracy: Proportion of cases where label is in retrieved codes
        - generation_accuracy_when_retrieved: Accuracy when label is in retrieved codes
        - label_in_retrieved_list: List of bool indicating if label is in retrieved codes
    """
    correct = 0
    total = 0
    result_list = []
    label_in_retrieved_list = []
    
    # For generation accuracy when retrieved
    correct_when_retrieved = 0
    total_when_retrieved = 0
    
    for record in records:
        pred_code = record.get(predicted_col)
        label_code = record.get(label_col)
        retrieved_codes = record.get(retrieved_col, [])
        
        # Truncate codes to specified level
        pred_truncated = truncate_code(pred_code, level)
        label_truncated = truncate_code(label_code, level)
        
        # Skip if either truncation failed
        # if pred_truncated is None or label_truncated is None:
        #     result_list.append(False)
        #     label_in_retrieved_list.append(False)
        #     continue
        
        # Check if prediction is correct
        is_correct = (pred_truncated == label_truncated)
        result_list.append(is_correct)
        
        # Check if label is in retrieved codes
        label_is_retrieved = check_label_in_retrieved(
            label_code, 
            retrieved_codes, 
            level
        )
        label_in_retrieved_list.append(label_is_retrieved)
        
        # Update overall accuracy counters
        total += 1
        if is_correct:
            correct += 1
        
        # Update generation accuracy when retrieved counters
        if label_is_retrieved:
            total_when_retrieved += 1
            if is_correct:
                correct_when_retrieved += 1
    
    # Calculate accuracies
    overall_accuracy = correct / total if total > 0 else 0.0
    retrieval_accuracy = sum(label_in_retrieved_list) / len(label_in_retrieved_list) if len(label_in_retrieved_list) > 0 else 0.0
    generation_accuracy_when_retrieved = correct_when_retrieved / total_when_retrieved if total_when_retrieved > 0 else 0.0
    
    return (
        overall_accuracy,
        result_list,
        retrieval_accuracy,
        generation_accuracy_when_retrieved,
        label_in_retrieved_list
    )



calculate_accuracy_at_level(
    records_test,
    "coicop_pred",
    "code",
    4,
    "list_retrieved_codes"
)


metrics = compute_hierarchical_metrics(records_rag)



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
