import os
os.chdir("coicop-rag")
import duckdb
import pandas as pd
import src.eval.metrics
con = duckdb.connect(database=":memory:")

s3_path = "s3://projet-budget-famille/data/rag/eval_test.parquet_20251220_154926.parquet"
query_definition = f"SELECT * FROM read_parquet('{s3_path}')"
df_eval = con.sql(query_definition).to_df()

df_eval.columns
df_eval["good_pred"].mean()
df_eval["parsed"].value_counts()
df_eval["parsed"].dtype
df_eval["codable"].dtype

truncate_code("01.2.3.0.7", level=5)


# Compute metrics
metrics = compute_hierarchical_metrics(df_eval)

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