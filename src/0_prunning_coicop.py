import os
# os.chdir("coicop-rag/src")
import duckdb
import yaml
from data.pruning import prune_linear_hierarchies

# Config
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get raw notices
con = duckdb.connect(database=":memory:")
query = f"""
    SELECT
        *
    FROM read_csv('{config["coicop"]["path_raw"]}');
"""
notices_raw = duckdb.sql(query).to_df()

# Removing Level 5 (Poste)
notices_raw = notices_raw.loc[notices_raw["type"] != "Poste"]

# Prunning (filter useless codes)
notices_filtered, mapping_table = prune_linear_hierarchies(notices_raw)

# notices_filtered.loc[notices_filtered["code"]=="01.2.1"]
# mapping_table.loc[mapping_table["code_parent_equivalent"]=="01.2.1"]

# Save prunned coicop notices
con.sql(f"""
    COPY notices_filtered 
    TO '{config["coicop"]["path_prunned_lvl4"]}'
    (FORMAT PARQUET)
""")

con.sql(f"""
    COPY mapping_table 
    TO '{config["coicop"]["path_mapping_lvl4"]}'
    (FORMAT PARQUET)
""")
