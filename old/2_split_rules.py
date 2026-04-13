"""
Rule-based Split
================
Splits pruned annotations into two populations:
- Products that can be coded deterministically via regex rules
- Products that need the RAG pipeline

Must be run after 0_prune_annotations.py.
"""
import argparse
import logging

import pandas as pd
import yaml

from coicop_rag.utils import apply_rules, create_duckdb_connection, load_rules


def main():
    parser = argparse.ArgumentParser(description="Split annotations by coding method")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("STARTING RULE-BASED SPLIT")
    logger.info("=" * 80)

    con = create_duckdb_connection()

    # ------------------------------------------------------------------
    # Load pruned annotations
    # ------------------------------------------------------------------

    logger.info("Loading pruned annotations...")
    annotations = con.sql(
        f"SELECT * FROM read_parquet('{config['annotations']['s3_path_pruned']}')"
    ).to_df()
    logger.info(f"  → {len(annotations)} annotations loaded")

    # ------------------------------------------------------------------
    # Apply deterministic rules
    # ------------------------------------------------------------------

    logger.info("Applying deterministic rules...")
    rules = load_rules(config["eval"]["rules_path"])
    records = annotations.to_dict(orient="records")
    records = apply_rules(records, rules)

    df = pd.DataFrame(records)

    df_regex = df[df["coding_tool"] == "regex"].copy()
    df_rag = df[df["coding_tool"] == "rag"].copy()

    logger.info(f"  → Regex: {len(df_regex)} products")
    logger.info(f"  → RAG:   {len(df_rag)} products")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    logger.info("Exporting splits...")

    path_regex = config["annotations"]["s3_path_regex"]
    con.sql(f"COPY df_regex TO '{path_regex}' (FORMAT PARQUET)")
    logger.info(f"  → Regex annotations exported: {path_regex}")

    path_rag = config["annotations"]["s3_path_rag"]
    con.sql(f"COPY df_rag TO '{path_rag}' (FORMAT PARQUET)")
    logger.info(f"  → RAG annotations exported: {path_rag}")

    logger.info("=" * 80)
    logger.info("RULE-BASED SPLIT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main()
