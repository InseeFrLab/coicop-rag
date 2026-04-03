import os
import re
import yaml
from typing import List, Tuple, Dict, Optional
import pandas as pd
import unicodedata
import duckdb


def create_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Create a DuckDB in-memory connection configured for S3/MinIO access.
    Returns:
        Configured DuckDB connection.
    """
    con = duckdb.connect(database=":memory:")

    con.execute(f"""
        SET s3_endpoint='{os.getenv("AWS_S3_ENDPOINT")}';
        SET s3_access_key_id='{os.getenv("AWS_ACCESS_KEY_ID")}';
        SET s3_secret_access_key='{os.getenv("AWS_SECRET_ACCESS_KEY")}';
        SET s3_session_token='';
    """)
    return con


def merge_eval_and_retreived(
    df_eval: pd.DataFrame,
    retrieved_codes: pd.DataFrame,
    retrieval_size: int,
    code_name: str,
    col_retrieved_codes_name: str = "list_retrieved_codes"
) -> List[Dict]:
    """
    Merge evaluation predictions with retrieved codes and compute retrieval indicator.

    Combines the wide-format retrieved codes DataFrame (one column per retrieved code)
    into a single list column, joins it onto the evaluation DataFrame on 'id', then
    adds a boolean column indicating whether the ground truth code was retrieved.

    Args:
        df_eval: Evaluation DataFrame containing at least 'id' and the ground truth
            column named by code_name.
        retrieved_codes: DataFrame with 'id' and one numeric string column per
            retrieved code ("0", "1", ..., str(retrieval_size - 1)).
        retrieval_size: Number of retrieved codes per record (number of numeric columns).
        code_name: Name of the ground truth code column in df_eval.
        col_retrieved_codes_name: Name of the list column to create. Default:
            "list_retrieved_codes".

    Returns:
        List of dicts (records) with all evaluation fields, the retrieved codes list,
        and an 'in_retrieved' boolean flag.
    """
    cols = [str(i) for i in range(retrieval_size)]
    retrieved_codes[col_retrieved_codes_name] = retrieved_codes[cols].values.tolist()
    retrieved_codes = retrieved_codes.drop(cols, axis=1)

    df_eval = df_eval.merge(retrieved_codes, how="left", on="id")

    df_eval["in_retrieved"] = df_eval.apply(
        lambda row: row[code_name] in row[col_retrieved_codes_name],
        axis=1
    )

    return df_eval.to_dict('records')


def load_rules(path: str) -> List[Tuple[re.Pattern, str]]:
    """
    Load deterministic coding rules from a YAML file and compile their regex patterns.

    Each rule maps a regex pattern to a COICOP code. Rules are used to bypass
    the RAG pipeline for products that can be coded deterministically.

    Args:
        path: Path to the YAML rules file. Expected structure:
            rules:
              - pattern: "<regex>"
                code: "<coicop_code>"

    Returns:
        List of (compiled_pattern, coicop_code) tuples.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    compiled_rules = []

    for rule in data["rules"]:
        pattern = re.compile(rule["pattern"], re.IGNORECASE)
        compiled_rules.append((pattern, rule["code"]))

    return compiled_rules


def predict_code(product: str, rules: List[Tuple[re.Pattern, str]]) -> tuple[str, str | None]:
    """
    Apply deterministic regex rules to a product description.

    Tries each rule in order and returns on the first full match.
    If no rule matches, the product is routed to the RAG pipeline.

    Args:
        product: Normalized product description string.
        rules: List of (compiled_pattern, coicop_code) tuples from load_rules().

    Returns:
        Tuple (coding_tool, code) where:
        - coding_tool is "regex" if a rule matched, "rag" otherwise.
        - code is the matched COICOP code, or None if no rule matched.
    """
    for pattern, code in rules:
        if pattern.fullmatch(product):
            return "regex", code

    return "rag", None

def normalize(text: str) -> str:
    """
    Normalize a text string by removing diacritics (accents).

    Applies Unicode NFKD decomposition and strips combining characters,
    so that regex rules can match accented product descriptions without
    having to handle accent variants explicitly.

    Args:
        text: Input string, or None.

    Returns:
        Normalized string with diacritics removed, or empty string if input is None.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))

def apply_rules(
    records: List[Dict],
    rules: List[Tuple[re.Pattern, str]],
) -> List[Dict]:
    """
    Apply deterministic coding rules to a list of product records.

    For each record, the product description is normalized and matched against
    the rules. Each record is tagged with a 'coding_tool' field ("regex" or "rag"),
    and matched records additionally get a 'code_rule_predict' field.

    Args:
        records: List of product dicts, each containing at least 'l_pr_product'.
        rules: List of (compiled_pattern, coicop_code) tuples from load_rules().

    Returns:
        Updated list of records with 'coding_tool' and optionally 'code_rule_predict'
        added to each entry.
    """
    records_out = records.copy()

    for entry in records_out:
        product = normalize(entry["l_pr_product"])
        tool, code = predict_code(product, rules)
        entry["coding_tool"] = tool
        if code is not None:
            entry["code_rule_predict"] = code
    return records_out


def truncate_code(code: str, level: int) -> Optional[str]:
    """
    Truncate code to specified hierarchical level
    
    Args:
        code: Full code (e.g., '08.1.2.3.4' or '08.1.6')
        level: Level to truncate to (1-5)
    
    Returns:
        Truncated code or original if already at or below target level,
        None if invalid
    """
    if code is None or not isinstance(code, str) or code == '':
        return None
    
    # Split by dot separator
    parts = code.split('.')
    
    # If code is already at or below target level, return as-is
    if len(parts) <= level:
        return code
    
    # Otherwise truncate to target level
    return '.'.join(parts[:level])

def get_parents(code: str) -> List[str]:
    """
    Get all parent codes for a given code by truncating at each hierarchical level.

    Args:
        code: The full hierarchical code (e.g., '08.1.2.3.4' or '08.1.6')

    Returns:
        List of parent codes, each representing a higher level in the hierarchy.
        Returns empty list if input is invalid or has no parents.
    """
    code_level = len(code.split('.'))
    parents = []
    for level in range(1, code_level):
        parents.append(
            truncate_code(code, level)
        )

    return parents