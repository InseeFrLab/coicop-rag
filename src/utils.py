import re
import yaml
from typing import List, Tuple, Dict, Optional
import pandas as pd
import unicodedata


def merge_eval_and_retreived(
    df_eval: pd.DataFrame,
    retrieved_codes: pd.DataFrame,
    retrieval_size: int,
    code_name: str,
    col_retrieved_codes_name: str = "list_retrieved_codes"
    ):

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
    Charge les règles YAML et compile les regex.
    Retourne une liste (pattern compilé, code).
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
    Retourne (coding_tool, code_predict)
    """
    for pattern, code in rules:
        if pattern.fullmatch(product):
            return "regex", code

    return "rag", None

def normalize(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))

def apply_rules(
    records: List[Dict],
    rules,
):
    records_out = records.copy()
    
    for entry in records_out:
        product = normalize(entry["product"])
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