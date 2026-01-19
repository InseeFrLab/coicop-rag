import re
import yaml
from typing import List, Tuple, Dict
import pandas as pd
import unicodedata


def merge_eval_and_retreived(
    df_eval: pd.DataFrame,
    retrieved_codes: pd.DataFrame,
    retrieval_size: int
    ):

    cols = [str(i) for i in range(retrieval_size)]
    retrieved_codes["list_retrieved_codes"] = retrieved_codes[cols].values.tolist()
    retrieved_codes = retrieved_codes.drop(cols, axis=1)

    df_eval = df_eval.merge(retrieved_codes, how="left", on="id")

    df_eval["in_retrieved"] = df_eval.apply(
        lambda row: row["code"] in row["list_retrieved_codes"],
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
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))

def apply_rules(
    records: List[Dict],
    path_rules: str = "src/eval/rules.yaml"
):
    records_out = records.copy()
    rules = load_rules(path_rules)
    for entry in records_out:
        product = normalize(entry["product"])
        tool, code = predict_code(product, rules)
        entry["coding_tool"] = tool
        if code is not None:
            entry["code_predict"] = code
    return records_out