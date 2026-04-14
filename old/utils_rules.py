import re
import unicodedata
import yaml
from typing import List, Tuple, Dict


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
