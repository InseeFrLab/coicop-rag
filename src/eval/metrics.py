from typing import Dict, List, Optional, Tuple

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


def filter_records(
    records: List[Dict],
    parsed_col: str,
    codable_col: str,
    filter_type: str
) -> List[Dict]:
    """
    Filter records based on filter type
    
    Args:
        records: List of all records
        parsed_col: Key name for parsing flag
        codable_col: Key name for codability flag
        filter_type: One of 'all_raw', 'all_parsed', 'codable_only', 'parsed_and_codable'
    
    Returns:
        Filtered list of records
    """
    if filter_type == 'all_raw':
        return records
    
    elif filter_type == 'all_parsed':
        return [r for r in records if r.get(parsed_col) == True]
    
    elif filter_type == 'codable_only':
        return [r for r in records if r.get(codable_col) == True]
    
    elif filter_type == 'parsed_and_codable':
        return [
            r for r in records 
            if r.get(parsed_col) == True and r.get(codable_col) == True
        ]
    
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")


def compute_hierarchical_metrics(
    records: List[Dict],
    product_col: str = "product",
    predicted_col: str = "coicop_pred",
    label_col: str = "code",
    confidence_col: str = "confidence",
    codable_col: str = "codable",
    parsed_col: str = "parsed"
) -> Dict[str, Dict[str, float]]:
    """
    Compute hierarchical accuracy metrics for COICOP/NACE classification
    
    Args:
        df: DataFrame with predictions and labels
        product_col: Column name for product description
        predicted_col: Column name for predicted code (e.g., '08.1.2.3.4')
        label_col: Column name for labeled/ground truth code
        confidence_col: Column name for LLM confidence score
        codable_col: Column name for codability flag (True/False)
        parsed_col: Column name for parsing success flag (True/False)
    
    Returns:
        Dictionary with four metric types, each containing accuracy at 5 levels:
        {
            'all_raw': {
                'level_1': float,
                'level_2': float,
                'level_3': float,
                'level_4': float,
                'level_5': float,
                'n_samples': int
            },
            'all_parsed': {...},
            'codable_only': {...},
            'parsed_and_codable': {...}
        }
    """
    
    # Initialize results dictionary
    results = {
        'all_raw': {},
        'all_parsed': {},
        'codable_only': {},
        'parsed_and_codable': {}
    }
    
    # Define filter types
    filter_types = ['all_raw', 'all_parsed', 'codable_only', 'parsed_and_codable']
    
    # Calculate metrics for each filter type
    for filter_type in filter_types:
        # Filter records according to type
        filtered_records = filter_records(
            records, 
            parsed_col, 
            codable_col, 
            filter_type
        )
        
        # Store number of samples
        results[filter_type]['n_samples'] = len(filtered_records)
        
        # Calculate accuracy at each hierarchical level
        for level in range(1, 6):
            accuracy, _ = calculate_accuracy_at_level(
                filtered_records,
                predicted_col,
                label_col,
                level
            )
            results[filter_type][f'level_{level}'] = accuracy
    
    return results


def print_metrics_report(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted report of the metrics
    
    Args:
        metrics: Dictionary returned by compute_hierarchical_metrics
    """
    print("=" * 80)
    print("HIERARCHICAL CLASSIFICATION METRICS")
    print("=" * 80)
    
    for metric_type, values in metrics.items():
        print(f"\n{'─' * 80}")
        print(f"Metric Type: {metric_type.upper().replace('_', ' ')}")
        print(f"{'─' * 80}")
        print(f"Number of samples: {values['n_samples']}")
        print()
        
        for level in range(1, 6):
            level_key = f'level_{level}'
            accuracy = values[level_key]
            print(f"  Level {level} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n" + "=" * 80)


def export_metrics_to_list(metrics: Dict[str, Dict[str, float]]) -> List[Dict]:
    """
    Export metrics to a list of dictionaries for easy analysis
    
    Args:
        metrics: Dictionary returned by compute_hierarchical_metrics
    
    Returns:
        List of dictionaries with metrics in tabular format
    """
    rows = []
    
    for metric_type, values in metrics.items():
        for level in range(1, 6):
            rows.append({
                'metric_type': metric_type,
                'level': level,
                'accuracy': values[f'level_{level}'],
                'n_samples': values['n_samples']
            })
    
    return rows
