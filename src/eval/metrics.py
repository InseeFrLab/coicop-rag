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
    filter_type: str,
    confidence_col: str = "confidence",
    threshold: float = 0.7,
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
    elif filter_type == 'threshold':
        return [
            r for r in records 
            if r.get(parsed_col) == True and r.get(codable_col) == True and r.get(confidence_col) >= threshold 
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
    parsed_col: str = "parsed",
    retrieved_col: str = "list_retrieved_codes",
    threshold: float = 0.7,
) -> Dict[str, Dict]:
    """
    Compute hierarchical accuracy metrics with retrieval analysis
    
    Args:
        records: List of dictionnaries
        product_col: Column name for product description
        predicted_col: Column name for predicted code
        label_col: Column name for labeled/ground truth code
        confidence_col: Column name for LLM confidence score
        codable_col: Column name for codability flag (True/False)
        parsed_col: Column name for parsing success flag (True/False)
        retrieved_col: Column name for list of retrieved codes
    
    Returns:
        Dictionary with metrics including retrieval analysis:
        {
            'all_raw': {
                'level_1': float,
                'level_1_retrieval_accuracy': float,
                'level_1_generation_accuracy_when_retrieved': float,
                ...
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
        'parsed_and_codable': {},
        'threshold': {},
    }
    
    # Define filter types
    filter_types = ['all_raw', 'all_parsed', 'codable_only', 'parsed_and_codable', 'threshold']
    
    # Calculate metrics for each filter type
    for filter_type in filter_types:
        # Filter records according to type
        filtered_records = filter_records(
            records, 
            parsed_col, 
            codable_col, 
            filter_type,
            confidence_col,
            threshold
        )
        
        # Store number of samples
        results[filter_type]['n_samples'] = len(filtered_records)
        
        # Calculate accuracy at each hierarchical level
        for level in range(1, 6):
            (
                overall_acc,
                result_list,
                retrieval_acc,
                generation_acc_when_retrieved,
                label_in_retrieved_list
            ) = calculate_accuracy_at_level(
                filtered_records,
                predicted_col,
                label_col,
                level,
                retrieved_col
            )
            
            # Store all metrics
            results[filter_type][f'level_{level}'] = overall_acc
            results[filter_type][f'level_{level}_retrieval_accuracy'] = retrieval_acc
            results[filter_type][f'level_{level}_generation_accuracy_when_retrieved'] = generation_acc_when_retrieved
    
    return results


def print_metrics_report(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted report of the metrics including retrieval analysis
    
    Args:
        metrics: Dictionary returned by compute_hierarchical_metrics
    """
    print("=" * 100)
    print("HIERARCHICAL CLASSIFICATION METRICS WITH RETRIEVAL ANALYSIS")
    print("=" * 100)
    
    for metric_type, values in metrics.items():
        print(f"\n{'─' * 100}")
        print(f"Metric Type: {metric_type.upper().replace('_', ' ')}")
        print(f"{'─' * 100}")
        print(f"Number of samples: {values['n_samples']}")
        print()
        print(f"{'Level':<8} {'Overall Acc':<15} {'Retrieval Acc':<18} {'Gen Acc (Retrieved)':<20}")
        print(f"{'-'*8} {'-'*15} {'-'*18} {'-'*20}")
        
        for level in range(1, 6):
            overall_acc = values[f'level_{level}']
            retrieval_acc = values[f'level_{level}_retrieval_accuracy']
            gen_acc = values[f'level_{level}_generation_accuracy_when_retrieved']
            
            print(
                f"{level:<8} "
                f"{overall_acc:<15.4f} "
                f"{retrieval_acc:<18.4f} "
                f"{gen_acc:<20.4f}"
            )
    
    print("\n" + "=" * 100)


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
                'overall_accuracy': values[f'level_{level}'],
                'retrieval_accuracy': values[f'level_{level}_retrieval_accuracy'],
                'generation_accuracy_when_retrieved': values[f'level_{level}_generation_accuracy_when_retrieved'],
                'n_samples': values['n_samples']
            })
    
    return rows


def analyze_error_sources(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Analyze the proportion of errors due to retrieval vs generation
    
    Args:
        metrics: Dictionary returned by compute_hierarchical_metrics
    
    Returns:
        Dictionary with error analysis for each metric type and level
    """
    error_analysis = {}
    
    for metric_type, values in metrics.items():
        error_analysis[metric_type] = {}
        
        for level in range(1, 6):
            overall_acc = values[f'level_{level}']
            retrieval_acc = values[f'level_{level}_retrieval_accuracy']
            gen_acc_when_retrieved = values[f'level_{level}_generation_accuracy_when_retrieved']
            
            # Calculate error rates
            overall_error_rate = 1 - overall_acc
            
            # Retrieval errors: cases where label is NOT in retrieved codes
            retrieval_error_rate = 1 - retrieval_acc
            
            # Generation errors when retrieved: label IS in retrieved but prediction wrong
            # This is: (retrieval_acc * (1 - gen_acc_when_retrieved))
            generation_error_rate_when_retrieved = retrieval_acc * (1 - gen_acc_when_retrieved)
            
            error_analysis[metric_type][f'level_{level}'] = {
                'overall_error_rate': overall_error_rate,
                'retrieval_error_rate': retrieval_error_rate,
                'generation_error_rate_when_retrieved': generation_error_rate_when_retrieved,
                'retrieval_error_proportion': retrieval_error_rate / overall_error_rate if overall_error_rate > 0 else 0.0,
                'generation_error_proportion': generation_error_rate_when_retrieved / overall_error_rate if overall_error_rate > 0 else 0.0
            }
    
    return error_analysis


def print_error_analysis(error_analysis: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Print error analysis showing proportion of retrieval vs generation errors
    
    Args:
        error_analysis: Dictionary returned by analyze_error_sources
    """
    print("\n" + "=" * 100)
    print("ERROR SOURCE ANALYSIS")
    print("=" * 100)
    
    for metric_type, values in error_analysis.items():
        print(f"\n{'─' * 100}")
        print(f"Metric Type: {metric_type.upper().replace('_', ' ')}")
        print(f"{'─' * 100}")
        print(f"{'Level':<8} {'Overall Err':<15} {'Retrieval Err':<18} {'Generation Err':<18} {'% Retrieval':<15} {'% Generation':<15}")
        print(f"{'-'*8} {'-'*15} {'-'*18} {'-'*18} {'-'*15} {'-'*15}")
        
        for level in range(1, 6):
            level_data = values[f'level_{level}']
            
            print(
                f"{level:<8} "
                f"{level_data['overall_error_rate']:<15.4f} "
                f"{level_data['retrieval_error_rate']:<18.4f} "
                f"{level_data['generation_error_rate_when_retrieved']:<18.4f} "
                f"{level_data['retrieval_error_proportion']:<15.2%} "
                f"{level_data['generation_error_proportion']:<15.2%}"
            )
    
    print("\n" + "=" * 100)

