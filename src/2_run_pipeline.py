#!/usr/bin/env python3
"""
RAG COICOP Pipeline
===================
Pipeline for automatic COICOP coding using RAG (Retrieval-Augmented Generation)
"""

import os
# os.chdir("coicop-rag/src")
import yaml
import datetime
import uuid
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import duckdb
import pandas as pd
from qdrant_client import QdrantClient
from openai import OpenAI
from langfuse import Langfuse
import mlflow
import subprocess
import random

from data.parsing import extract_json_from_response, ReponseFormat
from data.pruning import prune_annotation_lvl4, trunc_and_prune_lvl4, _trunc_and_prune_lvl4
from utils import merge_eval_and_retreived, apply_rules, load_rules
from eval.metrics import (
    compute_hierarchical_metrics,
    flatten_metrics,
    write_metrics_report,
)
from generation_tools import generate_llm_responses

# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging():
    """Configure logging with both console and file handlers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'pipeline_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_argument_parser():
    """
    Setup command-line argument parser
    
    Arguments override values from config.yaml when provided
    """
    parser = argparse.ArgumentParser(
        description='RAG COICOP Pipeline - Automatic COICOP coding',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default='src/config.yaml',
        help='Path to config YAML file'
    )
    
    # Sample size override
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of products to sample (overrides config)'
    )
    
    # Model parameters
    parser.add_argument(
        '--model-name',
        type=str,
        help='LLM model name (overrides config)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        help='LLM temperature (overrides config)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        help='Maximum tokens for LLM generation (overrides config)'
    )
    
    # Retrieval parameters
    parser.add_argument(
        '--retrieval-size',
        type=int,
        help='Number of documents to retrieve (overrides config)'
    )
    
    # Data parameters
    parser.add_argument(
        '--collection-name',
        type=str,
        help='Qdrant collection name (overrides config)'
    )
    
    parser.add_argument(
        '--nature-annotation',
        type=str,
        help='Type of annotation to filter (overrides config)'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--threshold-confidence',
        type=float,
        help='Confidence threshold for evaluation (overrides config)'
    )
    
    # MLflow parameters
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='MLflow experiment name (overrides config)'
    )

    parser.add_argument(
        '--noprune',
        action='store_true',
        help='Enable pruning'
    )
    
    return parser


def merge_config_with_args(config, args):
    """
    Merge command-line arguments with config file
    Command-line arguments take precedence over config file values
    
    Args:
        config: Configuration dictionary from YAML
        args: Parsed command-line arguments
        
    Returns:
        dict: Merged configuration
    """
    # Override config values with command-line arguments if provided
    if args.sample_size is not None:
        config['annotations']['sample_size'] = args.sample_size
        
    if args.model_name is not None:
        config['llm']['model_name'] = args.model_name
        
    if args.temperature is not None:
        config['llm']['temperature'] = args.temperature
        
    if args.max_tokens is not None:
        config['llm']['max_tokens'] = args.max_tokens
        
    if args.retrieval_size is not None:
        config['retrieval']['size'] = args.retrieval_size
        
    if args.collection_name is not None:
        config['qdrant']['collection_name'] = args.collection_name
        
    if args.nature_annotation is not None:
        config['annotations']['nature'] = args.nature_annotation
        
    if args.threshold_confidence is not None:
        config['eval']['threshold_confidence'] = args.threshold_confidence
        
    if args.experiment_name is not None:
        config['mlflow']['experiment_name'] = args.experiment_name
    
    if args.noprune is False:
        config['eval']['prune'] = True
    
    return config


# ============================================================================
# Pipeline Steps
# ============================================================================

def initialize_clients(config):
    """
    Initialize connections to external services
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (duckdb_connection, qdrant_client, llm_client)
    """
    logger.info("Initializing external service connections...")
    
    # DuckDB connection
    logger.info("  → Connecting to DuckDB...")
    con = duckdb.connect(database=":memory:")
    
    # Qdrant connection
    logger.info("  → Connecting to Qdrant...")
    client_qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"], 
        api_key=os.environ["QDRANT_API_KEY"],
        port=os.environ["QDRANT_API_PORT"]
    )
    logger.info(f"  → Qdrant collection: {config['qdrant']['collection_name']}")
    
    # LLM connection
    logger.info("  → Connecting to LLM...")
    # client_llm = OpenAI(
    #     api_key=os.environ["OLLAMA_API_KEY"],
    #     base_url=os.environ["OLLAMA_URL"]
    # )
    logger.info("  → Connecting to vLLM generation model...")
    client_vllm_gen = OpenAI(
        base_url=os.environ["VLLM_GENERATION_URL"],
        api_key=os.environ["VLLM_GENERATION_API_KEY"]
    )

    client_vllm_emb = OpenAI(
        base_url=os.environ["VLLM_EMBEDDING_URL"],
        api_key=os.environ["VLLM_EMBEDDING_API_KEY"]
    )

    try:
        models = client_vllm_gen.models.list()
        
        if not models.data:
            raise ValueError("No generation model in vLLM server.")

        server_model_id = models.data[0].id
        expected_model_id = config["llm"]["model_name"]

        if server_model_id != expected_model_id:
            raise ValueError(
                f"Model mismatch : server='{server_model_id}' "
                f"vs config='{expected_model_id}'"
            )

        print("✔ Valid VLLM model with config")

    except KeyError as e:
        print(f"Missing config key : {e}")

    except Exception as e:
        print(f"Error between vllm's model and config : {e}")
    
    try:
        models = client_vllm_emb.models.list()
        
        if not models.data:
            raise ValueError("No embedding model in vLLM server.")

        server_model_id = models.data[0].id
        expected_model_id = config["embedding"]["model_name"]

        if server_model_id != expected_model_id:
            raise ValueError(
                f"Model mismatch : server='{server_model_id}' "
                f"vs config='{expected_model_id}'"
            )

        print("✔ Valid VLLM embedding model with config")

    except KeyError as e:
        print(f"Missing embedding config key : {e}")

    except Exception as e:
        print(f"Error between vllm's embedding model and config : {e}")

    logger.info("✓ All clients initialized successfully")
    
    return con, client_qdrant, client_vllm_gen, client_vllm_emb


def load_prompt_template(config):
    """
    Load prompt template from Langfuse
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Prompt template object
    """
    logger.info("Loading prompt template from Langfuse...")
    
    prompt_template = Langfuse().get_prompt(
        config["llm"]["prompt_name"], 
        version=int(config["llm"]["prompt_version"])
    )
    
    logger.info(
        f"✓ Prompt loaded: {config['llm']['prompt_name']} "
        f"v{config['llm']['prompt_version']}"
    )
    
    return prompt_template


def load_and_prepare_annotations(con, config, mapping_table_lvl4, notices_raw):
    """
    Load annotations from S3 and apply filtering/pruning
    
    Args:
        con: DuckDB connection
        config: Configuration dictionary
        
    Returns:
        list: List of product dictionaries ready for processing
    """
    logger.info("Loading and preparing annotations...")
    
    # Load annotations from S3
    query_definition = f"SELECT * FROM read_parquet('{config['annotations']['s3_path']}')"
    annotations = con.sql(query_definition).to_df()
    
    # Filter by annotation type if specified
    nature_annotation = config["annotations"]["nature"]
    if nature_annotation:
        annotations = annotations.loc[annotations[nature_annotation]]
    
    annotations = annotations[["product", "code", "coicop", "enseigne", "budget"]]
    
    logger.info(
        f"✓ Annotations loaded: {len(annotations)} rows "
        f"(type: {nature_annotation or 'all'})"
    )
    
 
    # Prune annotations (remove children codes in linear relation)
    logger.info("Pruning annotations...")
    annotations = prune_annotation_lvl4(
        annotations, 
        mapping_table_lvl4, 
        notices_raw
    )
    
    # Add unique IDs
    searched_products = (
        annotations
        .assign(id=lambda x: [str(uuid.uuid4()) for _ in range(len(x))])
        .to_dict(orient="records")
    )
    
    # Apply sampling if configured
    sample_size = config["annotations"]["sample_size"]
    if sample_size:
        import random
        random.seed(42)
        searched_products = random.sample(searched_products, sample_size)
        logger.info(f"✓ Sampling applied: {sample_size} products")
    
    logger.info(f"✓ Total products to process: {len(searched_products)}")
    
    return searched_products, nature_annotation


def generate_embeddings(searched_products, client_emb, config):
    """
    Generate embeddings for all product descriptions
    
    Args:
        searched_products: List of product dictionaries
        client_emb: OpenAI client for embedding generation
        config: Configuration dictionary
        
    Returns:
        list: List of embedding vectors
    """
    logger.info("=" * 80)
    logger.info("STEP 1: GENERATING EMBEDDINGS")
    logger.info("=" * 80)
    
    search_embeddings = []
    
    for searched_product in tqdm(searched_products, desc="Generating embeddings"):
        response = client_emb.embeddings.create(
            model=config["embedding"]["model_name"],
            input=searched_product['product']
        )
        search_embeddings.append(response.data[0].embedding)
    
    embedding_dim = len(search_embeddings[0])
    logger.info(
        f"✓ Embeddings generated: {len(search_embeddings)} vectors "
        f"(dimension: {embedding_dim})"
    )
    
    return search_embeddings, embedding_dim


def perform_vector_search(search_embeddings, client_qdrant, config):
    """
    Perform vector search in Qdrant to retrieve relevant documents
    
    Args:
        search_embeddings: List of embedding vectors
        client_qdrant: Qdrant client
        config: Configuration dictionary
        
    Returns:
        tuple: (texts, codes) - Retrieved document texts and COICOP codes
    """
    logger.info("=" * 80)
    logger.info("STEP 2: VECTOR SEARCH IN QDRANT")
    logger.info("=" * 80)
    
    qdrant_results_texts = []
    qdrant_results_codes = []
    
    for search_embedding in tqdm(search_embeddings, desc="Vector search"):
        points = client_qdrant.query_points(
            collection_name=config["qdrant"]["collection_name"],
            query=search_embedding,
            limit=config["retrieval"]["size"],
        )
        
        qdrant_results_texts.append(
            [point["payload"]["text"] for point in points.model_dump()["points"]]
        )
        qdrant_results_codes.append(
            [point["payload"]["code"] for point in points.model_dump()["points"]]
        )
    
    logger.info(
        f"✓ Vector searches completed: {len(qdrant_results_texts)} searches, "
        f"{len(qdrant_results_texts[0])} points per search"
    )
    
    return qdrant_results_texts, qdrant_results_codes


def prepare_prompts(searched_products, qdrant_results_texts, qdrant_results_codes, prompt_template):
    """
    Prepare prompts for LLM generation
    
    Args:
        searched_products: List of product dictionaries
        qdrant_results_texts: Retrieved document texts
        qdrant_results_codes: Retrieved COICOP codes
        prompt_template: Langfuse prompt template
        
    Returns:
        list: List of compiled prompt messages
    """
    logger.info("=" * 80)
    logger.info("STEP 3: PREPARING PROMPTS")
    logger.info("=" * 80)
    
    messages = []
    
    for i, searched_product in enumerate(searched_products):
        # Include store information if available
        if searched_product["enseigne"]:
            enseigne_bloc = (
                f"# Pour information, ce produit a été acheté dans cette enseigne : "
                f"{searched_product['enseigne']}"
            )
        else:
            enseigne_bloc = None
        
        if searched_product["budget"] and isinstance(searched_product["budget"], float):
            price_bloc = (
                f"# Pour information, ce produit a coûté : {searched_product['enseigne']} euros."
            )
        else:
            price_bloc = None
        
        messages.append(
            prompt_template.compile(
                product=searched_product["product"],
                enseigne_bloc=enseigne_bloc,
                price_bloc=price_bloc,
                proposed_codes="\n\n## ".join(qdrant_results_texts[i]),
                list_proposed_codes=qdrant_results_codes[i]
            )
        )
    
    logger.info(f"✓ Prompts prepared: {len(messages)}")
    
    return messages


def log_prompts_sample(messages, n, base_filename: str = "prompts/prompt"):
    n_max = len(messages)
    n = n_max if n > n_max else n
    index = random.sample(range(n_max), n)
    messages_to_log = [messages[m] for m in index]

    for idx, prompt in enumerate(messages_to_log):
        filename = f"{base_filename}_{idx}.md"
        # Concatène le contenu de tous les messages dans le prompt
        text = "\n\n".join(f"### {msg['role'].capitalize()}\n{msg['content']}" for msg in prompt)
        mlflow.log_text(text, filename)


# def generate_llm_responses(messages, client_gen, config):
#     """
#     Generate predictions using LLM
    
#     Args:
#         messages: List of prompt messages
#         client_gen: OpenAI client for generation
#         config: Configuration dictionary
        
#     Returns:
#         list: List of LLM response objects
#     """
#     logger.info("=" * 80)
#     logger.info("STEP 4: LLM GENERATION")
#     logger.info("=" * 80)
    
#     llm_responses = []
    
#     for message in tqdm(messages, desc="LLM generation"):
#         llm_responses.append(
#             client_gen.chat.completions.create(
#                 model=config["llm"]["model_name"],
#                 messages=message,
#                 temperature=config["llm"]["temperature"],
#                 max_tokens=config["llm"]["max_tokens"],
#                 response_format={"type": "json_object"}
#             )
#         )
    
#     logger.info(f"✓ LLM responses generated: {len(llm_responses)}")
    
#     return llm_responses


def parse_llm_responses(llm_responses):
    """
    Parse JSON responses from LLM
    
    Args:
        llm_responses: List of LLM response objects
        
    Returns:
        tuple: (parsed_responses, parse_errors_count)
    """
    logger.info("Parsing LLM responses...")
    
    llm_responses_parsed = []
    
    for llm_response in llm_responses:
        content = llm_response.choices[0].message.content
        try:
            llm_responses_parsed.append(extract_json_from_response(content))
        except Exception as e:
            logger.warning(f"Parsing error: {e}")
            llm_responses_parsed.append({'parsed': False})
    
    parse_errors = sum(dic == {'parsed': False} for dic in llm_responses_parsed)
    
    logger.info(
        f"✓ Responses parsed: {len(llm_responses_parsed)} "
        f"({parse_errors} errors)"
    )
    
    return llm_responses_parsed, parse_errors


def create_evaluation_dataframe(
        llm_responses_parsed,
        searched_products,
        qdrant_results_codes,
        prune,
        mapping_table_lvl4,
    ):
    """
    Create evaluation dataframe combining predictions and ground truth
    
    Args:
        llm_responses_parsed: Parsed LLM responses
        searched_products: Original product data with annotations
        qdrant_results_codes: Retrieved COICOP codes
        
    Returns:
        tuple: (evaluation_df, retrieved_codes_df)
    """
    logger.info("=" * 80)
    logger.info("STEP 5: CREATING EVALUATION DATASET")
    logger.info("=" * 80)
    
    rows = []
    for i in range(len(llm_responses_parsed)):
        pred = llm_responses_parsed[i]
        annotation = searched_products[i]
        row = pred | annotation
        if prune:
            # Trunc and prune LLM's prediction
            row["coicop_pred_tprune"] = _trunc_and_prune_lvl4(
                code=row.get("coicop_pred", None), # None if not parsed
                mapping_table_lvl4=mapping_table_lvl4
            )
            # Trunc and prune annotations
            row["code_tprune"] = _trunc_and_prune_lvl4(
                code=row["code"],
                mapping_table_lvl4=mapping_table_lvl4
            )
        rows.append(row)
    
    df_eval = pd.DataFrame(rows)
    
    if prune: 
        qdrant_results_codes_tprune = [
            [_trunc_and_prune_lvl4(code, mapping_table_lvl4) for code in sublist]
            for sublist in qdrant_results_codes
        ]
        df_retrieved_codes_tprune = pd.DataFrame(qdrant_results_codes_tprune)
        df_retrieved_codes_tprune.columns = df_retrieved_codes_tprune.columns.astype(str)
        df_retrieved_codes_tprune["id"] = df_eval["id"]

    df_retrieved_codes = pd.DataFrame(qdrant_results_codes)
    df_retrieved_codes.columns = df_retrieved_codes.columns.astype(str)
    df_retrieved_codes["id"] = df_eval["id"]
    
    logger.info(f"✓ Evaluation dataset created: {len(df_eval)} rows")

    if prune: 
        return df_eval, df_retrieved_codes, df_retrieved_codes_tprune
    
    return df_eval, df_retrieved_codes
  


def export_predictions(con, df_eval, df_retrieved_codes, config, timestamp):
    """
    Export predictions to S3
    
    Args:
        con: DuckDB connection
        df_eval: Evaluation dataframe
        df_retrieved_codes: Retrieved codes dataframe
        config: Configuration dictionary
        timestamp: Timestamp string for file naming
        
    Returns:
        tuple: (eval_path, retrieved_path)
    """
    logger.info("=" * 80)
    logger.info("STEP 6: EXPORTING PREDICTIONS")
    logger.info("=" * 80)
    
    eval_path = config['predictions']['s3_path'].format(timestamp=timestamp)
    retrieved_path = config['predictions']['s3_path_retrieved_codes'].format(
        timestamp=timestamp
    )
    
    # Export evaluation results
    con.sql(f"""
        COPY df_eval 
        TO '{eval_path}'
        (FORMAT PARQUET)
    """)
    logger.info(f"✓ Predictions exported: {eval_path}")
    
    # Export retrieved codes
    con.sql(f"""
        COPY df_retrieved_codes 
        TO '{retrieved_path}'
        (FORMAT PARQUET)
    """)
    logger.info(f"✓ Retrieved codes exported: {retrieved_path}")
    
    return eval_path, retrieved_path


def get_git_commit_hash():
    """Récupère le hash du commit Git actuel"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()
    except:
        return None


def get_git_branch():
    """Récupère la branche Git actuelle"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).decode('ascii').strip()
    except:
        return None


def compute_and_log_metrics(df_eval, df_retrieved_codes, config, prune, rules):
    """
    Compute evaluation metrics and log to MLflow
    
    Args:
        df_eval: Evaluation dataframe
        df_retrieved_codes: Retrieved codes dataframe
        config: Configuration dictionary
        
    Returns:
        dict: Computed metrics
    """
    logger.info("=" * 80)
    logger.info("STEP 7: COMPUTING METRICS")
    logger.info("=" * 80)
    
    # Merge evaluation and retrieved data
    records = merge_eval_and_retreived(
        df_eval=df_eval,
        retrieved_codes=df_retrieved_codes,
        retrieval_size=config["retrieval"]["size"],
        code_name="code_tprune" if prune else "code",
        col_retrieved_codes_name="list_retrieved_codes",
    )
    
    # # Apply business rules
    # records = apply_rules(
    #     records=records,
    #     rules=rules
    # )
    
    # # Split records by coding tool
    # records_rag = [record for record in records if record["coding_tool"] == "rag"]
    # records_regex = [record for record in records if record["coding_tool"] == "regex"]
    
    # logger.info(f"  → RAG records: {len(records_rag)}")
    # logger.info(f"  → Regex records: {len(records_regex)}")
    
    # mlflow.log_metric("num_records_rag", len(records_rag))
    # mlflow.log_metric("num_records_regex", len(records_regex))
    
    # Compute hierarchical metrics
    metrics = compute_hierarchical_metrics(
        records=records,
        threshold=config["eval"]["threshold_confidence"],
        predicted_col="coicop_pred_tprune" if prune else "coicop_pred",
        label_col="code_tprune" if prune else "code",
        retrieved_col="list_retrieved_codes"
    )
    
    metrics_mlflow = flatten_metrics(metrics)
    
    # Log metrics to MLflow
    logger.info("Logging metrics to MLflow:")
    for metric_name, metric_value in metrics_mlflow.items():
        mlflow.log_metric(metric_name, metric_value)
        logger.info(f"  → {metric_name}: {metric_value:.4f}")
    
    logger.info("✓ Metrics computed and logged")
    
    return metrics


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main pipeline execution"""
    
    logger.info("=" * 80)
    logger.info("STARTING RAG COICOP PIPELINE")
    logger.info("=" * 80)
    
    # ---------------------------------------------------------------------------
    # Parse arguments and load configuration
    # ---------------------------------------------------------------------------
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # config = load_config("config.yaml")
    config = load_config(args.config)
    config = merge_config_with_args(config, args)
    
    logger.info(f"✓ Configuration loaded: {config['llm']['model_name']}")

    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ---------------------------------------------------------------------------
    # Setup MLflow tracking
    # ---------------------------------------------------------------------------
    
    logger.info("Setting up MLflow experiment tracking...")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"run_{timestamp}"):
        logger.info(f"✓ MLflow run started: {mlflow.active_run().info.run_id}")
        mlflow.set_tag("git.commit", get_git_commit_hash())
        mlflow.set_tag("git.branch", get_git_branch())
        mlflow.set_tag("git.repo", "https://github.com/InseeFrLab/coicop-rag")

        
        # Log parameters
        mlflow.log_params({
            "collection_name": config['qdrant']['collection_name'],
            "model_name": config["llm"]["model_name"],
            "embedding_model": config["embedding"]["model_name"],
            "temperature": config["llm"]["temperature"],
            "max_tokens": config["llm"]["max_tokens"],
            "retrieval_size": config["retrieval"]["size"],
            "sample_size": config["annotations"]["sample_size"],
            "prompt_name": config["llm"]["prompt_name"],
            "prompt_version": config["llm"]["prompt_version"],
            "threshold_confidence": config["eval"]["threshold_confidence"],
            "prune": config['eval']['prune']
        })
        
        # -----------------------------------------------------------------------
        # Initialize external service connections
        # -----------------------------------------------------------------------
        
        con, client_qdrant, client_vllm_gen, client_vllm_emb = initialize_clients(config)
        
        # -----------------------------------------------------------------------
        # Load prompt template
        # -----------------------------------------------------------------------
        
        prompt_template = load_prompt_template(config)
        
        # -----------------------------------------------------------------------
        # Load and prepare annotations
        # -----------------------------------------------------------------------
        
        # Load mapping tables for pruning
        mapping_table_lvl4 = con.sql(
            f"SELECT * FROM read_parquet('{config['coicop']['path_mapping_lvl4']}')"
        ).to_df()
        
        notices_raw = con.sql(
            f"SELECT * FROM read_csv('{config['coicop']['path_raw']}')"
        ).to_df()
        
        # Import products to code (annotated)
        mlflow.log_param("input_data_path", config['annotations']['s3_path'])
        searched_products, nature_annotation = load_and_prepare_annotations(con, config, mapping_table_lvl4, notices_raw)
    
        mlflow.log_param("nature_annotation", nature_annotation)
        mlflow.log_metric("num_products", len(searched_products))

        # Import deterministic coding rules
        # rules = load_rules("eval/rules.yaml")
        rules = load_rules(config["eval"]["rules_path"])

        # -----------------------------------------------------------------------
        # Execute main pipeline steps
        # -----------------------------------------------------------------------
        
        # Step 0: Deterministic classification
        
        # Apply business rules
        searched_products = apply_rules(
            records=searched_products,
            rules=rules
        )
        
        # Split records by coding tool
        searched_products_rag = [searched_product for searched_product in searched_products if searched_product["coding_tool"] == "rag"]
        
        # count_None = 0
        # for prod in searched_products_regex:
        #     if prod["product"] is None:
        #         count_None += 1
        
        searched_products_regex = [searched_product for searched_product in searched_products if searched_product["coding_tool"] == "regex"]
        
        logger.info(f"  → RAG records: {len(searched_products_rag)}")
        logger.info(f"  → Regex records: {len(searched_products_regex)}")
        
        mlflow.log_metric("num_records_rag", len(searched_products_rag))
        mlflow.log_metric("num_records_regex", len(searched_products_regex)) 

        # Step 1: Generate embeddings
        search_embeddings, embedding_dim = generate_embeddings(
            searched_products_rag, 
            client_vllm_emb, 
            config
        )
        mlflow.log_param("embedding_dimension", embedding_dim)
        
        # Step 2: Vector search
        qdrant_results_texts, qdrant_results_codes = perform_vector_search(
            search_embeddings,
            client_qdrant,
            config
        )
        
        # Step 3: Prepare prompts
        messages = prepare_prompts(
            searched_products_rag,
            qdrant_results_texts,
            qdrant_results_codes,
            prompt_template
        )

        log_prompts_sample(messages, n=6)
        
        # Step 4: Generate LLM responses
        llm_responses = generate_llm_responses(messages, client_vllm_gen, config)
        
        # Step 5: Parse responses
        llm_responses_parsed, n_parse_errors = parse_llm_responses(llm_responses)
        mlflow.log_metric("parse_errors", n_parse_errors)
        
        # Step 6: Create evaluation dataset
        df_eval, df_retrieved_codes, df_retrieved_codes_tprune = (
            create_evaluation_dataframe(
                    llm_responses_parsed=llm_responses_parsed,
                    searched_products=searched_products_rag,
                    qdrant_results_codes=qdrant_results_codes,
                    prune=config['eval']['prune'],
                    mapping_table_lvl4=mapping_table_lvl4,
                )
        )

        # Step 7: Export RAG predictions
        eval_path, retrieved_path = export_predictions(
            con,
            df_eval,
            df_retrieved_codes,
            config,
            timestamp
        )
        
        mlflow.log_param("eval_output_path", eval_path)
        mlflow.log_param("retrieved_codes_output_path", retrieved_path)
        
        # Step 8: Compute and log metrics

        metrics = compute_and_log_metrics(
            df_eval, 
            df_retrieved_codes_tprune, 
            config, 
            config['eval']['prune'],
            rules,
        )

        
        # -----------------------------------------------------------------------
        # Generate and save metrics report
        # -----------------------------------------------------------------------
        
        logger.info("=" * 80)
        logger.info("GENERATING METRICS REPORT")
        logger.info("=" * 80)
        
        # write_metrics_report(metrics, "report.txt")
        write_metrics_report(
            metrics=metrics,
            output_path="report.txt",
            include_product_types=True,
            include_comparison=True
            )
        
        mlflow.log_artifact("report.txt", artifact_path="reports")
        logger.info("✓ Metrics report saved")
        
        # Log config as artifact
        mlflow.log_dict(config, "config.yaml")
        
        # -----------------------------------------------------------------------
        # Pipeline completion
        # -----------------------------------------------------------------------
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        logger.info("=" * 80)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}", exc_info=True)
        raise