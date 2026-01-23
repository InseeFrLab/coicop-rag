import os
import yaml
import datetime
import uuid
import logging
from tqdm import tqdm
import duckdb
import pandas as pd
from qdrant_client import QdrantClient
from openai import OpenAI
from langfuse import Langfuse
import mlflow
from data.parsing import extract_json_from_response
from utils import (
    merge_eval_and_retreived, 
    apply_rules
)
from eval.metrics import (
    compute_hierarchical_metrics,
    flatten_metrics,
    write_metrics_report,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'pipeline_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("Starting RAG COICOP pipeline")
    logger.info("="*80)
    
    # Load config
    logger.info("Loading configuration...")

    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded: {config['llm']['model_name']}")
    
    sample_size = config["annotations"]["sample_size"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # MLflow setup
    logger.info("Setting up MLflow...")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"run_{timestamp}"):
        logger.info(f"MLflow run started: {mlflow.active_run().info.run_id}")
        
        # Log parameters
        mlflow.log_params({
            "model_name": config["llm"]["model_name"],
            "embedding_model": config["embedding"]["model_name"],
            "temperature": config["llm"]["temperature"],
            "max_tokens": config["llm"]["max_tokens"],
            "retrieval_size": config["retrieval"]["size"],
            "sample_size": sample_size,
            "prompt_name": config["llm"]["prompt_name"],
            "prompt_version": config["llm"]["prompt_version"],
            "threshold_confidence": config["eval"]["threshold_confidence"]
        })
        
        # Database connection
        logger.info("Connecting to DuckDB...")
        con = duckdb.connect(database=":memory:")
        
        # Qdrant config
        logger.info("Connecting to Qdrant...")
        client_qdrant = QdrantClient(
            url=os.environ["QDRANT_URL"], 
            api_key=os.environ["QDRANT_API_KEY"],
            port=os.environ["QDRANT_API_PORT"]
        )
        logger.info(f"Connected to Qdrant - Collection: {config['qdrant']['collection_name']}")

        # Langfuse config
        lf = Langfuse()
        
        # LLM config
        logger.info("Connecting to LLM...")
        client_llm = OpenAI(
            api_key=os.environ["OLLAMA_API_KEY"],
            base_url=os.environ["OLLAMA_URL"]
        )
        logger.info("LLM client initialized")
        
        # Import prompt template
        logger.info("Loading prompt template...")
        prompt_template = lf.get_prompt(
            config["llm"]["prompt_name"], 
            label=config["llm"]["prompt_version"]
        )
        logger.info(f"Prompt template loaded: {config['llm']['prompt_name']} v{config['llm']['prompt_version']}")
        
        # Import annotations
        logger.info("Loading annotations...")
        mlflow.log_param("input_data_path", config['annotations']['s3_path'])
        query_definition = f"SELECT * FROM read_parquet('{config['annotations']['s3_path']}')"
        annotations = con.sql(query_definition).to_df()
        logger.info(f"Annotations loaded: {len(annotations)} rows")
        
        searched_products = (
            annotations.loc[
                annotations["manual_from_books"],
                ["product", "code", "coicop", "enseigne", "budget"]
            ]
            .assign(id=lambda x: [str(uuid.uuid4()) for _ in range(len(x))])
            .to_dict(orient="records")
        )
        
        if sample_size:
            import random
            random.seed(42)
            searched_products = random.sample(searched_products, sample_size)
            logger.info(f"Sampling applied: {sample_size} products")
        
        num_products = len(searched_products)
        logger.info(f"Number of spendings to code: {num_products}")
        mlflow.log_metric("num_products", num_products)
        
        # Generate embeddings
        logger.info("="*80)
        logger.info("STEP 1: Generating embeddings")
        logger.info("="*80)
                    
        search_embeddings = []

        with lf.start_as_current_span(name="rag_coicop"):
            lf.update_current_trace(
                user_id=os.environ["GIT_USER_NAME"],
                metadata={"service": "stats"}
            )

            with lf.start_as_current_span(name="stage_embedding"):

                for id, searched_product in enumerate(tqdm(searched_products, desc="Generating embeddings")):

                    with lf.start_as_current_span(name="gen_embedding", metadata={"index": id}):

                        response = client_llm.embeddings.create(
                            model=config["embedding"]["model_name"],
                            input=searched_product['product']
                        )
                        lf.update_current_generation(
                            name=searched_product['product'],
                            model=config["embedding"]["model_name"],
                            input=searched_product,
                            metadata={
                                "total_tokens": getattr(response.usage, "total_tokens", None)
                            }
                        )
                        search_embeddings.append(response.data[0].embedding)

                embedding_dim = len(search_embeddings[0])
                logger.info(f"Embeddings generated: {len(search_embeddings)} vectors of dimension {embedding_dim}")
                mlflow.log_params({"embedding_dimension": embedding_dim})
            
            # Vectorial search
            logger.info("="*80)
            logger.info("STEP 2: Vector search")
            logger.info("="*80)
                        
            qdrant_results_texts = []
            qdrant_results_codes = []

            with lf.start_as_current_span(name="stage_vector_search"):
                    
                for id, search_embedding in enumerate(tqdm(search_embeddings, desc="Vector search")):
                    with lf.start_as_current_span(name="Vector search",metadata={"index": id}):
                        points = client_qdrant.query_points(
                            collection_name=config["qdrant"]["collection_name"],
                            query=search_embedding,
                            limit=config["retrieval"]["size"],
                        )

                        topk_text = [point["payload"]["text"] for point in points.model_dump()["points"]]
                        topk_code = [point["payload"]["code"] for point in points.model_dump()["points"]]
                                    
                        qdrant_results_texts.append(topk_text)
                        qdrant_results_codes.append(topk_code)

                        lf.update_current_span(output=topk_code, metadata={"top_k": len(topk_code)})
                    
            logger.info(f"Vector searches completed: {len(qdrant_results_texts)}")
            logger.info(f"Points returned per search: {len(qdrant_results_texts[0])}")
                
            # Generate prompts
            logger.info("="*80)
            logger.info("STEP 3: Preparing prompts")
            logger.info("="*80)
                
            messages = []
            for i, searched_product in enumerate(searched_products):
                if searched_product["enseigne"]:
                    enseigne_bloc = f"# Pour information, ce produit a été acheté dans cette enseigne : {searched_product['enseigne']}"
                else:
                    enseigne_bloc = None
                    
                messages.append(
                    prompt_template.compile(
                        product=searched_product["product"],
                        enseigne_bloc=enseigne_bloc,
                        proposed_codes=qdrant_results_texts[i],
                        list_proposed_codes=qdrant_results_codes[i]
                    )
                )
                        
            logger.info(f"Prompts prepared: {len(messages)}")
            
            # LLM generation
            logger.info("="*80)
            logger.info("STEP 4: LLM generation")
            logger.info("="*80)
                    
            llm_responses = []

            with lf.start_as_current_span(name="stage_llm"):

                for id, message in enumerate(tqdm(messages, desc="LLM generation")):

                    with lf.start_as_current_span(name="gen_llm", metadata={"index": id}):

                        llm_response = client_llm.chat.completions.create(
                            model=config["llm"]["model_name"],
                            messages=message,
                            temperature=config["llm"]["temperature"],
                            max_tokens=config["llm"]["max_tokens"],
                            response_format={"type": "json_object"}
                        )
                        llm_responses.append(llm_response)

                        lf.update_current_generation(
                            name=searched_products[id]["product"],
                            model=config["llm"]["model_name"],
                            model_parameters={
                                "temperature": config['llm']['temperature'],
                                "max_tokens": config['llm']['max_tokens']
                            },
                            input=message,
                            output=llm_response.choices[0].message.content,
                            metadata={
                                "index": id
                            },
                            usage_details={
                                "input_tokens": llm_response.usage.prompt_tokens,
                                "output_tokens": llm_response.usage.completion_tokens,
                                "total_tokens": llm_response.usage.total_tokens
                            }
                        )
                
                logger.info(f"LLM responses generated: {len(llm_responses)}")

            # Parse responses
            logger.info("Parsing LLM responses...")
            llm_responses_parsed = []
            parse_errors = 0
                    
            for llm_response in llm_responses:
                content = llm_response.choices[0].message.content
                try:
                    llm_responses_parsed.append(extract_json_from_response(content))
                except Exception as e:
                    logger.warning(f"Parsing error: {e}")
                    parse_errors += 1
                    llm_responses_parsed.append({})
                        
            logger.info(f"Responses parsed: {len(llm_responses_parsed)} ({parse_errors} errors)")
            mlflow.log_metric("parse_errors", parse_errors)
        
            # Evaluation
            logger.info("="*80)
            logger.info("STEP 5: Evaluation")
            logger.info("="*80)
                        
            rows = []
            for i in range(len(llm_responses_parsed)):
                pred = llm_responses_parsed[i]
                annotation = searched_products[i]
                row = pred | annotation
                row["good_pred"] = (row.get("code") == row.get("coicop_pred"))
                rows.append(row)
                    
            df_eval = pd.DataFrame(rows)
            df_retrieved_codes = pd.DataFrame(qdrant_results_codes)
            df_retrieved_codes.columns = df_retrieved_codes.columns.astype(str)
            df_retrieved_codes["id"] = df_eval["id"]
            
        
            # Export predictions
            logger.info("="*80)
            logger.info("STEP 6: Exporting predictions")
            logger.info("="*80)
                    
            eval_path = config['predictions']['s3_path'].format(timestamp=timestamp)
            retrieved_path = config['predictions']['s3_path_retrieved_codes'].format(timestamp=timestamp)
                        
            con.sql(f"""
                COPY df_eval 
                TO '{eval_path}'
                (FORMAT PARQUET)
            """)
            logger.info(f"Predictions exported: {eval_path}")
                    
            con.sql(f"""
                COPY df_retrieved_codes 
                TO '{retrieved_path}'
                (FORMAT PARQUET)
            """)
            logger.info(f"Retrieved codes exported: {retrieved_path}")

            # Log artifacts to MLflow
            mlflow.log_param("eval_output_path", eval_path)
            mlflow.log_param("retrieved_codes_output_path", retrieved_path)
                
            # Compute metrics
            logger.info("="*80)
            logger.info("STEP 7: Computing metrics")
            logger.info("="*80)
            
            records = merge_eval_and_retreived(
                df_eval=df_eval,
                retrieved_codes=df_retrieved_codes,
                retrieval_size=config["retrieval"]["size"],
            )
                
            # records = apply_rules(
            #     records=records,
            #     path_rules='eval/rules.yaml'
            # )
            records = apply_rules(
                records=records,
                path_rules=config["eval"]["rules_path"]
            )
                    
            records_rag = [record for record in records if record["coding_tool"] == "rag"]
            records_regex = [record for record in records if record["coding_tool"] == "regex"]
                
            logger.info(f"RAG records: {len(records_rag)}")
            logger.info(f"Regex records: {len(records_regex)}")
                
            mlflow.log_metric("num_records_rag", len(records_rag))
            mlflow.log_metric("num_records_regex", len(records_regex))
                
            metrics = compute_hierarchical_metrics(
                records=records_rag,
                threshold=config["eval"]["threshold_confidence"]
            )
                
            metrics_mlflow = flatten_metrics(metrics)

            # Log all metrics to MLflow
            logger.info("Logging metrics to MLflow...")
            for metric_name, metric_value in metrics_mlflow.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"  {metric_name}: {metric_value:.4f}")
                
            # Print metrics report
            logger.info("="*80)
            logger.info("METRICS REPORT")
            logger.info("="*80)
            #print_metrics_report(metrics)
            # print_metrics_report(metrics)
            # report_path = save_metrics_report_as_artifact(metrics, output_path="reports.txt")
            write_metrics_report(metrics, "report.txt")
            mlflow.log_artifact("report.txt", artifact_path="reports")

            # Log config as artifact
            mlflow.log_dict(config, "config.yaml")
        
            logger.info("="*80)
            logger.info("Pipeline completed successfully!")
            logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
            logger.info("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}", exc_info=True)
        raise