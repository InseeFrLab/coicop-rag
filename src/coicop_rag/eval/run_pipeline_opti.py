import os
# os.chdir("coicop-rag")
import yaml
import datetime
import uuid
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import duckdb
import pandas as pd
from qdrant_client import QdrantClient
from openai import OpenAI
from langfuse import Langfuse
from data.parsing import extract_json_from_response
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# ============================================================================
# CONFIGURATION
# ============================================================================

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

sample_size = config["annotations"]["sample_size"]

# Database
con = duckdb.connect(database=":memory:")

# Qdrant client
client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"], 
    api_key=os.environ["QDRANT_API_KEY"],
    port=os.environ["QDRANT_API_PORT"]
)

# LLM client (synchronous)
client_llm = OpenAI(
    api_key=os.environ["OLLAMA_API_KEY"],
    base_url=os.environ["OLLAMA_URL"]
)

# Langfuse
prompt_template = Langfuse().get_prompt("prompt-multi-level", label="latest")


# ============================================================================
# OPTIMIZED FUNCTIONS WITH BATCHING (SYNC)
# ============================================================================

def generate_embeddings_batch(
    texts: List[str],
    model: str,
    batch_size: int = 100,
    show_progress: bool = True
) -> List[List[float]]:
    """
    Generate embeddings in batches (synchronous)
    
    Args:
        texts: List of texts to embed
        model: Embedding model name
        batch_size: Number of texts per batch
        show_progress: Show progress bar
    
    Returns:
        List of embeddings
    """
    all_embeddings = []
    
    # Create batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    # Process each batch
    iterator = tqdm(batches, desc="Generating embeddings") if show_progress else batches
    
    for batch in iterator:
        try:
            # Single API call for the whole batch
            response = client_llm.embeddings.create(
                model=model,
                input=batch  # Send entire batch at once
            )
            
            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"\nError processing batch: {e}")
            # Add zero embeddings for failed batch
            embedding_dim = len(all_embeddings[0]) if all_embeddings else 768
            all_embeddings.extend([[0.0] * embedding_dim] * len(batch))
    
    return all_embeddings


def search_qdrant_parallel(
    embeddings: List[List[float]],
    collection_name: str,
    limit: int = 5,
    max_workers: int = 10,
    show_progress: bool = True
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Search Qdrant in parallel using ThreadPoolExecutor
    
    Args:
        embeddings: List of query embeddings
        collection_name: Qdrant collection name
        limit: Number of results per query
        max_workers: Number of parallel threads
        show_progress: Show progress bar
    
    Returns:
        Tuple of (texts_list, codes_list)
    """
    def search_one(embedding: List[float]) -> Tuple[List[str], List[str]]:
        """Search for a single embedding"""
        try:
            points = client_qdrant.query_points(
                collection_name=collection_name,
                query=embedding,
                limit=limit,
            )
            
            points_dump = points.model_dump()["points"]
            texts = [point["payload"]["text"] for point in points_dump]
            codes = [point["payload"]["code"] for point in points_dump]
            
            return texts, codes
            
        except Exception as e:
            print(f"\nError in Qdrant search: {e}")
            return [], []
    
    # Execute searches in parallel
    qdrant_results_texts = []
    qdrant_results_codes = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(search_one, emb) for emb in embeddings]
        
        # Collect results with progress bar
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Vectorial search")
        
        # Store results in order
        results = [None] * len(embeddings)
        future_to_idx = {futures[i]: i for i in range(len(futures))}
        
        for future in iterator:
            idx = future_to_idx[future]
            texts, codes = future.result()
            results[idx] = (texts, codes)
        
        # Separate texts and codes
        for texts, codes in results:
            qdrant_results_texts.append(texts)
            qdrant_results_codes.append(codes)
    
    return qdrant_results_texts, qdrant_results_codes


def generate_llm_responses_parallel(
    messages: List[List[Dict]],
    model: str,
    temperature: float,
    max_tokens: int,
    max_workers: int = 10,
    show_progress: bool = True
) -> List[str]:
    """
    Generate LLM responses in parallel using ThreadPoolExecutor
    
    Args:
        messages: List of message conversations
        model: LLM model name
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        max_workers: Number of parallel threads
        show_progress: Show progress bar
    
    Returns:
        List of raw LLM responses in same order as input
    """
    def generate_one(message: List[Dict]) -> str:
        """Generate response for one message"""
        try:
            response = client_llm.chat.completions.create(
                model=model,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"\nError generating response: {e}")
            return None
    
    # Execute generations in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(generate_one, msg) for msg in messages]
        
        # Collect results with progress bar
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="LLM generation")
        
        # Store results in order
        results = [None] * len(messages)
        future_to_idx = {futures[i]: i for i in range(len(futures))}
        
        for future in iterator:
            idx = future_to_idx[future]
            results[idx] = future.result()
    
    return results


# ============================================================================
# MAIN OPTIMIZED WORKFLOW (SYNCHRONOUS)
# ============================================================================

def main_optimized():
    """Optimized RAG workflow with batching and parallelization (synchronous)"""
    
    total_start = time.time()
    
    # 1. LOAD DATA
    print("="*80)
    print("STEP 1: Loading annotations")
    print("="*80)
    
    query_definition = f"SELECT * FROM read_parquet('{config['annotations']['s3_path']}')"
    annotations = con.sql(query_definition).to_df()
    
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
    
    print(f"✓ Loaded {len(searched_products)} products to code")
    
    # 2. GENERATE EMBEDDINGS (BATCHED)
    print("\n" + "="*80)
    print("STEP 2: Generating embeddings (batched)")
    print("="*80)
    
    step_start = time.time()
    
    product_texts = [p['product'] for p in searched_products]
    
    search_embeddings = generate_embeddings_batch(
        texts=product_texts,
        model=config["embedding"]["model_name"],
        batch_size=config.get("performance", {}).get("embedding_batch_size", 100)
    )
    
    step_time = time.time() - step_start
    print(f"✓ Generated {len(search_embeddings)} embeddings in {step_time:.2f}s")
    print(f"  Embedding dimension: {len(search_embeddings[0])}")
    print(f"  Speed: {len(search_embeddings)/step_time:.1f} embeddings/sec")
    
    # 3. SEARCH QDRANT (PARALLEL)
    print("\n" + "="*80)
    print("STEP 3: Vectorial search in Qdrant (parallel)")
    print("="*80)
    
    step_start = time.time()
    
    qdrant_results_texts, qdrant_results_codes = search_qdrant_parallel(
        embeddings=search_embeddings,
        collection_name=config["qdrant"]["collection_name"],
        limit=config["retrieval"]["size"],
        max_workers=config.get("performance", {}).get("qdrant_max_workers", 10)
    )
    
    step_time = time.time() - step_start
    print(f"✓ Completed {len(qdrant_results_texts)} searches in {step_time:.2f}s")
    print(f"  Results per search: {len(qdrant_results_texts[0])}")
    print(f"  Speed: {len(qdrant_results_texts)/step_time:.1f} searches/sec")
    
    # 4. PREPARE PROMPTS
    print("\n" + "="*80)
    print("STEP 4: Preparing prompts")
    print("="*80)
    
    step_start = time.time()
    
    messages = []
    for i, searched_product in enumerate(searched_products):
        if searched_product.get("enseigne"):
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
    messages = []
    for i, searched_product in enumerate(searched_products):
        if searched_product["enseigne"]:
            enseigne_bloc = f"# Pour information, ce produit a été acheté dans cette enseigne : {searched_product["enseigne"]}"
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
    len(qdrant_results_texts)
    len(qdrant_results_codes)

    step_time = time.time() - step_start
    print(f"✓ Prepared {len(messages)} prompts in {step_time:.2f}s")
    
    # 5. GENERATE LLM RESPONSES (PARALLEL)
    print("\n" + "="*80)
    print("STEP 5: LLM generation (parallel)")
    print("="*80)
    
    step_start = time.time()
    
    llm_responses_raw = generate_llm_responses_parallel(
        messages=messages,
        model=config["llm"]["model_name"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
        max_workers=config.get("performance", {}).get("llm_max_workers", 10)
    )
    
    step_time = time.time() - step_start
    print(f"✓ Generated {len(llm_responses_raw)} responses in {step_time:.2f}s")
    print(f"  Speed: {len(llm_responses_raw)/step_time:.1f} responses/sec")
    
    # 6. PARSE RESPONSES
    print("\n" + "="*80)
    print("STEP 6: Parsing LLM responses")
    print("="*80)
    
    step_start = time.time()
    
    llm_responses_parsed = []
    for response_raw in tqdm(llm_responses_raw, desc="Parsing responses"):
        if response_raw:
            llm_responses_parsed.append(extract_json_from_response(response_raw))
        else:
            llm_responses_parsed.append({"parsed": False})
    
    step_time = time.time() - step_start
    parsed_count = sum(1 for r in llm_responses_parsed if r.get("parsed") != False)
    print(f"✓ Parsed {parsed_count}/{len(llm_responses_parsed)} responses in {step_time:.2f}s")
    print(f"  Success rate: {parsed_count/len(llm_responses_parsed)*100:.1f}%")
    
    # 7. CREATE EVALUATION DATAFRAME
    print("\n" + "="*80)
    print("STEP 7: Creating evaluation dataframe")
    print("="*80)
    
    rows = []
    for i in range(len(llm_responses_parsed)):
        pred = llm_responses_parsed[i]
        annotation = searched_products[i]
        row = pred | annotation
        row["good_pred"] = (row.get("code") == row.get("coicop_pred"))
        row["list_retrieved_codes"] = qdrant_results_codes[i]
        rows.append(row)
    
    df_eval = pd.DataFrame(rows)
    
    print(f"✓ Created evaluation dataframe with {len(df_eval)} rows")
    
    # 8. EXPORT RESULTS
    print("\n" + "="*80)
    print("STEP 8: Exporting results")
    print("="*80)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export evaluation results
    con.sql(f"""
        COPY df_eval 
        TO '{config['predictions']['s3_path'].format(timestamp=timestamp)}'
        (FORMAT PARQUET)
    """)
    
    # Export retrieved codes
    df_retrieved_codes = pd.DataFrame(qdrant_results_codes)
    df_retrieved_codes["id"] = df_eval["id"]
    
    con.sql(f"""
        COPY df_retrieved_codes 
        TO '{config['predictions']['s3_path_retrieved_codes'].format(timestamp=timestamp)}'
        (FORMAT PARQUET)
    """)
    
    print(f"✓ Exported results to S3 (timestamp: {timestamp})")
    
    # PERFORMANCE SUMMARY
    total_time = time.time() - total_start
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"Products processed: {len(searched_products)}")
    print(f"Average time per product: {total_time/len(searched_products):.3f}s")
    print(f"Throughput: {len(searched_products)/total_time:.1f} products/sec")
    
    print("\n" + "="*80)
    print("ALL DONE! 🎉")
    print("="*80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main_optimized()