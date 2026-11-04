"""
SpecMatch Experiment 02: Semantic Retrieval Baseline.
Executes end-to-end dense semantic search pipeline:
  1. Loads WANDS raw products, queries, and labels.
  2. Loads sentence-transformers/all-MiniLM-L6-v2 on CPU.
  3. Checks for cached product embeddings (outputs/product_embeddings.npy) or encodes catalog.
  4. Encodes 480 user search queries with L2-normalized embeddings.
  5. Computes cosine similarity matrix (dot product) and retrieves top-10 products per query.
  6. Validates ranking constraints (no duplicates, sorted scores, ranks 1..10).
  7. Evaluates ranking quality with nDCG@10, MRR@10, Recall@10, and Judged Coverage.
  8. Saves outputs/semantic_results.csv and updates outputs/baseline_comparison.csv.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import OUTPUTS_DIR
from src.data_loader import load_all_data
from src.semantic_retrieval import (
    load_semantic_model,
    get_or_create_product_embeddings,
    encode_queries,
    semantic_retrieve,
    save_semantic_results,
    validate_semantic_results,
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIM
)
from src.evaluation import evaluate_retrieval


def run_semantic_experiment(
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 128,
    top_k: int = 10,
    force_recompute: bool = False
):
    print("=" * 70)
    print("SPECMATCH EXPERIMENT 02: SEMANTIC RETRIEVAL BASELINE")
    print(f"Model: {model_name} (CPU)")
    print("=" * 70)
    total_start = time.perf_counter()
    
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    print("\n[Step 1/5] Loading WANDS dataset...")
    products_df, queries_df, labels_df = load_all_data(validate=True)
    t_load = time.perf_counter() - t0
    print(f"Loaded: {len(products_df):,} products, {len(queries_df):,} queries, {len(labels_df):,} labels ({t_load:.2f}s).")
    
    # ---------------------------------------------------------
    # 2. Load Model
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    print("\n[Step 2/5] Initializing SentenceTransformer model...")
    model = load_semantic_model(model_name=model_name, device="cpu")
    t_model = time.perf_counter() - t0
    
    # ---------------------------------------------------------
    # 3. Product Embeddings (Check Cache or Compute)
    # ---------------------------------------------------------
    print(f"\n[Step 3/5] Resolving product embeddings (dim={EMBEDDING_DIM})...")
    product_embs, product_pids, enc_time = get_or_create_product_embeddings(
        products_df=products_df,
        model=model,
        batch_size=batch_size,
        force_recompute=force_recompute
    )
    
    # ---------------------------------------------------------
    # 4. Query Embeddings & Cosine Retrieval
    # ---------------------------------------------------------
    print(f"\n[Step 4/5] Encoding queries and retrieving top-{top_k} products...")
    query_embs, query_ids, q_enc_time = encode_queries(queries_df, model)
    
    results_df, retrieval_time = semantic_retrieve(
        query_embeddings=query_embs,
        query_ids=query_ids,
        product_embeddings=product_embs,
        product_ids=product_pids,
        top_k=top_k
    )
    
    # Validate result constraints
    validate_semantic_results(results_df, total_queries=len(queries_df), top_k=top_k)
    
    # Save outputs/semantic_results.csv
    output_csv_path = OUTPUTS_DIR / "semantic_results.csv"
    save_semantic_results(results_df, output_csv_path)
    
    # ---------------------------------------------------------
    # 5. Evaluate Performance
    # ---------------------------------------------------------
    print(f"\n[Step 5/5] Evaluating semantic retrieval quality against WANDS ground truth...")
    t0 = time.perf_counter()
    eval_max = evaluate_retrieval(results_df, labels_df, top_k=top_k, policy='max')
    eval_min = evaluate_retrieval(results_df, labels_df, top_k=top_k, policy='min')
    t_eval = time.perf_counter() - t0
    
    total_time = time.perf_counter() - total_start
    
    # ---------------------------------------------------------
    # Report Findings
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("SEMANTIC BASELINE EVALUATION REPORT")
    print("=" * 70)
    print(f"Model:                           {model_name}")
    print(f"Embedding Dimension:             {EMBEDDING_DIM}")
    print(f"Products Encoded:                {len(product_pids):,}")
    print(f"Queries Evaluated:               {eval_max['num_queries_evaluated']}")
    print(f"Retrieved Results per Query:     {top_k}")
    print(f"Mean Judged Coverage @ 10:       {eval_max['judged_coverage@10_mean'] * 100:.2f}%")
    print(f"Queries with 0 Exact Judgements: {eval_max['queries_with_zero_exact']} ({eval_max['queries_with_zero_exact_pct']:.2f}%)")
    print("-" * 70)
    print("METRICS (Default 'max' aggregation policy):")
    print(f"  nDCG@{top_k}:                    {eval_max['ndcg@10_mean']:.4f}  (std: {eval_max['ndcg@10_std']:.4f})")
    print(f"  MRR@{top_k} (Exact only):        {eval_max['mrr@10_exact']:.4f}")
    print(f"  MRR@{top_k} (Any Relevant):      {eval_max['mrr@10_any_relevant']:.4f}")
    print(f"  Recall@{top_k} (Exact only):     {eval_max['recall@10_exact']:.4f}  [evaluated on {eval_max['recall@10_exact_num_valid_queries']} queries with Exact items]")
    print(f"  Recall@{top_k} (Any Relevant):   {eval_max['recall@10_any_relevant']:.4f}  [evaluated on {eval_max['recall@10_any_num_valid_queries']} queries with Relevant items]")
    print("-" * 70)
    print("EXECUTION TIMING BREAKDOWN:")
    print(f"  Data Loading:                  {t_load:.2f}s")
    print(f"  Model Initialization:          {t_model:.2f}s")
    print(f"  Product Embedding Time:        {enc_time:.2f}s (cached: {enc_time == 0.0})")
    print(f"  Query Encoding Time:           {q_enc_time:.3f}s")
    print(f"  Cosine Retrieval Time:         {retrieval_time:.3f}s ({retrieval_time/len(queries_df)*1000:.2f} ms/query)")
    print(f"  Evaluation Time:               {t_eval:.2f}s")
    print(f"  Total Pipeline Time:           {total_time:.2f}s")
    print("=" * 70)
    
    # ---------------------------------------------------------
    # Sample Retrieval Inspection
    # ---------------------------------------------------------
    print("\nSAMPLE RETRIEVAL RESULTS (First 3 Queries):")
    sample_qids = queries_df['query_id'].iloc[:3].tolist()
    for sqid in sample_qids:
        q_text = queries_df[queries_df['query_id'] == sqid]['query'].iloc[0]
        q_class = queries_df[queries_df['query_id'] == sqid]['query_class'].iloc[0]
        print(f"\nQuery {sqid}: \"{q_text}\" (Class: {q_class})")
        q_res = results_df[results_df['query_id'] == sqid].sort_values('rank').head(5)
        for _, r in q_res.iterrows():
            pid = int(r['product_id'])
            p_name = products_df[products_df['product_id'] == pid]['product_name'].iloc[0]
            match_lbl = labels_df[(labels_df['query_id'] == sqid) & (labels_df['product_id'] == pid)]
            lbl_str = match_lbl['label'].iloc[0] if len(match_lbl) > 0 else "UNJUDGED"
            print(f"  Rank {int(r['rank'])}: Product {pid:5d} | Sim: {r['score']:.4f} | Label: {lbl_str:10s} | {p_name[:65]}")
            
    # ---------------------------------------------------------
    # Update baseline_comparison.csv
    # ---------------------------------------------------------
    comp_path = OUTPUTS_DIR / "baseline_comparison.csv"
    new_record = {
        'model': 'all-MiniLM-L6-v2',
        'retrieval_type': 'dense_semantic',
        'corpus_fields': 'product_name, category_hierarchy, product_features, product_description',
        'k1': np.nan,
        'b': np.nan,
        'ndcg@10': round(eval_max['ndcg@10_mean'], 4),
        'mrr@10_exact': round(eval_max['mrr@10_exact'], 4),
        'mrr@10_any': round(eval_max['mrr@10_any_relevant'], 4),
        'recall@10_exact': round(eval_max['recall@10_exact'], 4),
        'recall@10_any': round(eval_max['recall@10_any_relevant'], 4),
        'judged_coverage@10': round(eval_max['judged_coverage@10_mean'] * 100, 2),
        'total_latency_sec': round(total_time, 2)
    }
    
    if comp_path.exists():
        existing_df = pd.read_csv(comp_path)
        # Avoid duplicate model entries
        existing_df = existing_df[existing_df['model'] != new_record['model']]
        updated_df = pd.concat([existing_df, pd.DataFrame([new_record])], ignore_index=True)
    else:
        updated_df = pd.DataFrame([new_record])
        
    updated_df.to_csv(comp_path, index=False)
    print(f"\nUpdated baseline comparison matrix at: {comp_path}")
    print(updated_df[['model', 'retrieval_type', 'ndcg@10', 'mrr@10_exact', 'recall@10_exact', 'judged_coverage@10']])
    
    return results_df, eval_max


if __name__ == "__main__":
    run_semantic_experiment()

