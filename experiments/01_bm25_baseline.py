"""
SpecMatch Experiment 01: BM25 Retrieval Baseline.
Executes end-to-end BM25 pipeline:
  1. Loads WANDS raw products, queries, and labels.
  2. Builds rich searchable product text from name, features, description, and hierarchy.
  3. Tokenizes corpus and queries with consistent alphanumeric tokenizer.
  4. Fits BM25Okapi index and retrieves top-10 products for all 480 queries.
  5. Validates ranking constraints (no duplicates, sorted scores, ranks 1..10).
  6. Evaluates ranking quality with nDCG@10, MRR@10, Recall@10, and Judged Coverage.
  7. Assesses impact of duplicate label aggregation policies ('max' vs 'min').
  8. Saves outputs/bm25_results.csv and outputs/baseline_comparison.csv.
"""

import sys
import time
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import OUTPUTS_DIR
from src.data_loader import load_all_data
from src.preprocessing import prepare_corpus, prepare_query
from src.bm25_retrieval import BM25Retriever, save_retrieval_results, validate_results_dataframe
from src.evaluation import evaluate_retrieval


def run_bm25_experiment(
    k1: float = 1.5,
    b: float = 0.75,
    top_k: int = 10
):
    print("=" * 70)
    print("SPECMATCH EXPERIMENT 01: BM25 RETRIEVAL BASELINE")
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
    # 2. Preprocess & Tokenize Corpus
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    print("\n[Step 2/5] Preprocessing product corpus (name, features, description, hierarchy)...")
    tokenized_corpus, corpus_pids, pid_map = prepare_corpus(products_df)
    t_prep = time.perf_counter() - t0
    print(f"Tokenized {len(tokenized_corpus):,} documents in {t_prep:.2f}s (avg {sum(len(d) for d in tokenized_corpus)/len(tokenized_corpus):.1f} tokens/doc).")
    
    # ---------------------------------------------------------
    # 3. Build BM25 Index
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    print(f"\n[Step 3/5] Indexing with BM25Okapi (k1={k1}, b={b})...")
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.fit(tokenized_corpus, corpus_pids)
    t_index = time.perf_counter() - t0
    
    # ---------------------------------------------------------
    # 4. Execute Retrieval for All 480 Queries
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    print(f"\n[Step 4/5] Retrieving top-{top_k} products across {len(queries_df)} queries...")
    results_df = retriever.retrieve_all(queries_df, top_k=top_k)
    t_retrieve = time.perf_counter() - t0
    
    # Validate result constraints
    validate_results_dataframe(results_df, total_queries=len(queries_df), top_k=top_k)
    
    # Save outputs/bm25_results.csv
    output_csv_path = OUTPUTS_DIR / "bm25_results.csv"
    save_retrieval_results(results_df, output_csv_path)
    
    # ---------------------------------------------------------
    # 5. Evaluate Performance
    # ---------------------------------------------------------
    print(f"\n[Step 5/5] Evaluating retrieval quality against WANDS ground truth...")
    t0 = time.perf_counter()
    
    # Evaluate with standard 'max' policy
    eval_max = evaluate_retrieval(results_df, labels_df, top_k=top_k, policy='max')
    
    # Also evaluate with 'min' policy to measure sensitivity to annotator disagreements
    eval_min = evaluate_retrieval(results_df, labels_df, top_k=top_k, policy='min')
    t_eval = time.perf_counter() - t0
    
    total_time = time.perf_counter() - total_start
    
    # ---------------------------------------------------------
    # Report Findings
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("BM25 BASELINE EVALUATION REPORT")
    print("=" * 70)
    print(f"Queries Evaluated:               {eval_max['num_queries_evaluated']}")
    print(f"Retrieved Results per Query:     {top_k}")
    print(f"Total Products Retrieved:        {len(results_df):,}")
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
    print("POLICY SENSITIVITY CHECK ('min' conservative policy):")
    print(f"  nDCG@{top_k} ('min'):            {eval_min['ndcg@10_mean']:.4f}  (delta vs max: {eval_min['ndcg@10_mean'] - eval_max['ndcg@10_mean']:+.5f})")
    print(f"  MRR@{top_k} Exact ('min'):       {eval_min['mrr@10_exact']:.4f}")
    print("-" * 70)
    print("EXECUTION TIMING BREAKDOWN:")
    print(f"  Data Loading:                  {t_load:.2f}s")
    print(f"  Corpus Tokenization:           {t_prep:.2f}s")
    print(f"  BM25 Index Building:           {t_index:.2f}s")
    print(f"  Query Retrieval (480 queries): {t_retrieve:.2f}s ({t_retrieve/len(queries_df)*1000:.2f} ms/query)")
    print(f"  Evaluation:                    {t_eval:.2f}s")
    print(f"  Total Execution Time:          {total_time:.2f}s")
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
            # Check ground truth label
            match_lbl = labels_df[(labels_df['query_id'] == sqid) & (labels_df['product_id'] == pid)]
            lbl_str = match_lbl['label'].iloc[0] if len(match_lbl) > 0 else "UNJUDGED"
            print(f"  Rank {int(r['rank'])}: Product {pid:5d} | Score: {r['score']:6.2f} | Label: {lbl_str:10s} | {p_name[:65]}")
            
    # ---------------------------------------------------------
    # Save baseline_comparison.csv
    # ---------------------------------------------------------
    comparison_records = [
        {
            'model': 'BM25Okapi',
            'retrieval_type': 'lexical',
            'corpus_fields': 'product_name, category_hierarchy, product_features, product_description',
            'k1': k1,
            'b': b,
            'ndcg@10': round(eval_max['ndcg@10_mean'], 4),
            'mrr@10_exact': round(eval_max['mrr@10_exact'], 4),
            'mrr@10_any': round(eval_max['mrr@10_any_relevant'], 4),
            'recall@10_exact': round(eval_max['recall@10_exact'], 4),
            'recall@10_any': round(eval_max['recall@10_any_relevant'], 4),
            'judged_coverage@10': round(eval_max['judged_coverage@10_mean'] * 100, 2),
            'total_latency_sec': round(total_time, 2)
        }
    ]
    comparison_df = pd.DataFrame(comparison_records)
    comp_path = OUTPUTS_DIR / "baseline_comparison.csv"
    comparison_df.to_csv(comp_path, index=False)
    print(f"\nSaved initial baseline benchmark to: {comp_path}")
    
    return results_df, eval_max


if __name__ == "__main__":
    run_bm25_experiment()

