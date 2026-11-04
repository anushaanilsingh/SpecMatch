"""
03_hybrid_baseline.py
---------------------
End-to-end Hybrid Retrieval experiment using Reciprocal Rank Fusion (RRF).

Fuses BM25 and Semantic rankings from pre-computed output files.
Evaluates using the same methodology as experiments 01 and 02.

Run from NLP/ directory:
    $env:PYTHONPATH="SpecMatch"; & "SpecMatch\\.venv\\Scripts\\python.exe" "SpecMatch\\experiments\\03_hybrid_baseline.py"
"""

import sys
import os
import time

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import OUTPUTS_DIR
from src.data_loader import load_labels
from src.evaluation import evaluate_retrieval
from src.hybrid_retrieval import (
    load_rrf_inputs,
    compute_rrf,
    validate_hybrid_results,
    save_hybrid_results,
    DEFAULT_K,
    DEFAULT_TOP_K,
    HYBRID_RESULTS_PATH,
    BM25_RESULTS_PATH,
    SEMANTIC_RESULTS_PATH,
)

BASELINE_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "baseline_comparison.csv")


# ===========================================================================
# Helper: per-query nDCG comparison
# ===========================================================================

def per_query_ndcg(results_df: pd.DataFrame, labels_df: pd.DataFrame, k: int = 10) -> dict:
    """
    Compute per-query nDCG@k.

    Returns dict: query_id -> nDCG@k score.
    """
    from src.evaluation import aggregate_labels
    import math

    relevance_map, _ = aggregate_labels(labels_df, policy="max")
    per_query: dict = {}

    for qid, grp in results_df.groupby("query_id"):
        grp_sorted = grp.sort_values("rank").head(k)
        ranked_pids = grp_sorted["product_id"].tolist()

        # DCG
        dcg = 0.0
        for i, pid in enumerate(ranked_pids):
            rel = relevance_map.get((int(qid), int(pid)), 0)
            dcg += rel / math.log2(i + 2)

        # Ideal DCG: best possible from all judged products for this query
        all_rels = [
            v for (q, _p), v in relevance_map.items() if q == int(qid)
        ]
        ideal_rels = sorted(all_rels, reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

        per_query[int(qid)] = (dcg / idcg) if idcg > 0 else 0.0

    return per_query


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_total_start = time.time()

    print("=" * 65)
    print("SpecMatch — Experiment 03: Hybrid RRF Baseline")
    print(f"RRF k={DEFAULT_K}, top-k={DEFAULT_TOP_K}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load BM25 and Semantic result files
    # ------------------------------------------------------------------
    print("\n[1/6] Loading existing retrieval results...")
    bm25_df, semantic_df = load_rrf_inputs(BM25_RESULTS_PATH, SEMANTIC_RESULTS_PATH)

    # Snapshot row counts to verify files are not modified later
    bm25_rows_before = len(bm25_df)
    semantic_rows_before = len(semantic_df)

    # ------------------------------------------------------------------
    # 2. Compute RRF fusion
    # ------------------------------------------------------------------
    print(f"\n[2/6] Computing RRF fusion (k={DEFAULT_K})...")
    t_fusion_start = time.time()
    hybrid_df = compute_rrf(bm25_df, semantic_df, k=DEFAULT_K, top_k=DEFAULT_TOP_K)
    t_fusion = time.time() - t_fusion_start
    print(f"Fusion time: {t_fusion:.2f}s")

    # ------------------------------------------------------------------
    # 3. Validate results
    # ------------------------------------------------------------------
    print("\n[3/6] Validating hybrid results...")
    valid = validate_hybrid_results(hybrid_df, expected_queries=480, top_k=DEFAULT_TOP_K)
    if not valid:
        print("\n[ERROR] Validation failed. Aborting.")
        sys.exit(1)
    print("Validation PASSED.")

    # ------------------------------------------------------------------
    # 4. Save hybrid results
    # ------------------------------------------------------------------
    print(f"\n[4/6] Saving hybrid results...")
    save_hybrid_results(hybrid_df, HYBRID_RESULTS_PATH)

    # Verify originals are still intact
    bm25_check = pd.read_csv(BM25_RESULTS_PATH)
    semantic_check = pd.read_csv(SEMANTIC_RESULTS_PATH)
    assert len(bm25_check) == bm25_rows_before, "BM25 results file was modified!"
    assert len(semantic_check) == semantic_rows_before, "Semantic results file was modified!"
    print("Original BM25 and Semantic result files verified UNCHANGED.")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print(f"\n[5/6] Evaluating hybrid retrieval...")
    labels_df = load_labels()

    t_eval_start = time.time()
    hybrid_metrics = evaluate_retrieval(hybrid_df, labels_df, top_k=10, policy="max")
    t_eval = time.time() - t_eval_start
    print(f"Evaluation time: {t_eval:.2f}s")

    # ------------------------------------------------------------------
    # 6. Update baseline comparison
    # ------------------------------------------------------------------
    print(f"\n[6/6] Updating baseline comparison...")
    hybrid_row = {
        "model": "Hybrid (RRF)",
        **hybrid_metrics,
    }

    if os.path.exists(BASELINE_COMPARISON_PATH):
        comparison_df = pd.read_csv(BASELINE_COMPARISON_PATH)
        # Remove any existing hybrid row to avoid duplicates
        comparison_df = comparison_df[comparison_df["model"] != "Hybrid (RRF)"]
        comparison_df = pd.concat(
            [comparison_df, pd.DataFrame([hybrid_row])], ignore_index=True
        )
    else:
        comparison_df = pd.DataFrame([hybrid_row])

    comparison_df.to_csv(BASELINE_COMPARISON_PATH, index=False)
    print(f"Updated -> {BASELINE_COMPARISON_PATH}")

    # ------------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("HYBRID RRF METRICS")
    print("=" * 65)
    for metric, value in hybrid_metrics.items():
        if isinstance(value, float):
            print(f"  {metric:<35} {value:.4f}")
        else:
            print(f"  {metric:<35} {value}")

    # ------------------------------------------------------------------
    # Three-way comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("BASELINE COMPARISON (all three systems)")
    print("=" * 65)
    print(comparison_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Per-query improvement analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PER-QUERY ANALYSIS (nDCG@10)")
    print("=" * 65)

    bm25_per_q = per_query_ndcg(bm25_df, labels_df, k=10)
    sem_per_q = per_query_ndcg(semantic_df, labels_df, k=10)
    hyb_per_q = per_query_ndcg(hybrid_df, labels_df, k=10)

    all_qids = sorted(set(bm25_per_q) | set(sem_per_q) | set(hyb_per_q))

    hybrid_beats_both = 0
    hybrid_beats_one = 0
    hybrid_between = 0
    hybrid_worse_than_both = 0
    hybrid_beats_bm25 = 0
    hybrid_beats_sem = 0
    improvements = []
    degradations = []

    for qid in all_qids:
        b = bm25_per_q.get(qid, 0.0)
        s = sem_per_q.get(qid, 0.0)
        h = hyb_per_q.get(qid, 0.0)
        delta_vs_bm25 = h - b
        delta_vs_sem = h - s

        if h > b:
            hybrid_beats_bm25 += 1
        if h > s:
            hybrid_beats_sem += 1

        if h > b and h > s:
            hybrid_beats_both += 1
            improvements.append((qid, h, b, s, h - max(b, s)))
        elif h < b and h < s:
            hybrid_worse_than_both += 1
            degradations.append((qid, h, b, s, h - max(b, s)))
        else:
            hybrid_between += 1

    pct = lambda n: f"{n:3d} / {len(all_qids)} ({100*n/len(all_qids):.1f}%)"
    print(f"  Hybrid beats BOTH baselines    : {pct(hybrid_beats_both)}")
    print(f"  Hybrid between the two         : {pct(hybrid_between)}")
    print(f"  Hybrid worse than BOTH         : {pct(hybrid_worse_than_both)}")
    print(f"  Hybrid > BM25 (regardless)     : {pct(hybrid_beats_bm25)}")
    print(f"  Hybrid > Semantic (regardless) : {pct(hybrid_beats_sem)}")

    # Top 5 improvements
    if improvements:
        improvements.sort(key=lambda x: x[4], reverse=True)
        print("\n  Top-5 queries where Hybrid gains most over best baseline:")
        print(f"  {'qid':>6}  {'Hybrid':>7}  {'BM25':>7}  {'Sem':>7}  {'D vs best':>10}")
        for qid, h, b, s, delta in improvements[:5]:
            print(f"  {qid:>6}  {h:>7.4f}  {b:>7.4f}  {s:>7.4f}  {delta:>+10.4f}")

    # Top 5 degradations
    if degradations:
        degradations.sort(key=lambda x: x[4])
        print("\n  Top-5 queries where Hybrid loses most vs best baseline:")
        print(f"  {'qid':>6}  {'Hybrid':>7}  {'BM25':>7}  {'Sem':>7}  {'D vs best':>10}")
        for qid, h, b, s, delta in degradations[:5]:
            print(f"  {qid:>6}  {h:>7.4f}  {b:>7.4f}  {s:>7.4f}  {delta:>+10.4f}")

    # ------------------------------------------------------------------
    # Sample hybrid results with labels
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SAMPLE HYBRID RESULTS (first 3 queries, top 5 each)")
    print("=" * 65)
    from src.evaluation import aggregate_labels
    rel_map, _ = aggregate_labels(labels_df, policy="max")
    sample_qids = sorted(hybrid_df["query_id"].unique())[:3]
    for qid in sample_qids:
        grp = hybrid_df[hybrid_df["query_id"] == qid].sort_values("rank").head(5)
        print(f"\n  Query {qid}:")
        print(f"  {'Rank':>4}  {'ProductID':>10}  {'RRF Score':>10}  {'Rel Label':>10}")
        for _, row in grp.iterrows():
            rel = rel_map.get((qid, int(row["product_id"])), "unjudged")
            print(
                f"  {int(row['rank']):>4}  {int(row['product_id']):>10}  "
                f"{row['score']:>10.6f}  {str(rel):>10}"
            )

    # ------------------------------------------------------------------
    # Timing summary
    # ------------------------------------------------------------------
    t_total = time.time() - t_total_start
    print("\n" + "=" * 65)
    print("TIMING")
    print("=" * 65)
    print(f"  RRF fusion          : {t_fusion:.2f}s")
    print(f"  Evaluation          : {t_eval:.2f}s")
    print(f"  Total               : {t_total:.2f}s")

    # ------------------------------------------------------------------
    # Rerun command
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RERUN COMMAND")
    print("=" * 65)
    print(
        '  $env:PYTHONPATH="SpecMatch"; '
        '& "SpecMatch\\.venv\\Scripts\\python.exe" '
        '"SpecMatch\\experiments\\03_hybrid_baseline.py"'
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
