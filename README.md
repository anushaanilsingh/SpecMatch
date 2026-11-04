# SpecMatch: Constraint-Aware Semantic Retrieval for E-Commerce Product Search

SpecMatch is an MSc-level NLP research project investigating whether semantic product retrieval can improve search relevance while preserving explicit user constraints (colour, material, size, brand, compatibility).

## Dataset: WANDS (Wayfair ANNotation Dataset)
- Official repository: https://github.com/wayfair/WANDS
- Contains 42,994 products, 480 queries, and 233,448 relevance annotations (Exact, Partial, Irrelevant).

## Directory Structure
```text
SpecMatch/
│
├── data/
│   ├── raw/            # Pristine original dataset files (product.csv, query.csv, label.csv)
│   ├── processed/      # Cleaned and normalized data
│   └── annotations/    # Constraint annotations and extraction ground truths
│
├── notebooks/          # Exploratory data analysis and experimental prototyping
│
├── src/                # Modular Python source code (data loaders, retrieval, evaluation)
│
├── experiments/        # Experiment configurations, evaluation scripts, and baselines
│
├── outputs/            # Output indices, evaluation metrics, logs, figures
│
├── research/           # Research notes, literature review, dataset analysis reports
│
├── report/             # Final MSc dissertation/report write-up and LaTeX sources
│
├── presentation/       # Presentation slides and visual assets
│
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── .gitignore          # Version control ignore rules
```

## Setup & Environment
The project is configured to run on CPU by default.

### 1. Virtual Environment Setup
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. CPU Execution
All retrieval algorithms, transformer encoders, and evaluations are optimized for CPU-first execution.

## Retrieval Baselines & Evaluation Summary

All three retrieval baselines have been fully implemented, executed, and benchmarked on the 480 queries of WANDS:

| Model | Retrieval Type | nDCG@10 | MRR@10 (Exact) | MRR@10 (Any) | Recall@10 (Exact) | Recall@10 (Any) | Judged Coverage@10 | Latency (sec) |
|---|---|---|---|---|---|---|---|---|
| **BM25Okapi** | Lexical | **0.6457** | **0.5469** | 0.8347 | **0.2718** | **0.0567** | **79.02%** | 91.05s |
| **all-MiniLM-L6-v2** | Dense Semantic | 0.5550 | 0.4327 | 0.8087 | 0.1753 | 0.0524 | 71.17% | 7.79s |
| **Hybrid (RRF, k=60)** | Rank Fusion | 0.6224 | 0.5191 | **0.8454** | 0.2689 | 0.0562 | 76.50% | 0.03s |

### Reproducing Baseline Experiments

```powershell
# Set PYTHONPATH and execute from project root:
$env:PYTHONPATH="SpecMatch"

# 1. Lexical Baseline (BM25)
& "SpecMatch\.venv\Scripts\python.exe" "SpecMatch\experiments\01_bm25_baseline.py"

# 2. Dense Semantic Baseline (SentenceTransformers all-MiniLM-L6-v2)
& "SpecMatch\.venv\Scripts\python.exe" "SpecMatch\experiments\02_semantic_baseline.py"

# 3. Hybrid RRF Baseline
& "SpecMatch\.venv\Scripts\python.exe" "SpecMatch\experiments\03_hybrid_baseline.py"
```


