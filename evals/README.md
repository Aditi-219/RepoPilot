# RepoPilot Evaluation Suite (`evals/`)

This directory houses all components, datasets, runners, tests, and experiment logs for evaluating RepoPilot's multi-stage retrieval architecture and AI agent workflows.

## Directory Structure

```
evals/
├── README.md                     # Evaluation guide & overview
├── RESULTS.md                    # Benchmark scorecards, calibrations & progression reports
├── data/
│   ├── golden_v1.jsonl           # Immutable 22-query baseline dataset
│   ├── golden_v2.jsonl           # 32-query dual-target dataset (26 answerable, 6 unanswerable)
│   └── symbol_ranges.json        # Precomputed AST line ranges for 41 symbol targets
├── scripts/
│   ├── bootstrap_ci.py           # 95% Bootstrap confidence intervals calculator (B=2000)
│   ├── eval_agent.py             # LangSmith & local agent comprehension/tool evaluation suite
│   ├── eval_chunking.py          # AST vs token chunking structural evaluator
│   ├── eval_llm_judge.py         # LLM-as-a-judge evaluation (DeepEval methodology)
│   ├── eval_reranker.py          # Cross-encoder reranker ceiling, lift & latency evaluator
│   ├── index_local_snapshot.py   # Dual-corpus snapshot freeze & PostgreSQL indexer
│   ├── log_langsmith.py          # LangSmith experiment logging & tracking
│   ├── metrics.py                # Deterministic IR metrics (Hit@k, Recall@k, Precision@k, MRR, nDCG@k)
│   ├── precompute_symbols.py     # AST symbol line range precomputer
│   └── run_eval.py               # Main retrieval benchmark runner (dual-corpus, multi-stage)
├── tests/
│   ├── test_agent_evaluation.py  # Unit tests for LangSmith evaluators and offline runner
│   ├── test_dual_corpus.py       # Corpus isolation & database partition tests
│   └── test_eval_metrics.py      # Unit tests for deterministic IR metrics & nDCG deduplication
└── runs/                         # Official baselines, benchmark runs, and diagnostic reports
```

## Running Evaluations

### 1. Snapshot Ingestion (Dual-Corpus)
```bash
# Index Full-Index corpus (includes documentation)
python evals/scripts/index_local_snapshot.py --corpus full

# Index Code-Only corpus (pure source code, excludes docs/)
python evals/scripts/index_local_snapshot.py --corpus code
```

### 2. Multi-Stage Retrieval Benchmarks
```bash
# Code-Only File-level evaluation (k_pool=50)
python evals/scripts/run_eval.py --corpus code --level file --k-pool 50

# Code-Only Symbol-level evaluation (k_pool=50)
python evals/scripts/run_eval.py --corpus code --level symbol --k-pool 50

# Full-Index File-level evaluation (k_pool=50)
python evals/scripts/run_eval.py --corpus full --level file --k-pool 50

# Full-Index Symbol-level evaluation (k_pool=50)
python evals/scripts/run_eval.py --corpus full --level symbol --k-pool 50
```

### 3. Statistical Confidence Intervals
```bash
python evals/scripts/bootstrap_ci.py evals/runs/run_code_file_<timestamp>.json
```

### 4. Running Evaluation Tests
```bash
# Run all evaluation-specific tests
pytest evals/tests

# Run entire repository test suite (backend + evals)
pytest backend evals/tests
```

### 5. Application-Level Evaluation (Level 3)

On Windows, the FastAPI backend uses `psycopg` async connections for LangGraph checkpointing, which requires `SelectorEventLoop` instead of the default `ProactorEventLoop`. Start the backend with:

```powershell
cd backend
.\.venv\Scripts\python.exe -c "import asyncio, uvicorn; asyncio.run(uvicorn.Server(uvicorn.Config('app.main:app', host='127.0.0.1', port=8000)).serve(), loop_factory=asyncio.SelectorEventLoop)"
```

Then run generation (decoupled from judging):

```powershell
backend\.venv\Scripts\python.exe evals/scripts/run_application_generate.py --dry-run
backend\.venv\Scripts\python.exe evals/scripts/run_application_generate.py --limit 3
backend\.venv\Scripts\python.exe evals/scripts/run_application_generate.py
```
