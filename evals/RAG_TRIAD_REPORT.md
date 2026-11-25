# RepoPilot End-to-End RAG Triad Evaluation Report
**Single Baseline Run ID**: `rag_triad_20261125_130722`  
**Dataset**: `evals/data/golden_v2.jsonl` (32 rows: 26 answerable evaluated, 6 unanswerable skipped)  
**Judge**: `gemini/gemini-flash-lite-latest` (unpinned alias; API `modelVersion`: `not_logged_in_130722`)  
**Baseline Generator**: `generator_provider=None`, `generator_model=None` (not logged in `130722`)  

---

## 1. Summary Table (`rag_triad_20261125_130722`)

| Metric | Valid $n$ | Errors | Mean | P50 | Std Dev | Min / Max | Pass Rate (Threshold) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Context Relevancy** | 26 | 0 | 0.4354 | 0.3902 | 0.1967 | 0.1333 / 0.9333 | 7.7% ($\ge 0.7$) |
| **Chunk File Precision** | 26 | 0 | 0.4308 | 0.4000 | 0.2168 | 0.2000 / 1.0000 | 11.5% ($\ge 0.7$) |
| **Contextual Precision** | 25 | 1 | 0.7644 | 0.8333 | — | — | 76.0% (19/25 at $\ge 0.7$) |
| **Contextual Recall** | 24 | 2 | 0.6667 | 1.0000 | — | — | 66.7% (16/24 at $\ge 0.7$) |
| **Answer Relevancy** | 26 | 0 | 0.9676 | 1.0000 | 0.1064 | 0.5000 / 1.0000 | 96.2% ($\ge 0.7$) |
| **Faithfulness (Strict 1.0)** | 26 | 0 | 0.9837 | 1.0000 | 0.0671 | 0.6667 / 1.0000 | **92.3% (24/26 at $= 1.0$)** |

- **% of Answers with Any Unsupported Claim (`faithfulness < 1.0`)**: **7.7% (2/26)** (`rp-009`, `rp-024` — judge-flagged only, pending human review).
- **Note on `generations_baseline_k5.json`**: `generations_baseline_k5.json` used `groq/openai/gpt-oss-120b`, which is not RepoPilot's shipped generator, so it is not part of the findings.

---

## 2. Limitations & Incomplete Tracks

### Limitations
- **Single judge**: Evaluated with `gemini/gemini-flash-lite-latest` (unpinned alias; API `modelVersion` not logged).
- **Sample size**: $n = 26$ answerable queries from `golden_v2.jsonl`.
- **Baseline generator not logged**: Run `130722` did not record `generator_provider` or `generator_model`.
- **No noise estimate**: Stochastic judge variance ($\sigma$) could not be completed due to quota exhaustion.
- **Unverified faithfulness flags**: `rp-009` and `rp-024` are judge-flagged only, pending human review.

### Collapsed Track Statuses
- **Ollama-Qwen judge**: not completed (~3 min/question on CPU; single completed row not comparable to Gemini).
- **Groq judge**: not completed (rate-limited with ~480s 429 backoffs after generation exhausted token quota).
- **Gold comparison**: not reliable (`generator_baseline_gold.json` used `ollama/qwen2.5-coder:1.5b` and `gemini/gemini-3.5-flash-lite`, differing from `130722`).
- **Noise estimation**: not completed (cache-off re-run halted at `rp-007` of Repeat 1 by Gemini daily quota).
- **`tau=2.0` Triad**: not completed (Gemini daily quota exhausted).
- **Judge dependence**: not reliable (only partial 10-row `rp-009`/`rp-024` + 8-row check; full 26-row paired run blocked by quota).

---

## 3. Unified Recall Table (`pool=15`) & Decision

- **System Boundary**: Pipeline retrieval (pool 15) and the component retriever baseline (pool 50) are different systems.
- **Component Retriever Baseline (`k_pool=50`, `retriever_baseline_full_file.json`)**: Hit@5 = **1.0000**, Fractional Recall@5 = **0.5994**.

| Configuration (`pool=15`) | Mean Chunks | Binary Hit@k | Fractional Recall@k |
| :--- | :--- | :--- | :--- |
| Baseline (`k=5`) | 5.00 | 1.0000 | 0.6410 |
| `k=4` | 4.00 | 1.0000 | 0.5865 |
| `k=3` | 3.00 | 0.9231 | 0.5160 |
| `tau = 1.0` (`keep >= 1`) | 3.81 | 0.9231 | 0.5321 |
| `tau = 2.0` (`keep >= 1`) | 3.12 | 0.8462 | 0.4679 |
| `tau = 3.0` (`keep >= 1`) | 2.38 | 0.7692 | 0.3974 |

- **Decision**: **Keep `k=5`** (no score cutoff, `pool=15`). All tested score cutoffs (`tau=1.0/2.0/3.0`) and `k=3` reduce Binary Hit@k below `1.0000`.

---

## 4. Contextual Recall Audit (Errors Excluded)

- **Parse Failures Marked `evaluation_error` (Excluded)**:
  - `rp-004` (`ContextualRecall`): `evaluation_error` — `Evaluation LLM outputted an invalid JSON. Please use a better evaluation model.`
  - `rp-015` (`ContextualRecall`): `evaluation_error` — `Evaluation LLM outputted an invalid JSON. Please use a better evaluation model.`
  - `rp-001` (`ContextualPrecision`): `evaluation_error` — `Evaluation LLM outputted an invalid JSON. Please use a better evaluation model.`
- **Valid Contextual Recall ($n=24$)**: Mean = **0.6667**, P50 = **1.0000**, Pass Rate = **66.7% (16/24)**.
- **Valid Contextual Precision ($n=25$)**: Mean = **0.7644**, P50 = **0.8333**, Pass Rate = **76.0% (19/25)**.

### Valid Questions Scoring `< 1.0` on Contextual Recall ($n=8$)

- **`rp-008`** (score=0.0000): Missing `gt_files`: `['backend/app/agent/graph.py']`. Reason: *"The score is 0.00 because none of the information regarding the agent checking the recursion step against the recursion_"*
- **`rp-013`** (score=0.0000): Missing `gt_files`: `['docs/12-github-and-mcp-integration.md']`. Reason: *"The score is 0.00 because the retrieval context lacks the necessary implementation details regarding the GitHub Contents"*
- **`rp-014`** (score=0.0000): Missing `gt_files`: `['backend/app/routers/chat.py']`. Reason: *"The score is 0.00 because the sentence regarding `init_checkpointer` establishing a pool and its initialization during s"*
- **`rp-017`** (score=0.0000): Missing `gt_files`: `['backend/app/agent/graph.py', 'docs/14-security-and-error-handling.md']`. Reason: *"The score is 0.00 because the expected output regarding `agent_node` in `backend/app/agent/graph.py` handling interrupte"*
- **`rp-024`** (score=0.0000): Missing `gt_files`: `[]`. Reason: *"The score is 0.00 because the entire expected output regarding the upgrade function creating the vector extension and ta"*
- **`rp-026`** (score=0.0000): Missing `gt_files`: `['backend/app/schemas.py', 'docs/08-backend-and-api-reference.md']`. Reason: *"The score is 0.00 because the mention of the RepositoryCreate schema in backend/app/schemas.py validating URL formatting"*
- **`rp-027`** (score=0.0000): Missing `gt_files`: `['backend/app/agent/state.py']`. Reason: *"The score is 0.00 because the specific file path 'backend/app/agent/state.py' and details regarding 'repository_id', 'se"*
- **`rp-028`** (score=0.0000): Missing `gt_files`: `['docs/14-security-and-error-handling.md']`. Reason: *"The score is 0.00 because the 1st node in the retrieval context only partially covers the expected output by mentioning "*

---

## 5. Per-Question Diagnoses (`rp-009`, `rp-024`, and Eye-Inspection `rp-001`..`rp-005`)

### `rp-009` (Faithfulness = 0.9091 — Status: **judge-flagged, pending human review**)
- **Question**: `How does RepoPilot decide whether to use local Ollama or cloud Gemini for RAG answer generation?`
- **Judge-Flagged Claim**: The score is 0.91 because the actual output incorrectly claims that setting OLLAMA_MODEL=qwen2.5-coder:latest indicates a fallback to the Google Gemini API, whereas the context clearly states it makes RepoPilot prefer the local model.
- **Generated Answer**:
```text
RepoPilot decides whether to use local Ollama or cloud Gemini for RAG answer generation based on the following criteria:

1. **Local Ollama Check**: RepoPilot first checks if a local Ollama model is installed and running. It does this by attempting to connect to the Ollama HTTP REST daemon at `http://localhost:11434`. If the connection is successful, RepoPilot uses the local model without hitting any daily rate limits.

2. **Cloud Gemini Fallback**: If the local Ollama is not available, RepoPilot falls back to executing the RAG generation process via the Google Gemini API. This is indicated by the `ollama_model` environment variable being set to a specific model identifier, such as `qwen2.5-coder:latest`.

3. **Model Preference**: The `OLLAMA_MODEL` environment variable is used to specify the preferred local model installed in Ollama. If this variable is not set, RepoPilot picks the first installed model.

4. **Fallback Mechanism**: The `GEMINI_MODEL` environment variable is used to specify the Google Gemini model identifier used for generation. If this variable is not set, RepoPilot defaults to using the `gemini-3.5-flash-lite` model.

5. **Fallback Timeout**: The decision to use the local Ollama or cloud Gemini API is made with a timeout of 3.0 seconds, ensuring that the process does not hang indefinitely if the local Ollama is not available.

The code snippet at `backend/app/llm_service.py:L153-L166` provides a clear example of how RepoPilot checks for the local Ollama and handles the fallback to the cloud Gemini API based on the specified environment variables.
```
- **Retrieved Chunks**:
  1. `File: backend/app/llm_service.py (lines 1-21)`
  2. `File: docs/16-setup-and-local-development.md (lines 134-149)`
  3. `File: docs/01-project-overview.md (lines 68-72)`
  4. `File: docs/20-resume-and-interview-preparation.md (lines 43-50)`
  5. `File: docs/16-setup-and-local-development.md (lines 34-37)`

### `rp-024` (Faithfulness = 0.6667 — Status: **judge-flagged, pending human review**)
- **Question**: `Where are the initial database tables and pgvector extension created in the Alembic migrations?`
- **Judge-Flagged Claim**: The score is 0.67 because the output incorrectly attributes the creation of the pgvector extension and LangGraph checkpoint tables to the initial migration, whereas the retrieval context states they are created by their respective libraries and the project relies on SQLAlchemy and raw SQL for schema setup.
- **Generated Answer**:
```text
The initial database tables and pgvector extension are created in the Alembic migration script `b71dccb6d03d_initial_schema.py` located at `backend/alembic/versions/b71dccb6d03d_initial_schema.py`. This script is executed when the database is initialized or upgraded using Alembic.
```
- **Retrieved Chunks**:
  1. `File: backend/alembic/versions/b71dccb6d03d_initial_schema.py (lines 1-26)`
  2. `File: docs/09-database-and-data-model.md (lines 151-159)`
  3. `File: docs/18-known-limitations-and-technical-debt.md (lines 24-36)`
  4. `File: docs/02-feature-inventory.md (lines 23-24)`
  5. `File: docs/04-repository-structure.md (lines 3-20)`

### Verbatim Eye-Inspection Samples (`rp-001`..`rp-005`)

- **`rp-001`** (Ctx=0.5217, Faith=1.0000, Ans=1.0000): `Where is Reciprocal Rank Fusion (RRF) implemented to combine semantic and keyword search rankings?`
  - **Answer**: Reciprocal Rank Fusion (RRF) is implemented in the `backend/app/search.py` file, specifically in the `_rrf_fusion` function, which is called within the `hybrid_search` function. The function takes in semantic and keyword results, along with a constant `k`, and returns a list of `SearchResult` objects that have been RRF-scored. The RRF score is calculated by summing the reciprocal rank positions of each document across both result lists, with a smoothing constant `k`.
  - **Chunks**: `['File: backend/app/search.py (lines 193-215)', 'File: docs/04-repository-structure.md (lines 207-217)', 'File: docs/20-resume-and-interview-preparation.md (lines 140-145)', 'File: backend/app/search.py (lines 1-20)', 'File: docs/13-rag-and-code-retrieval.md (lines 95-103)']`
- **`rp-002`** (Ctx=0.4286, Faith=1.0000, Ans=1.0000): `Where is the neural cross-encoder model loaded and invoked for candidate reranking?`
  - **Answer**: The neural cross-encoder model is loaded and invoked for candidate reranking in the `rerank` function of `reranker.py`, lines 118–143. The function `_neural_rerank` attempts to load the cross-encoder model using `sentence-transformers` and scores each `(query, chunk)` pair through the cross-encoder model. If the model is unavailable, it falls back to the heuristic mode.
  - **Chunks**: `['File: backend/app/reranker.py (lines 13-30)', 'File: docs/13-rag-and-code-retrieval.md (lines 104-110)', 'File: backend/app/reranker.py (lines 1-19)', 'File: backend/app/reranker.py (lines 118-143)', 'File: docs/18-known-limitations-and-technical-debt.md (lines 59-67)']`
- **`rp-003`** (Ctx=0.3529, Faith=1.0000, Ans=0.8000): `Where are the HNSW vector cosine distance index and full-text search GIN index created in the database?`
  - **Answer**: The HNSW vector cosine distance index and full-text search GIN index are created in the database using the following SQL commands:

```sql
CREATE INDEX IF NOT EXISTS idx_code_chunks_embedding_hnsw
ON code_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

```sql
CREATE INDEX IF NOT EXISTS idx_code_chunks_content_gin
ON code_chunks
USING gin(to_tsvector('english', content));
```

These commands are executed in the `backend/app/main.py` file, lines 67-71, as shown in the provided code snippet.
  - **Chunks**: `['File: docs/04-repository-structure.md (lines 207-217)', 'File: backend/app/main.py (lines 69-91)', 'File: docs/20-resume-and-interview-preparation.md (lines 163-169)', 'File: docs/01-project-overview.md (lines 63-67)', 'File: docs/03-architecture.md (lines 12-18)']`
- **`rp-004`** (Ctx=0.4167, Faith=1.0000, Ans=1.0000): `Where is Tree-sitter AST parsing configured to extract classes, functions, and methods for chunking?`
  - **Answer**: Tree-sitter AST parsing is configured to extract classes, functions, and methods for chunking in the `chunk_file_ast` function within the `chunker.py` file. This function uses the `tree-sitter` parser to identify syntactic boundaries and then processes the parsed AST to separate complete functions and classes into coherent units. The `CHUNK_STRATEGY` variable is set to `"ast"` by default, which means the function will use Tree-sitter to parse the source code and extract the required information.
  - **Chunks**: `['File: docs/01-project-overview.md (lines 60-62)', 'File: docs/13-rag-and-code-retrieval.md (lines 44-51)', 'File: backend/app/chunker.py (lines 303-328)', 'File: docs/05-technology-stack.md (lines 64-67)', 'File: backend/tests/test_chunker.py (lines 68-84)']`
- **`rp-005`** (Ctx=0.4545, Faith=1.0000, Ans=1.0000): `Where is the Server-Sent Events (SSE) streaming chat endpoint implemented?`
  - **Answer**: The Server-Sent Events (SSE) streaming chat endpoint is implemented in the `chat.py` file located at `backend/app/routers/chat.py` on line 232.
  - **Chunks**: `['File: docs/20-resume-and-interview-preparation.md (lines 170-176)', 'File: docs/02-feature-inventory.md (lines 221-229)', 'File: docs/01-project-overview.md (lines 73-76)', 'File: backend/app/routers/chat.py (lines 232-253)', 'File: frontend/src/services/api.ts (lines 1-34)']`

---

## 6. Next Steps
1. Run `k=4` when quota allows with the shipped generator.
2. Study docs-outranking-code misses (`rp-008`, `rp-017`, `rp-026`, `rp-027`).
3. Add a generator instruction for premise contradictions (e.g., `rp-009`, `rp-024`).
