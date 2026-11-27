# Golden Dataset v3 Changelog (PROPOSED)

> **Status:** `PROPOSED` — awaiting human verification of the cited code lines before treating as final.
> `evals/data/golden_v2.jsonl` was **not** modified.

## Summary of Changes (`golden_v2.jsonl` → `golden_v3.jsonl`)

### 1. `rp-003` — Index creation locations (incomplete in v2)
- **Justification**:
  - `backend/app/main.py` (`lifespan`, lines 65–78) creates `idx_code_chunks_embedding_hnsw` and `idx_code_chunks_content_gin` at server boot.
  - `backend/alembic/versions/b71dccb6d03d_initial_schema.py` (`upgrade()`, lines 93–109) **also** creates both `idx_code_chunks_embedding_hnsw` and `idx_code_chunks_content_gin`.
- **Old `ideal_answer` (v2)**:
  > Database indexes idx_code_chunks_embedding_hnsw and idx_code_chunks_content_gin are created during server boot in the lifespan context manager in backend/app/main.py.
- **Proposed `ideal_answer` (v3)**:
  > Database indexes idx_code_chunks_embedding_hnsw and idx_code_chunks_content_gin are created during server boot in the lifespan context manager in backend/app/main.py (lines 65-78) and/or in the Alembic initial schema migration in backend/alembic/versions/b71dccb6d03d_initial_schema.py (upgrade(), lines 93-109). Either location is valid.

### 2. `rp-009` — Ollama vs Gemini provider selection (stale in v2)
- **Justification**:
  - `backend/app/llm_service.py` (`ask_with_llm`, lines 155–179) now calls `acomplete(..., role="generator")` first and only falls back to `_try_ollama` (lines 66–106) on exception.
  - `backend/app/llm_provider.py` (`DEFAULT_CHAIN` and `get_generator_chain()`, lines 31–47) reads `GENERATOR_CHAIN` (default `["gemini", "groq", "ollama"]`).
- **Old `ideal_answer` (v2)**:
  > In ask_with_llm (backend/app/llm_service.py), the system first tries _try_ollama. If an Ollama server is running locally with the model available, it answers for free without cloud rate limits. If Ollama fails or is not present, it falls back to _try_gemini using candidate models (gemini-3.5-flash-lite, gemini-3.8-flash, etc.).
- **Proposed `ideal_answer` (v3)**:
  > In ask_with_llm (backend/app/llm_service.py, lines 155-179), RepoPilot calls acomplete(..., role="generator") from backend/app/llm_provider.py (lines 31-47), which resolves get_generator_chain() from GENERATOR_CHAIN (default: gemini, groq, ollama). If acomplete fails, ask_with_llm falls back to the legacy _try_ollama helper (backend/app/llm_service.py, lines 66-106).

### 3. `rp-027` — `AgentState` fields (wrong in v2)
- **Justification**:
  - `backend/app/agent/state.py` (`AgentState`, lines 11–25) defines `messages: Annotated[Sequence[BaseMessage], add_messages]`, `repo_id: int | None`, `repo_owner: str | None`, and `repo_name: str | None`. The v2 fields (`repository_id`, `session_id`, `recursion_step`) do not exist in `state.py`.
- **Old `ideal_answer` (v2)**:
  > AgentState in backend/app/agent/state.py is a TypedDict containing messages (annotated with add_messages), repository_id, session_id, and recursion_step counter.
- **Proposed `ideal_answer` (v3)**:
  > AgentState in backend/app/agent/state.py (lines 11-25) is a TypedDict containing messages (Annotated[Sequence[BaseMessage], add_messages]), repo_id (int | None), repo_owner (str | None), and repo_name (str | None).

### Unchanged
- `rp-026` was kept unchanged in `golden_v3.jsonl` (the v1 penalty was caused by Rubric v1 penalizing extra valid detail, which Rubric v2 Correctness resolves).

## Added `key_points` Field (26 Answerable Rows)
Each answerable row in `golden_v3.jsonl` now includes a `key_points` list (2–4 concrete files, symbols, or values) for count-based `Completeness_v2` evaluation:
- `rp-001`: `backend/app/search.py`, `_rrf_fusion`, `RRF_K = 60` (3)
- `rp-002`: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `_get_cross_encoder`, `_neural_rerank`, `backend/app/reranker.py` (4)
- `rp-003`: `idx_code_chunks_embedding_hnsw` & `idx_code_chunks_content_gin`, `lifespan` in `main.py` and/or `upgrade()` in `b71dccb6d03d_initial_schema.py` (2)
- `rp-004`: `_get_tree_sitter_parser`, `chunk_file_ast`, `backend/app/chunker.py` (3)
- `rp-005`: `POST /api/chat/{repository_id}/sessions/{session_id}/stream`, `stream_chat`, `backend/app/routers/chat.py` (3)
- `rp-006`: `index_repository`, `fetch_repo_tree`+downloads, `chunk_file`, `embed_chunks`+`CodeChunk` (4)
- `rp-007`: `hybrid_search`+`embed_query`, `_semantic_search`, `_keyword_search`, `_rrf_fusion` (4)
- `rp-008`: `agent_node` (`graph.py`), `recursion_limit` check, 2-step summary directive (3)
- `rp-009`: `ask_with_llm` (`acomplete`), `get_generator_chain()` (`llm_provider.py`), legacy `_try_ollama` fallback (3)
- `rp-010`: `chunk_file`→`_chunk_markdown`, `##` header boundaries, `chunk_size` limit (3)
- `rp-011`: `embed_chunks` (`embedder.py`), `all-MiniLM-L6-v2` + `384`-dim, `CodeChunk` (`models.py`) (3)
- `rp-012`: `ask_repository`→`_retrieve_and_rerank`, `hybrid_search`+`rerank`, `ask_with_llm` (3)
- `rp-013`: `read_source_file` (`github_tools.py`), `fetch_file_content` (`github_service.py`), GitHub Contents API + base64/utf-8 (3)
- `rp-014`: `init_checkpointer` (`AsyncPostgresSaver`), `lifespan` (`main.py`), `create_agent_graph` (`chat.py`) (3)
- `rp-015`: `fetch_repo_tree`, `SUPPORTED_EXTENSIONS`, `MAX_FILE_BYTES`/`MAX_FILE_SIZE_KB` (`100 KB`) (3)
- `rp-016`: `require_indexed` (`helpers.py`), `repo.status == 'completed'`, HTTP `409 Conflict` (3)
- `rp-017`: `agent_node` (`graph.py`), trailing `AIMessage` `tool_calls`, placeholder `ToolMessage`s (3)
- `rp-018`: `chunk_file_ast` (`chunker.py`), `_get_tree_sitter_parser` None/`parse` error, `_chunk_by_tokens` fallback (3)
- `rp-023`: `streamChatMessage` (`api.ts`), `POST` via `fetch`, `ReadableStream` SSE reader (3)
- `rp-024`: `upgrade()` (`b71dccb6d03d_initial_schema.py`), `CREATE EXTENSION IF NOT EXISTS vector`, tables created (3)
- `rp-025`: `mcp_app` (`mcp_server.py`), 5 registered MCP tools (2)
- `rp-026`: `RepositoryCreate` (`HttpUrl`), `parse_github_url` (`github_service.py`), rejects non-`github.com` (3)
- `rp-027`: `AgentState` (`state.py`), `messages`+`add_messages`, `repo_id`/`repo_owner`/`repo_name` (3)
- `rp-028`: `get_repository_or_404` & `require_indexed`, `404` if missing, `409 Conflict` if not `completed` (3)
- `rp-029`: `backend/app/models.py`, `Repository`→`repositories`, `CodeChunk`→`code_chunks` (3)
- `rp-030`: `rerank` (`reranker.py`), tracks seen chunk IDs/keys to deduplicate (2)

## Post-Draft Key-Point Edits (Frozen SHA-256: `ee98c15fe286c9758f750dcee1f781bd6e1aaaf8e129d0d64d827b3980fe8b76`)

1. **`rp-001` (`RRF_K = 60` constant)**:
   - **Old**: `"Constant \`RRF_K = 60\`"` (string, no aliases)
   - **New**: `{"text": "Constant \`RRF_K = 60\`", "kind": "literal", "match": ["RRF_K = 60"], "aliases": ["k = 60", "RRF_K=60"]}`
   - **Reason**: Allow common notation variants (`RRF_K=60`, `k = 60`) while normalizing backticks and whitespace.
2. **`rp-024` (`upgrade()` function)**:
   - **Old**: `"Function \`upgrade\`"` (bare word `upgrade` matched prose)
   - **New**: `{"text": "Function \`upgrade()\`", "kind": "literal", "match": ["upgrade()", "def upgrade"], "aliases": ["upgrade function"]}`
   - **Reason**: Tighten generic word `upgrade` to function-call/definition forms plus the phrasing `"upgrade function"` used in `ideal_answer`.
3. **`rp-029` (`repositories` and `code_chunks` table names)**:
   - **Old**: `"repositories"` and `"code_chunks"` (plain lowercase matched generic English words)
   - **New**: `raw_match` requiring backticks or quotes (`` `repositories` ``, `'repositories'`, `"repositories"`; `` `code_chunks` ``, `'code_chunks'`, `"code_chunks"`)
   - **Reason**: Prevent generic prose words like "the repositories are indexed" from false-matching SQL table name checks.

## Authoring Notes & Proposals (Key Points Remain FROZEN)
- **`rp-013` Authoring Error**: `golden_v2.jsonl` / `golden_v3.jsonl` wrote `"with utf-8 fallback handling"`, whereas `backend/app/github_service.py:L340-L345` decodes as `utf-8` and returns `None` on `UnicodeDecodeError` (no fallback encoding). Also, `rp-013`'s question does not ask for error/fallback handling.
- **`rp-024` Split Proposal (PROPOSAL only, not applied)**: Split `"Creates tables repositories, code_chunks, and chat_sessions"` into 3 atomic literal points: `` `repositories` ``, `` `code_chunks` ``, and `` `chat_sessions` ``.
