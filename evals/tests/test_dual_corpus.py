"""Unit and integration tests for Dual-Corpus Evaluation (RepoPilot v1.1).

Verifies:
1. 'repopilot-code' contains zero markdown (.md) or docs/ chunks.
2. 'docs/20-resume-and-interview-preparation.md' is strictly absent from 'repopilot-code'.
3. 'repopilot' (full) includes documentation chunks.
4. Production retrieve_all_stages respects repository_id isolation across all 4 stages.
"""

import os
import sys
import pytest

# Ensure backend is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

from app.database import SessionLocal
from app.models import CodeChunk, Repository
from app.search import retrieve_all_stages


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def test_corpus_isolation_in_database(db_session):
    full_repo = db_session.query(Repository).filter(Repository.full_name == "repopilot").first()
    code_repo = db_session.query(Repository).filter(Repository.full_name == "repopilot-code").first()

    assert full_repo is not None, "Repository 'repopilot' (full) must be indexed"
    assert code_repo is not None, "Repository 'repopilot-code' must be indexed"

    # 1. Full repo contains documentation
    full_paths = [
        r[0] for r in db_session.query(CodeChunk.file_path).filter(CodeChunk.repository_id == full_repo.id).distinct().all()
    ]
    docs_in_full = [p for p in full_paths if p.startswith("docs/") or p.endswith(".md")]
    assert len(docs_in_full) > 0, "Full-index corpus must contain documentation files"

    # 2. Code repo has zero documentation and zero markdown files
    code_paths = [
        r[0] for r in db_session.query(CodeChunk.file_path).filter(CodeChunk.repository_id == code_repo.id).distinct().all()
    ]
    docs_in_code = [p for p in code_paths if p.startswith("docs/") or p.endswith(".md")]
    assert len(docs_in_code) == 0, f"Code-only corpus must contain 0 documentation files, found: {docs_in_code}"

    # 3. Explicit check for docs/20-resume-and-interview-preparation.md
    assert "docs/20-resume-and-interview-preparation.md" not in code_paths
    assert "docs/20-resume-and-interview-preparation.md" in full_paths


def test_retrieval_pipeline_corpus_isolation(db_session):
    code_repo = db_session.query(Repository).filter(Repository.full_name == "repopilot-code").first()
    assert code_repo is not None

    # Query known to heavily match docs in full-index
    query = "How does the LangGraph agent prevent infinite tool execution loops?"
    stages = retrieve_all_stages(query=query, k_pool=20, repository_id=code_repo.id, db=db_session)

    assert "dense" in stages
    assert "fts" in stages
    assert "fused" in stages
    assert "reranked" in stages

    # Check every candidate across all 4 stages: MUST NOT be a markdown or doc file
    for st_name, candidates in stages.items():
        for cand in candidates:
            fp = cand["file_path"]
            assert not fp.startswith("docs/"), f"Doc file '{fp}' leaked into '{st_name}' stage of code-only corpus!"
            assert not fp.endswith(".md"), f"Markdown file '{fp}' leaked into '{st_name}' stage of code-only corpus!"
