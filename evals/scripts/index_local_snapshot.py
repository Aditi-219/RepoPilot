"""Index local repository snapshot directly into PostgreSQL for evaluation.

Supports Dual-Corpus Evaluation (RepoPilot v1.1):
1. 'full': All eligible repository files, including documentation and markdown.
2. 'code': Source code files only, excluding docs/ and all markdown content.

Ensures the database contains code chunks from the exact git snapshot
(commit a36d617658de5414f15879ed53338d6f83e090e5) matching evals/data/golden_v1.jsonl.
"""

import argparse
import os
import sys

# Repo root is 2 levels up from evals/scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

from app.chunker import chunk_file
from app.database import SessionLocal
from app.embedder import embed_chunks
from app.github_service import MAX_FILE_BYTES, SUPPORTED_EXTENSIONS, _get_file_extension
from app.models import CodeChunk, Repository
from sqlalchemy import delete

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".agents",
    "data", "brain", ".system_generated", "scripts", "evals",
}

CORPUS_CONFIGS = {
    "full": {
        "repo_name": "repopilot",
        "repo_url": "https://github.com/Aditi-219/RepoPilot",
        "github_id": 99999999,
        "description": "RepoPilot local repository snapshot - Full Index (Code + Documentation)",
        "excluded_extensions": set(),
        "excluded_path_prefixes": (),
    },
    "code": {
        "repo_name": "repopilot-code",
        "repo_url": "https://github.com/Aditi-219/RepoPilot-code",
        "github_id": 99999998,
        "description": "RepoPilot local repository snapshot - Code-Only (Excluding Docs & Markdown)",
        # Specifically excludes all markdown and any path starting with docs/
        # Explicitly excludes docs/20-resume-and-interview-preparation.md
        "excluded_extensions": {".md"},
        "excluded_path_prefixes": ("docs/",),
    },
}


def is_file_eligible(rel_path: str, ext: str, corpus: str) -> bool:
    """Check if a file is eligible for indexing in the specified corpus."""
    cfg = CORPUS_CONFIGS[corpus]
    if ext in cfg["excluded_extensions"]:
        return False
    normalized = rel_path.replace("\\", "/")
    for prefix in cfg["excluded_path_prefixes"]:
        if normalized.startswith(prefix):
            return False
    # Explicit safeguard against docs/20-resume-and-interview-preparation.md in code-only
    if corpus == "code" and "20-resume-and-interview-preparation" in normalized:
        return False
    return True


def index_snapshot(corpus: str = "full") -> int:
    """Freeze and index the repository snapshot for the requested corpus."""
    if corpus not in CORPUS_CONFIGS:
        raise ValueError(f"Unknown corpus '{corpus}'. Choose from {list(CORPUS_CONFIGS.keys())}.")

    cfg = CORPUS_CONFIGS[corpus]
    repo_name = cfg["repo_name"]
    repo_url = cfg["repo_url"]
    repo_github_id = cfg["github_id"]
    description = cfg["description"]

    print("\n" + "=" * 80)
    print(f"INDEXING LOCAL SNAPSHOT: CORPUS '{corpus.upper()}' ({repo_name})")
    print(f"Repository Root: {REPO_ROOT}")
    print("=" * 80)

    db = SessionLocal()
    try:
        # 1. Get or create Repository row
        repo = db.query(Repository).filter(Repository.full_name == repo_name).first()
        if repo is None:
            repo = Repository(
                github_id=repo_github_id,
                full_name=repo_name,
                url=repo_url,
                description=description,
                stars=0,
                language="Python",
                status="indexing",
            )
            db.add(repo)
            db.commit()
            db.refresh(repo)
            print(f"Created repository record ID={repo.id} ({repo_name})")
        else:
            repo.status = "indexing"
            db.commit()
            db.execute(delete(CodeChunk).where(CodeChunk.repository_id == repo.id))
            db.commit()
            print(f"Re-indexing repository record ID={repo.id} ({repo_name}); cleared previous chunks.")

        # 2. Gather indexable files
        file_entries = []
        for root, dirs, filenames in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            for fn in filenames:
                ext = _get_file_extension(fn)
                if ext in SUPPORTED_EXTENSIONS:
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, REPO_ROOT).replace("\\", "/")
                    if not is_file_eligible(rel, ext, corpus):
                        continue
                    size = os.path.getsize(full)
                    if size <= MAX_FILE_BYTES:
                        file_entries.append((rel, full))

        print(f"Found {len(file_entries)} eligible files for corpus '{corpus}'.")

        # 3. Chunk all files
        all_chunks = []
        for rel_path, full_path in file_entries:
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                file_chunks = chunk_file(rel_path, content)
                all_chunks.extend(file_chunks)
            except (OSError, UnicodeDecodeError, ValueError) as e:
                print(f"Error processing {rel_path}: {e}")

        print(f"Generated {len(all_chunks)} chunks across {len(file_entries)} files.")

        # 4. Batch embed and insert
        batch_size = 64
        total_inserted = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            embeddings = embed_chunks(batch)

            db_chunks = [
                CodeChunk(
                    repository_id=repo.id,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    language=chunk.language,
                    embedding=emb,
                )
                for chunk, emb in zip(batch, embeddings, strict=True)
            ]
            db.bulk_save_objects(db_chunks)
            db.commit()
            total_inserted += len(db_chunks)
            print(f"  Inserted {total_inserted}/{len(all_chunks)} chunks...")

        # 5. Mark completed
        repo.status = "completed"
        db.commit()
        print(f"Successfully indexed {total_inserted} chunks into repository ID={repo.id} ({repo_name})!\n")
        return repo.id

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index local snapshot for RepoPilot evaluation")
    parser.add_argument(
        "--corpus",
        choices=["full", "code", "all"],
        default="full",
        help="Corpus configuration to index ('full', 'code', or 'all')",
    )
    args = parser.parse_args()

    if args.corpus == "all":
        index_snapshot("full")
        index_snapshot("code")
    else:
        index_snapshot(args.corpus)
