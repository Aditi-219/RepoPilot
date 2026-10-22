"""Main application entry point for the RepoPilot FastAPI backend.

This module initializes the FastAPI application, mounts CORS middleware,
manages startup/shutdown lifecycle tasks (database table and index creation),
and registers modular APIRouters for:
- Repositories: /api/repositories
- Indexing:     /api/repositories/{id}/index & chunk endpoints
- RAG:          /api/repositories/{id}/search & /ask
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.database import Base, engine
from app.routers.indexing import router as indexing_router
from app.routers.rag import router as rag_router
from app.routers.repositories import router as repositories_router

# ---------------------------------------------------------------------------
# Application Configuration & Logging
# ---------------------------------------------------------------------------
VERSION = "0.3.1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Determine allowed frontend origins for Cross-Origin Resource Sharing (CORS).
# Defaults to standard Vite dev server ports (localhost and 127.0.0.1:5173).
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
ALLOWED_ORIGINS = list({
    FRONTEND_URL,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
})


# ---------------------------------------------------------------------------
# Lifespan Management (Startup and Shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events.

    Code before 'yield' runs when the server boots up.
    Code after 'yield' runs when the server shuts down.
    """
    logger.info("Starting up RepoPilot API v%s...", VERSION)

    # 1. Create ORM tables (repositories, code_chunks) if they don't exist yet
    Base.metadata.create_all(bind=engine)

    # 2. Ensure high-performance search indexes exist in PostgreSQL:
    #    - HNSW vector index for pgvector cosine distance search
    #    - GIN index for PostgreSQL full-text keyword search
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_code_chunks_embedding_hnsw
                ON code_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_code_chunks_content_gin
                ON code_chunks
                USING gin(to_tsvector('english', content))
            """))
            conn.commit()
        logger.info("Database search indexes (HNSW + GIN) verified.")
    except Exception as exc:
        # Non-fatal: search will still function with sequential table scans
        logger.warning("Could not create search indexes (non-fatal): %s", exc)

    yield

    logger.info("Shutting down RepoPilot API...")


# ---------------------------------------------------------------------------
# FastAPI Instance Creation & Middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RepoPilot API",
    description=(
        "Backend API for RepoPilot: AI-powered codebase intelligence, "
        "hybrid retrieval, and RAG-based codebase Q&A."
    ),
    version=VERSION,
    lifespan=lifespan,
)

# Configure CORS Middleware so the React frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Root & Health Check Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    """Root endpoint welcoming developers and linking to interactive Swagger docs."""
    return {
        "message": "Welcome to RepoPilot API",
        "docs": "/docs",
        "version": VERSION,
    }


@app.get("/api/health", tags=["General"])
def health() -> dict[str, str]:
    """Health check endpoint used by uptime monitors or container orchestrators."""
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Register Modular Routers
# ---------------------------------------------------------------------------
app.include_router(repositories_router)
app.include_router(indexing_router)
app.include_router(rag_router)
