"""Database configuration and session management for RepoPilot.

This module initializes the SQLAlchemy engine and provides a scoped session
generator for FastAPI route dependencies.
"""

import os
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# 1. Load environment variables from the .env file (e.g., DATABASE_URL)
# Loaded here because database.py is imported before main.py's startup code runs.
load_dotenv()

# 2. Retrieve the PostgreSQL connection string
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing from the .env file. "
        "Please provide a valid PostgreSQL connection string."
    )

# 3. Create the SQLAlchemy Engine
# The engine is the central source of database connections.
# - pool_pre_ping=True: Tests connections before using them to prevent
#   errors from stale or dropped connections.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

# 4. Create the Session Factory
# SessionLocal generates new database sessions.
# - autoflush=False: Prevents automatic flushing of pending changes before queries.
# - autocommit=False: Ensures all database writes require an explicit commit().
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# 5. Base Class for Declarative Models
# In SQLAlchemy 2.0, all ORM models inherit from DeclarativeBase.
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models in RepoPilot."""
    pass


# 6. Database Dependency for FastAPI Endpoints
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per HTTP request.

    How it works:
    1. A new Session is created when an endpoint receives a request.
    2. 'yield db' passes the active session to the route handler.
    3. The context manager ensures the session is cleanly closed once the
       request finishes, preventing connection leaks (no manual close needed).
    """
    with SessionLocal() as db:
        yield db
