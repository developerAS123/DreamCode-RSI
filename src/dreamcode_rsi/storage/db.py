from __future__ import annotations

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import structlog

from dreamcode_rsi.config import config
from dreamcode_rsi.storage.models import Base

logger = structlog.get_logger(__name__)

# Connect with SQLite check_same_thread=False or PostgreSQL
connect_args = {"check_same_thread": False} if config.db_url.startswith("sqlite") else {}

engine = create_engine(
    config.db_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully", db_url=config.db_url)


def get_db_session() -> Generator[Session, None, None]:
    """Dependency / context session generator."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

