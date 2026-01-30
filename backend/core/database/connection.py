"""
Database connection and session management.
Supports both sync and async operations.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError, DBAPIError
from typing import Generator, AsyncGenerator
import os
import logging
import time

logger = logging.getLogger(__name__)

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL', '')

# Convert postgres:// to postgresql:// if needed (Railway sometimes uses postgres://)
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# For async, replace postgresql:// with postgresql+asyncpg://
DATABASE_URL_ASYNC = DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://')

# Create engines
engine = None
async_engine = None
SessionLocal = None
AsyncSessionLocal = None

def init_db(max_retries: int = 3, retry_delay: float = 1.0):
    """
    Initialize database engines and session factories with retry logic.
    Call this once at application startup.
    
    Args:
        max_retries: Number of connection attempts
        retry_delay: Seconds to wait between retries
        
    Raises:
        RuntimeError: If connection fails after all retries
    """
    global engine, async_engine, SessionLocal, AsyncSessionLocal
    
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set - database features disabled")
        return
    
    last_error = None
    for attempt in range(max_retries):
        try:
            # Sync engine (for migrations, simple operations)
            engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,  # Verify connections before using
                pool_size=20,  # Increased for parallel processing (was 5)
                max_overflow=30,  # Increased for parallel processing (was 10)
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=False,  # Set to True for SQL debugging
                connect_args={
                    'connect_timeout': 10,  # Connection timeout (seconds)
                    'options': '-c statement_timeout=90000'  # Query timeout (90s) - increased for vector search with filters
                }
            )
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Async engine (for FastAPI endpoints)
            async_engine = create_async_engine(
                DATABASE_URL_ASYNC,
                pool_pre_ping=True,
                pool_size=20,  # Increased for parallel processing (was 5)
                max_overflow=30,  # Increased for parallel processing (was 10)
                pool_recycle=3600,
                echo=False
            )
            
            # Session factories
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            AsyncSessionLocal = async_sessionmaker(
                async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info(f"Database initialized: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'local'}")
            return
            
        except (OperationalError, DBAPIError) as e:
            last_error = e
            logger.warning(f"Database connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Database initialization failed after {max_retries} attempts")
                raise RuntimeError(f"Failed to connect to database: {e}") from e


def get_db() -> Generator[Session, None, None]:
    """
    Get a sync database session (for use in process_inbox.py, migrations).
    
    Usage:
        from backend.core.database import get_db
        
        with next(get_db()) as db:
            email = db.query(Email).first()
    """
    if not SessionLocal:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    db = SessionLocal()
    try:
        yield db
    except (OperationalError, DBAPIError) as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def get_fresh_session() -> Session:
    """
    Get a fresh database session, disposing of potentially stale connections.
    Use this after connection errors to get a clean session.
    """
    if not SessionLocal:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    # Dispose of all connections to clear stale ones
    if engine:
        engine.dispose()
    
    return SessionLocal()


def with_db_retry(func, max_retries: int = 3, retry_delay: float = 1.0):
    """
    Decorator/wrapper that retries database operations on connection errors.
    
    Usage:
        @with_db_retry
        def my_db_operation(session, ...):
            ...
            
        # Or as a wrapper:
        result = with_db_retry(lambda: do_something(session))
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (OperationalError, DBAPIError) as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check if this is a connection error (retryable)
                is_connection_error = any(pattern in error_msg for pattern in [
                    'closed the connection', 'connection refused', 'timeout',
                    'connection reset', 'server terminated', 'invalid transaction',
                    'cannot reconnect', 'connection timed out', 'broken pipe'
                ])
                
                if is_connection_error and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Database operation failed (attempt {attempt+1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    
                    # Dispose connections to force reconnect
                    if engine:
                        engine.dispose()
                    
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        
        raise last_error or Exception(f"Database operation failed after {max_retries} attempts")
    
    return wrapper


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session (for FastAPI endpoints).
    
    Usage:
        from fastapi import Depends
        from backend.core.database import get_async_db
        
        @app.get("/emails")
        async def list_emails(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(Email))
            return result.scalars().all()
    """
    if not AsyncSessionLocal:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except (OperationalError, DBAPIError) as e:
            logger.error(f"Async database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


def create_tables():
    """
    Create all tables in the database.
    Only use for initial setup - prefer Alembic migrations for production.
    """
    if not engine:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    from .models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


def drop_tables():
    """
    Drop all tables (DESTRUCTIVE - use with caution!).
    Only for development/testing.
    """
    if not engine:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    from .models import Base
    Base.metadata.drop_all(bind=engine)
    logger.warning("Database tables dropped")

