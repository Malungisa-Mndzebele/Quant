"""
Database connection pooling for improved performance.

Provides connection pooling for SQLite databases to avoid
creating new connections for every operation.
"""

import sqlite3
import logging
from typing import Optional, Any, Callable
from contextlib import contextmanager
from threading import Lock
from queue import Queue, Empty, Full
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PoolStats:
    """Connection pool statistics"""
    pool_size: int
    active_connections: int
    available_connections: int
    total_connections_created: int
    total_connections_closed: int
    total_checkouts: int
    total_checkins: int
    created_at: datetime


class ConnectionPool:
    """
    Thread-safe SQLite connection pool.
    
    Manages a pool of database connections to avoid the overhead
    of creating new connections for every operation.
    """
    
    def __init__(
        self,
        database: str,
        pool_size: int = 5,
        timeout: float = 30.0,
        check_same_thread: bool = False
    ):
        """
        Initialize connection pool.
        
        Args:
            database: Path to SQLite database file
            pool_size: Maximum number of connections in pool
            timeout: Timeout for getting connection from pool (seconds)
            check_same_thread: SQLite check_same_thread parameter
        """
        self.database = database
        self.pool_size = pool_size
        self.timeout = timeout
        self.check_same_thread = check_same_thread
        
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = Lock()
        self._active_connections = 0
        
        # Statistics
        self._total_created = 0
        self._total_closed = 0
        self._total_checkouts = 0
        self._total_checkins = 0
        self._created_at = datetime.now()
        
        # Pre-create connections
        self._initialize_pool()
        
        logger.info(
            f"Initialized connection pool for {database}: "
            f"size={pool_size}, timeout={timeout}s"
        )
    
    def _initialize_pool(self):
        """Pre-create connections to fill the pool"""
        for _ in range(self.pool_size):
            try:
                conn = self._create_connection()
                self._pool.put(conn, block=False)
            except Full:
                break
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
    
    def _create_connection(self) -> sqlite3.Connection:
        """
        Create a new database connection.
        
        Returns:
            SQLite connection object
        """
        try:
            conn = sqlite3.connect(
                self.database,
                timeout=self.timeout,
                check_same_thread=self.check_same_thread
            )
            conn.row_factory = sqlite3.Row
            
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Set journal mode to WAL for better concurrency
            conn.execute("PRAGMA journal_mode = WAL")
            
            with self._lock:
                self._total_created += 1
            
            logger.debug(f"Created new database connection (total: {self._total_created})")
            
            return conn
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    def get_connection(self, timeout: Optional[float] = None) -> sqlite3.Connection:
        """
        Get a connection from the pool.
        
        Args:
            timeout: Timeout for getting connection (uses pool timeout if None)
            
        Returns:
            SQLite connection object
            
        Raises:
            Empty: If no connection available within timeout
        """
        timeout = timeout or self.timeout
        
        try:
            # Try to get existing connection from pool
            conn = self._pool.get(timeout=timeout)
            
            with self._lock:
                self._active_connections += 1
                self._total_checkouts += 1
            
            logger.debug(
                f"Checked out connection (active: {self._active_connections})"
            )
            
            return conn
        except Empty:
            # Pool is empty, create new connection if under limit
            with self._lock:
                if self._active_connections < self.pool_size:
                    conn = self._create_connection()
                    self._active_connections += 1
                    self._total_checkouts += 1
                    return conn
            
            # Pool is full and all connections are in use
            raise Empty(
                f"Connection pool exhausted (size: {self.pool_size}, "
                f"active: {self._active_connections})"
            )
    
    def return_connection(self, conn: sqlite3.Connection):
        """
        Return a connection to the pool.
        
        Args:
            conn: Connection to return
        """
        try:
            # Rollback any uncommitted transactions
            conn.rollback()
            
            # Return to pool
            self._pool.put(conn, block=False)
            
            with self._lock:
                self._active_connections -= 1
                self._total_checkins += 1
            
            logger.debug(
                f"Returned connection to pool (active: {self._active_connections})"
            )
        except Full:
            # Pool is full, close the connection
            self._close_connection(conn)
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")
            self._close_connection(conn)
    
    def _close_connection(self, conn: sqlite3.Connection):
        """
        Close a database connection.
        
        Args:
            conn: Connection to close
        """
        try:
            conn.close()
            
            with self._lock:
                self._active_connections -= 1
                self._total_closed += 1
            
            logger.debug(f"Closed database connection (total closed: {self._total_closed})")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    @contextmanager
    def connection(self):
        """
        Context manager for getting and returning connections.
        
        Usage:
            with pool.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)
    
    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.
        
        Automatically commits on success or rolls back on error.
        
        Usage:
            with pool.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO table VALUES (?)", (value,))
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.return_connection(conn)
    
    def execute(
        self,
        query: str,
        parameters: tuple = (),
        fetch: str = 'none'
    ) -> Any:
        """
        Execute a query using a pooled connection.
        
        Args:
            query: SQL query
            parameters: Query parameters
            fetch: Fetch mode ('none', 'one', 'all')
            
        Returns:
            Query results based on fetch mode
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, parameters)
            
            if fetch == 'one':
                return cursor.fetchone()
            elif fetch == 'all':
                return cursor.fetchall()
            else:
                return cursor.lastrowid
    
    def close_all(self):
        """Close all connections in the pool"""
        closed_count = 0
        
        # Close connections in pool
        while not self._pool.empty():
            try:
                conn = self._pool.get(block=False)
                conn.close()
                closed_count += 1
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error closing pooled connection: {e}")
        
        with self._lock:
            self._total_closed += closed_count
        
        logger.info(f"Closed {closed_count} pooled connections")
    
    def get_stats(self) -> PoolStats:
        """
        Get connection pool statistics.
        
        Returns:
            PoolStats object with current statistics
        """
        with self._lock:
            return PoolStats(
                pool_size=self.pool_size,
                active_connections=self._active_connections,
                available_connections=self._pool.qsize(),
                total_connections_created=self._total_created,
                total_connections_closed=self._total_closed,
                total_checkouts=self._total_checkouts,
                total_checkins=self._total_checkins,
                created_at=self._created_at
            )
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_all()
        return False


# Global pool registry
_pools: dict[str, ConnectionPool] = {}
_pools_lock = Lock()


def get_pool(
    database: str,
    pool_size: int = 5,
    timeout: float = 30.0,
    check_same_thread: bool = False
) -> ConnectionPool:
    """
    Get or create a connection pool for a database.
    
    Args:
        database: Path to SQLite database file
        pool_size: Maximum number of connections in pool
        timeout: Timeout for getting connection from pool (seconds)
        check_same_thread: SQLite check_same_thread parameter
        
    Returns:
        ConnectionPool instance
    """
    with _pools_lock:
        if database not in _pools:
            _pools[database] = ConnectionPool(
                database=database,
                pool_size=pool_size,
                timeout=timeout,
                check_same_thread=check_same_thread
            )
        
        return _pools[database]


def close_all_pools():
    """Close all connection pools"""
    with _pools_lock:
        for pool in _pools.values():
            pool.close_all()
        _pools.clear()
        logger.info("Closed all connection pools")


def get_all_pool_stats() -> dict[str, PoolStats]:
    """
    Get statistics for all connection pools.
    
    Returns:
        Dictionary mapping database paths to their pool statistics
    """
    with _pools_lock:
        return {
            database: pool.get_stats()
            for database, pool in _pools.items()
        }
