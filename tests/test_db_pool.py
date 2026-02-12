"""Tests for database connection pooling."""

import pytest
import tempfile
import os
import time
from threading import Thread
from queue import Empty
from utils.db_pool import ConnectionPool, get_pool, close_all_pools


@pytest.fixture
def temp_db():
    """Create a temporary database file"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except:
        pass


class TestConnectionPool:
    """Test connection pool implementation"""
    
    def test_pool_initialization(self, temp_db):
        """Test pool initializes with correct size"""
        pool = ConnectionPool(temp_db, pool_size=3)
        
        stats = pool.get_stats()
        assert stats.pool_size == 3
        assert stats.available_connections == 3
        assert stats.active_connections == 0
        
        pool.close_all()
    
    def test_get_and_return_connection(self, temp_db):
        """Test getting and returning connections"""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        # Get connection
        conn = pool.get_connection()
        assert conn is not None
        
        stats = pool.get_stats()
        assert stats.active_connections == 1
        assert stats.available_connections == 1
        
        # Return connection
        pool.return_connection(conn)
        
        stats = pool.get_stats()
        assert stats.active_connections == 0
        assert stats.available_connections == 2
        
        pool.close_all()
    
    def test_connection_context_manager(self, temp_db):
        """Test connection context manager"""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        with pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        
        # Connection should be returned
        stats = pool.get_stats()
        assert stats.active_connections == 0
        
        pool.close_all()
    
    def test_transaction_context_manager(self, temp_db):
        """Test transaction context manager with commit"""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        # Create table and insert data
        with pool.transaction() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, value TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'hello')")
        
        # Verify data was committed
        with pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM test")
            result = cursor.fetchone()
            assert result['id'] == 1
            assert result['value'] == 'hello'
        
        pool.close_all()
    
    def test_transaction_rollback_on_error(self, temp_db):
        """Test transaction rolls back on error"""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        # Create table
        with pool.transaction() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        
        # Try to insert duplicate primary key (should fail and rollback)
        try:
            with pool.transaction() as conn:
                conn.execute("INSERT INTO test VALUES (1, 'first')")
                conn.execute("INSERT INTO test VALUES (1, 'duplicate')")  # Will fail
        except Exception:
            pass  # Expected
        
        # Verify no data was inserted
        with pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test")
            count = cursor.fetchone()[0]
            assert count == 0
        
        pool.close_all()
    
    def test_pool_exhaustion(self, temp_db):
        """Test behavior when pool is exhausted"""
        pool = ConnectionPool(temp_db, pool_size=2, timeout=0.1)
        
        # Get all connections
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        
        # Try to get another (should timeout)
        with pytest.raises(Empty):
            pool.get_connection(timeout=0.1)
        
        # Return one connection
        pool.return_connection(conn1)
        
        # Now we can get another
        conn3 = pool.get_connection()
        assert conn3 is not None
        
        pool.return_connection(conn2)
        pool.return_connection(conn3)
        pool.close_all()
    
    def test_execute_helper(self, temp_db):
        """Test execute helper method"""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        # Create table
        pool.execute("CREATE TABLE test (id INTEGER, value TEXT)")
        
        # Insert data
        pool.execute("INSERT INTO test VALUES (?, ?)", (1, 'hello'))
        pool.execute("INSERT INTO test VALUES (?, ?)", (2, 'world'))
        
        # Fetch one
        result = pool.execute("SELECT * FROM test WHERE id = ?", (1,), fetch='one')
        assert result['id'] == 1
        assert result['value'] == 'hello'
        
        # Fetch all
        results = pool.execute("SELECT * FROM test", fetch='all')
        assert len(results) == 2
        
        pool.close_all()
    
    def test_wal_mode_enabled(self, temp_db):
        """Test that WAL mode is enabled for better concurrency"""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        with pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.upper() == 'WAL'
        
        pool.close_all()
    
    def test_foreign_keys_enabled(self, temp_db):
        """Test that foreign keys are enabled"""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        with pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys")
            enabled = cursor.fetchone()[0]
            assert enabled == 1
        
        pool.close_all()
    
    def test_statistics_tracking(self, temp_db):
        """Test pool statistics tracking"""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        # Get and return connections
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        pool.return_connection(conn1)
        pool.return_connection(conn2)
        
        stats = pool.get_stats()
        
        assert stats.total_checkouts == 2
        assert stats.total_checkins == 2
        assert stats.total_connections_created >= 2
        
        pool.close_all()
    
    def test_concurrent_access(self, temp_db):
        """Test concurrent access from multiple threads"""
        pool = ConnectionPool(temp_db, pool_size=5)
        
        # Create table
        with pool.transaction() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, thread_id INTEGER)")
        
        def worker(thread_id):
            for i in range(10):
                with pool.transaction() as conn:
                    conn.execute(
                        "INSERT INTO test VALUES (?, ?)",
                        (i, thread_id)
                    )
        
        # Create multiple threads
        threads = [Thread(target=worker, args=(i,)) for i in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all inserts succeeded
        with pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test")
            count = cursor.fetchone()[0]
            assert count == 50  # 5 threads * 10 inserts each
        
        pool.close_all()
    
    def test_connection_reuse(self, temp_db):
        """Test that connections are reused from pool"""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        # Get and return connection multiple times
        conn1 = pool.get_connection()
        pool.return_connection(conn1)
        
        conn2 = pool.get_connection()
        pool.return_connection(conn2)
        
        # Connections should be reused (pool should not grow)
        stats = pool.get_stats()
        assert stats.total_connections_created <= 2
        
        pool.close_all()
    
    def test_close_all(self, temp_db):
        """Test closing all connections"""
        pool = ConnectionPool(temp_db, pool_size=3)
        
        # Get some connections
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        
        # Return them
        pool.return_connection(conn1)
        pool.return_connection(conn2)
        
        # Close all
        pool.close_all()
        
        stats = pool.get_stats()
        assert stats.available_connections == 0


class TestGlobalPoolFunctions:
    """Test global pool functions"""
    
    def test_get_pool_global(self, temp_db):
        """Test global get_pool function"""
        pool1 = get_pool(temp_db, pool_size=3)
        
        # Getting same database returns same pool
        pool2 = get_pool(temp_db)
        assert pool1 is pool2
        
        close_all_pools()
    
    def test_close_all_pools_global(self, temp_db):
        """Test closing all pools globally"""
        pool1 = get_pool(temp_db + "_1", pool_size=2)
        pool2 = get_pool(temp_db + "_2", pool_size=2)
        
        # Create tables
        with pool1.transaction() as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")
        
        with pool2.transaction() as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")
        
        # Close all
        close_all_pools()
        
        # Pools should be empty
        stats1 = pool1.get_stats()
        stats2 = pool2.get_stats()
        
        assert stats1.available_connections == 0
        assert stats2.available_connections == 0


# Property-based tests using Hypothesis
try:
    from hypothesis import given, strategies as st, settings, HealthCheck
    
    class TestPoolProperties:
        """Property-based tests for connection pool"""
        
        @given(
            pool_size=st.integers(min_value=1, max_value=10),
            num_operations=st.integers(min_value=1, max_value=50)
        )
        @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_pool_never_exceeds_size(self, temp_db, pool_size, num_operations):
            """Property: Active connections never exceed pool size"""
            pool = ConnectionPool(temp_db, pool_size=pool_size, timeout=0.1)
            
            connections = []
            
            for _ in range(min(num_operations, pool_size)):
                try:
                    conn = pool.get_connection(timeout=0.1)
                    connections.append(conn)
                except Empty:
                    break
            
            stats = pool.get_stats()
            assert stats.active_connections <= pool_size
            
            # Return all connections
            for conn in connections:
                pool.return_connection(conn)
            
            pool.close_all()
        
        @given(
            operations=st.lists(
                st.sampled_from(['get', 'return', 'execute']),
                min_size=1,
                max_size=20
            )
        )
        @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_operations_maintain_consistency(self, temp_db, operations):
            """Property: Any sequence of operations maintains pool consistency"""
            pool = ConnectionPool(temp_db, pool_size=3, timeout=0.1)
            
            connections = []
            
            for op in operations:
                try:
                    if op == 'get' and len(connections) < 3:
                        conn = pool.get_connection(timeout=0.1)
                        connections.append(conn)
                    elif op == 'return' and connections:
                        conn = connections.pop()
                        pool.return_connection(conn)
                    elif op == 'execute':
                        pool.execute("SELECT 1")
                except Empty:
                    pass  # Pool exhausted, continue
                except Exception as e:
                    pytest.fail(f"Operation {op} failed: {e}")
            
            # Return remaining connections
            for conn in connections:
                pool.return_connection(conn)
            
            # Pool should be consistent
            stats = pool.get_stats()
            assert stats.active_connections == 0
            
            pool.close_all()

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass
