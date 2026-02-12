"""Tests for cache manager utility."""

import time
import pytest
from threading import Thread
from utils.cache_manager import LRUCache, CacheManager, get_cache


class TestLRUCache:
    """Test LRU cache implementation"""
    
    def test_basic_get_set(self):
        """Test basic get and set operations"""
        cache = LRUCache(max_size=10, ttl_seconds=60, name="test")
        
        # Set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Get non-existent key
        assert cache.get("key2") is None
    
    def test_ttl_expiration(self):
        """Test that entries expire after TTL"""
        cache = LRUCache(max_size=10, ttl_seconds=0.1, name="test")
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        
        assert cache.get("key1") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = LRUCache(max_size=3, ttl_seconds=60, name="test")
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert len(cache) == 3
        
        # Add one more - should evict key1 (least recently used)
        cache.set("key4", "value4")
        
        assert len(cache) == 3
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_lru_access_updates_order(self):
        """Test that accessing an entry updates its position"""
        cache = LRUCache(max_size=3, ttl_seconds=60, name="test")
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add key4 - should evict key2 (now least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_update_existing_key(self):
        """Test updating an existing key"""
        cache = LRUCache(max_size=10, ttl_seconds=60, name="test")
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Update
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"
    
    def test_delete(self):
        """Test deleting entries"""
        cache = LRUCache(max_size=10, ttl_seconds=60, name="test")
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Delete
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        
        # Delete non-existent
        assert cache.delete("key2") is False
    
    def test_clear(self):
        """Test clearing all entries"""
        cache = LRUCache(max_size=10, ttl_seconds=60, name="test")
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert len(cache) == 3
        
        cache.clear()
        
        assert len(cache) == 0
        assert cache.get("key1") is None
    
    def test_clear_expired(self):
        """Test clearing only expired entries"""
        cache = LRUCache(max_size=10, ttl_seconds=0.15, name="test")
        
        cache.set("key1", "value1")
        time.sleep(0.1)
        cache.set("key2", "value2")
        
        # Wait for key1 to expire but not key2
        time.sleep(0.1)
        
        cleared = cache.clear_expired()
        
        assert cleared >= 1  # At least key1 should be cleared
        assert cache.get("key1") is None
        # key2 might or might not be expired depending on timing
    
    def test_statistics(self):
        """Test cache statistics tracking"""
        cache = LRUCache(max_size=10, ttl_seconds=60, name="test")
        
        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Generate hits and misses
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        stats = cache.get_stats()
        
        assert stats['name'] == "test"
        assert stats['size'] == 2
        assert stats['max_size'] == 10
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 2/3
    
    def test_thread_safety(self):
        """Test thread-safe operations"""
        cache = LRUCache(max_size=100, ttl_seconds=60, name="test")
        
        def worker(thread_id):
            for i in range(100):
                key = f"key_{thread_id}_{i}"
                cache.set(key, f"value_{thread_id}_{i}")
                cache.get(key)
        
        # Create multiple threads
        threads = [Thread(target=worker, args=(i,)) for i in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Cache should have entries (may be less than 500 due to eviction)
        assert len(cache) > 0
        assert len(cache) <= 100  # Should not exceed max_size
    
    def test_contains(self):
        """Test __contains__ method"""
        cache = LRUCache(max_size=10, ttl_seconds=60, name="test")
        
        cache.set("key1", "value1")
        
        assert "key1" in cache
        assert "key2" not in cache


class TestCacheManager:
    """Test cache manager"""
    
    def test_get_cache(self):
        """Test getting or creating caches"""
        manager = CacheManager()
        
        cache1 = manager.get_cache("cache1", max_size=10, ttl_seconds=60)
        cache2 = manager.get_cache("cache2", max_size=20, ttl_seconds=120)
        
        assert cache1 is not cache2
        
        # Getting same cache returns same instance
        cache1_again = manager.get_cache("cache1")
        assert cache1 is cache1_again
    
    def test_clear_all(self):
        """Test clearing all caches"""
        manager = CacheManager()
        
        cache1 = manager.get_cache("cache1")
        cache2 = manager.get_cache("cache2")
        
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")
        
        manager.clear_all()
        
        assert cache1.get("key1") is None
        assert cache2.get("key2") is None
    
    def test_clear_expired_all(self):
        """Test clearing expired entries from all caches"""
        manager = CacheManager()
        
        cache1 = manager.get_cache("cache1", ttl_seconds=0.1)
        cache2 = manager.get_cache("cache2", ttl_seconds=0.1)
        
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")
        
        time.sleep(0.2)
        
        cleared = manager.clear_expired_all()
        
        assert cleared == 2
    
    def test_get_all_stats(self):
        """Test getting statistics for all caches"""
        manager = CacheManager()
        
        cache1 = manager.get_cache("cache1")
        cache2 = manager.get_cache("cache2")
        
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")
        
        stats = manager.get_all_stats()
        
        assert "cache1" in stats
        assert "cache2" in stats
        assert stats["cache1"]["size"] == 1
        assert stats["cache2"]["size"] == 1
    
    def test_remove_cache(self):
        """Test removing a cache"""
        manager = CacheManager()
        
        cache1 = manager.get_cache("cache1")
        cache1.set("key1", "value1")
        
        assert manager.remove_cache("cache1") is True
        assert manager.remove_cache("cache1") is False  # Already removed
        
        # Getting cache again creates new instance
        cache1_new = manager.get_cache("cache1")
        assert cache1_new.get("key1") is None


class TestGlobalCacheFunctions:
    """Test global cache functions"""
    
    def test_get_cache_global(self):
        """Test global get_cache function"""
        cache = get_cache("test_global", max_size=10, ttl_seconds=60)
        
        cache.set("key1", "value1")
        
        # Getting same cache returns same instance
        cache_again = get_cache("test_global")
        assert cache_again.get("key1") == "value1"


# Property-based tests using Hypothesis
try:
    from hypothesis import given, strategies as st
    
    class TestCacheProperties:
        """Property-based tests for cache"""
        
        @given(
            keys=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
            values=st.lists(st.integers(), min_size=1, max_size=20)
        )
        def test_set_get_consistency(self, keys, values):
            """Property: Getting a key immediately after setting it returns the same value"""
            cache = LRUCache(max_size=100, ttl_seconds=60, name="test")
            
            # Ensure keys and values have same length
            min_len = min(len(keys), len(values))
            keys = keys[:min_len]
            values = values[:min_len]
            
            for key, value in zip(keys, values):
                cache.set(key, value)
                assert cache.get(key) == value
        
        @given(
            max_size=st.integers(min_value=1, max_value=100),
            num_items=st.integers(min_value=1, max_value=200)
        )
        def test_size_limit_property(self, max_size, num_items):
            """Property: Cache never exceeds max_size"""
            cache = LRUCache(max_size=max_size, ttl_seconds=60, name="test")
            
            for i in range(num_items):
                cache.set(f"key_{i}", f"value_{i}")
            
            assert len(cache) <= max_size
        
        @given(
            operations=st.lists(
                st.tuples(
                    st.sampled_from(['set', 'get', 'delete']),
                    st.text(min_size=1, max_size=10),
                    st.integers()
                ),
                min_size=1,
                max_size=50
            )
        )
        def test_operations_dont_crash(self, operations):
            """Property: Any sequence of operations should not crash"""
            cache = LRUCache(max_size=20, ttl_seconds=60, name="test")
            
            for op, key, value in operations:
                try:
                    if op == 'set':
                        cache.set(key, value)
                    elif op == 'get':
                        cache.get(key)
                    elif op == 'delete':
                        cache.delete(key)
                except Exception as e:
                    pytest.fail(f"Operation {op} crashed: {e}")

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass
