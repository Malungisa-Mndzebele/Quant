"""
Improved caching manager with size limits and TTL support.

Replaces simple dictionary-based caching with a more robust solution
that prevents memory issues and provides better cache management.
"""

import time
import logging
from typing import Any, Optional, Dict, Tuple, Callable
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value and metadata"""
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0


class LRUCache:
    """
    Thread-safe LRU cache with TTL and size limits.
    
    Features:
    - Least Recently Used eviction policy
    - Time-to-live (TTL) expiration
    - Maximum size limit (number of entries)
    - Thread-safe operations
    - Cache statistics
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,
        name: str = "cache"
    ):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries in seconds
            name: Cache name for logging
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.name = name
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(
            f"Initialized {name} cache: max_size={max_size}, ttl={ttl_seconds}s"
        )
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if time.time() - entry.timestamp > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                logger.debug(f"{self.name}: Cache expired for key {key}")
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.access_count += 1
            self._hits += 1
            
            return entry.value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # If key exists, update it
            if key in self._cache:
                self._cache[key] = CacheEntry(
                    value=value,
                    timestamp=time.time(),
                    access_count=0
                )
                self._cache.move_to_end(key)
                return
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size:
                # Remove least recently used (first item)
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(
                    f"{self.name}: Evicted key {evicted_key} "
                    f"(cache full: {len(self._cache)}/{self.max_size})"
                )
            
            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                access_count=0
            )
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"{self.name}: Cleared {count} cache entries")
    
    def clear_expired(self) -> int:
        """
        Clear expired entries.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if current_time - entry.timestamp > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug(
                    f"{self.name}: Cleared {len(expired_keys)} expired entries"
                )
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'name': self.name,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'ttl_seconds': self.ttl_seconds
            }
    
    def reset_stats(self) -> None:
        """Reset cache statistics"""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    def __len__(self) -> int:
        """Get number of entries in cache"""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (doesn't check expiration)"""
        with self._lock:
            return key in self._cache


class CacheManager:
    """
    Global cache manager for managing multiple named caches.
    
    Provides a centralized way to create and manage caches across
    the application with consistent configuration.
    """
    
    def __init__(self):
        """Initialize cache manager"""
        self._caches: Dict[str, LRUCache] = {}
        self._lock = Lock()
        
        logger.info("Initialized global cache manager")
    
    def get_cache(
        self,
        name: str,
        max_size: int = 1000,
        ttl_seconds: float = 300.0
    ) -> LRUCache:
        """
        Get or create a named cache.
        
        Args:
            name: Cache name
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live in seconds
            
        Returns:
            LRUCache instance
        """
        with self._lock:
            if name not in self._caches:
                self._caches[name] = LRUCache(
                    max_size=max_size,
                    ttl_seconds=ttl_seconds,
                    name=name
                )
            
            return self._caches[name]
    
    def clear_all(self) -> None:
        """Clear all caches"""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            logger.info("Cleared all caches")
    
    def clear_expired_all(self) -> int:
        """
        Clear expired entries from all caches.
        
        Returns:
            Total number of entries cleared
        """
        with self._lock:
            total_cleared = 0
            for cache in self._caches.values():
                total_cleared += cache.clear_expired()
            
            if total_cleared > 0:
                logger.info(f"Cleared {total_cleared} expired entries from all caches")
            
            return total_cleared
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.
        
        Returns:
            Dictionary mapping cache names to their statistics
        """
        with self._lock:
            return {
                name: cache.get_stats()
                for name, cache in self._caches.items()
            }
    
    def remove_cache(self, name: str) -> bool:
        """
        Remove a named cache.
        
        Args:
            name: Cache name
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                logger.info(f"Removed cache: {name}")
                return True
            return False


# Global cache manager instance
_cache_manager = CacheManager()


def get_cache(
    name: str,
    max_size: int = 1000,
    ttl_seconds: float = 300.0
) -> LRUCache:
    """
    Get or create a named cache from the global cache manager.
    
    Args:
        name: Cache name
        max_size: Maximum number of entries
        ttl_seconds: Time-to-live in seconds
        
    Returns:
        LRUCache instance
    """
    return _cache_manager.get_cache(name, max_size, ttl_seconds)


def clear_all_caches() -> None:
    """Clear all caches in the global cache manager"""
    _cache_manager.clear_all()


def clear_expired_caches() -> int:
    """
    Clear expired entries from all caches.
    
    Returns:
        Total number of entries cleared
    """
    return _cache_manager.clear_expired_all()


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for all caches.
    
    Returns:
        Dictionary mapping cache names to their statistics
    """
    return _cache_manager.get_all_stats()
