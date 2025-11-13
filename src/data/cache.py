"""Data caching mechanism for market data."""
import logging
import os
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.utils.exceptions import DataError

logger = logging.getLogger(__name__)


class DataCache:
    """File-based cache for market data."""
    
    def __init__(self, cache_dir: str = "./data/cache", max_age_days: int = 1):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Directory to store cached data
            max_age_days: Maximum age of cached data in days before invalidation
        """
        self.cache_dir = Path(cache_dir)
        self.max_age_days = max_age_days
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DataCache at {self.cache_dir} with max_age={max_age_days} days")
    
    def _get_cache_path(self, symbol: str, start_date: date, end_date: date) -> Path:
        """
        Generate cache file path for a symbol and date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Path to cache file
        """
        filename = f"{symbol}_{start_date.isoformat()}_{end_date.isoformat()}.pkl"
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cached data is still valid based on age.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False
        
        # Get file modification time
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        
        is_valid = age.days < self.max_age_days
        
        if not is_valid:
            logger.debug(f"Cache expired: {cache_path.name} (age: {age.days} days)")
        
        return is_valid
    
    def cache_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        start_date: date,
        end_date: date
    ) -> None:
        """
        Store data in cache.
        
        Args:
            symbol: Stock symbol
            data: DataFrame to cache
            start_date: Start date of data
            end_date: End date of data
            
        Raises:
            DataError: If caching fails
        """
        try:
            cache_path = self._get_cache_path(symbol, start_date, end_date)
            
            # Save data using pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Cached data for {symbol} ({start_date} to {end_date}) at {cache_path}")
            
        except Exception as e:
            error_msg = f"Failed to cache data for {symbol}: {str(e)}"
            logger.error(error_msg)
            raise DataError(error_msg) from e
    
    def get_cached_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available and valid.
        
        Args:
            symbol: Stock symbol
            start_date: Start date of data
            end_date: End date of data
            
        Returns:
            Cached DataFrame if available and valid, None otherwise
        """
        try:
            cache_path = self._get_cache_path(symbol, start_date, end_date)
            
            # Check if cache exists and is valid
            if not self._is_cache_valid(cache_path):
                return None
            
            # Load cached data
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.debug(f"Retrieved cached data for {symbol} ({start_date} to {end_date})")
            return data
            
        except FileNotFoundError:
            logger.debug(f"No cache found for {symbol} ({start_date} to {end_date})")
            return None
        except Exception as e:
            logger.warning(f"Failed to retrieve cached data for {symbol}: {str(e)}")
            return None
    
    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """
        Clear cached data.
        
        Args:
            symbol: If provided, only clear cache for this symbol. Otherwise clear all.
            
        Returns:
            Number of cache files deleted
        """
        try:
            count = 0
            
            if symbol:
                # Clear cache for specific symbol
                pattern = f"{symbol}_*.pkl"
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
                    count += 1
                logger.info(f"Cleared {count} cache files for {symbol}")
            else:
                # Clear all cache
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    count += 1
                logger.info(f"Cleared all {count} cache files")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return 0
    
    def get_cache_info(self) -> dict:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            valid_count = sum(1 for f in cache_files if self._is_cache_valid(f))
            expired_count = len(cache_files) - valid_count
            
            return {
                'total_files': len(cache_files),
                'valid_files': valid_count,
                'expired_files': expired_count,
                'total_size_bytes': total_size,
                'cache_dir': str(self.cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {str(e)}")
            return {}
