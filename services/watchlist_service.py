"""
Watchlist Management Service

Provides functionality for creating, managing, and analyzing stock watchlists.
Supports multiple watchlists with AI recommendations and import/export capabilities.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import os

logger = logging.getLogger(__name__)


@dataclass
class Watchlist:
    """Represents a stock watchlist."""
    name: str
    symbols: List[str]
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'symbols': self.symbols,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Watchlist':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            symbols=data['symbols'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )


class WatchlistService:
    """Service for managing stock watchlists."""
    
    def __init__(self, storage_path: str = 'data/watchlists.json'):
        """
        Initialize watchlist service.
        
        Args:
            storage_path: Path to store watchlist data
        """
        self.storage_path = storage_path
        self.watchlists: Dict[str, Watchlist] = {}
        self._ensure_storage_dir()
        self._load_watchlists()
    
    def _ensure_storage_dir(self):
        """Ensure storage directory exists."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
    
    def _load_watchlists(self):
        """Load watchlists from storage."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.watchlists = {
                        name: Watchlist.from_dict(wl_data)
                        for name, wl_data in data.items()
                    }
                logger.info(f"Loaded {len(self.watchlists)} watchlists")
        except Exception as e:
            logger.error(f"Error loading watchlists: {e}")
            self.watchlists = {}
    
    def _save_watchlists(self):
        """Save watchlists to storage."""
        try:
            data = {
                name: wl.to_dict()
                for name, wl in self.watchlists.items()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.watchlists)} watchlists")
        except Exception as e:
            logger.error(f"Error saving watchlists: {e}")
            raise
    
    def create_watchlist(self, name: str, symbols: Optional[List[str]] = None) -> Watchlist:
        """
        Create a new watchlist.
        
        Args:
            name: Name of the watchlist
            symbols: Initial list of symbols (optional)
        
        Returns:
            Created watchlist
        
        Raises:
            ValueError: If watchlist name already exists
        """
        if name in self.watchlists:
            raise ValueError(f"Watchlist '{name}' already exists")
        
        now = datetime.now()
        watchlist = Watchlist(
            name=name,
            symbols=symbols or [],
            created_at=now,
            updated_at=now
        )
        
        self.watchlists[name] = watchlist
        self._save_watchlists()
        logger.info(f"Created watchlist: {name}")
        
        return watchlist
    
    def delete_watchlist(self, name: str) -> bool:
        """
        Delete a watchlist.
        
        Args:
            name: Name of the watchlist to delete
        
        Returns:
            True if deleted, False if not found
        """
        if name not in self.watchlists:
            return False
        
        del self.watchlists[name]
        self._save_watchlists()
        logger.info(f"Deleted watchlist: {name}")
        
        return True
    
    def rename_watchlist(self, old_name: str, new_name: str) -> bool:
        """
        Rename a watchlist.
        
        Args:
            old_name: Current name of the watchlist
            new_name: New name for the watchlist
        
        Returns:
            True if renamed, False if old name not found
        
        Raises:
            ValueError: If new name already exists
        """
        if old_name not in self.watchlists:
            return False
        
        if new_name in self.watchlists:
            raise ValueError(f"Watchlist '{new_name}' already exists")
        
        watchlist = self.watchlists[old_name]
        watchlist.name = new_name
        watchlist.updated_at = datetime.now()
        
        self.watchlists[new_name] = watchlist
        del self.watchlists[old_name]
        self._save_watchlists()
        logger.info(f"Renamed watchlist: {old_name} -> {new_name}")
        
        return True
    
    def add_symbol(self, watchlist_name: str, symbol: str) -> bool:
        """
        Add a symbol to a watchlist.
        
        Args:
            watchlist_name: Name of the watchlist
            symbol: Stock symbol to add
        
        Returns:
            True if added, False if watchlist not found or symbol already exists
        """
        if watchlist_name not in self.watchlists:
            return False
        
        watchlist = self.watchlists[watchlist_name]
        symbol = symbol.upper()
        
        if symbol in watchlist.symbols:
            return False
        
        watchlist.symbols.append(symbol)
        watchlist.updated_at = datetime.now()
        self._save_watchlists()
        logger.info(f"Added {symbol} to watchlist: {watchlist_name}")
        
        return True
    
    def remove_symbol(self, watchlist_name: str, symbol: str) -> bool:
        """
        Remove a symbol from a watchlist.
        
        Args:
            watchlist_name: Name of the watchlist
            symbol: Stock symbol to remove
        
        Returns:
            True if removed, False if watchlist or symbol not found
        """
        if watchlist_name not in self.watchlists:
            return False
        
        watchlist = self.watchlists[watchlist_name]
        symbol = symbol.upper()
        
        if symbol not in watchlist.symbols:
            return False
        
        watchlist.symbols.remove(symbol)
        watchlist.updated_at = datetime.now()
        self._save_watchlists()
        logger.info(f"Removed {symbol} from watchlist: {watchlist_name}")
        
        return True
    
    def get_watchlist(self, name: str) -> Optional[Watchlist]:
        """
        Get a watchlist by name.
        
        Args:
            name: Name of the watchlist
        
        Returns:
            Watchlist if found, None otherwise
        """
        return self.watchlists.get(name)
    
    def get_all_watchlists(self) -> List[Watchlist]:
        """
        Get all watchlists.
        
        Returns:
            List of all watchlists
        """
        return list(self.watchlists.values())
    
    def get_watchlist_names(self) -> List[str]:
        """
        Get names of all watchlists.
        
        Returns:
            List of watchlist names
        """
        return list(self.watchlists.keys())
    
    def export_watchlist(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Export a watchlist to dictionary format.
        
        Args:
            name: Name of the watchlist to export
        
        Returns:
            Dictionary representation of watchlist, or None if not found
        """
        watchlist = self.get_watchlist(name)
        if not watchlist:
            return None
        
        return watchlist.to_dict()
    
    def import_watchlist(self, data: Dict[str, Any], overwrite: bool = False) -> bool:
        """
        Import a watchlist from dictionary format.
        
        Args:
            data: Dictionary representation of watchlist
            overwrite: Whether to overwrite if watchlist already exists
        
        Returns:
            True if imported successfully, False otherwise
        
        Raises:
            ValueError: If watchlist exists and overwrite is False
        """
        try:
            watchlist = Watchlist.from_dict(data)
            
            if watchlist.name in self.watchlists and not overwrite:
                raise ValueError(f"Watchlist '{watchlist.name}' already exists")
            
            self.watchlists[watchlist.name] = watchlist
            self._save_watchlists()
            logger.info(f"Imported watchlist: {watchlist.name}")
            
            return True
        except ValueError:
            # Re-raise ValueError for duplicate watchlist
            raise
        except Exception as e:
            logger.error(f"Error importing watchlist: {e}")
            return False
    
    def export_all_watchlists(self) -> Dict[str, Any]:
        """
        Export all watchlists.
        
        Returns:
            Dictionary of all watchlists
        """
        return {
            name: wl.to_dict()
            for name, wl in self.watchlists.items()
        }
    
    def import_all_watchlists(self, data: Dict[str, Any], overwrite: bool = False) -> int:
        """
        Import multiple watchlists.
        
        Args:
            data: Dictionary of watchlists
            overwrite: Whether to overwrite existing watchlists
        
        Returns:
            Number of watchlists imported
        """
        count = 0
        for name, wl_data in data.items():
            try:
                if self.import_watchlist(wl_data, overwrite):
                    count += 1
            except Exception as e:
                logger.error(f"Error importing watchlist {name}: {e}")
        
        return count
