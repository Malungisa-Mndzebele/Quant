"""Integration tests for system initialization."""

import pytest
import os
import tempfile
import yaml
from pathlib import Path

from src.utils.config_loader import ConfigLoader, load_config, ConfigurationError
from src.trading_system import TradingSystem, TradingSystemError
from src.models.config import TradingMode


class TestConfigurationLoading:
    """Test configuration loading and validation."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000,
                'symbols': ['AAPL', 'GOOGL'],
                'update_interval_seconds': 60,
                'enable_logging': True,
                'log_level': 'INFO'
            },
            'brokerage': {
                'provider': 'simulated',
                'credentials': {
                    'api_key': None,
                    'api_secret': None
                },
                'timeout_seconds': 30,
                'max_retries': 3
            },
            'data': {
                'provider': 'yfinance',
                'cache_enabled': True,
                'cache_dir': './data/cache',
                'cache_ttl_seconds': 3600,
                'max_retries': 3
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {
                        'fast_period': 10,
                        'slow_period': 30
                    }
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load()
        
        assert config.trading.mode == TradingMode.SIMULATION
        assert config.trading.initial_capital == 100000
        assert config.brokerage.provider == 'simulated'
        assert config.data.provider == 'yfinance'
        assert config.risk.max_position_size_pct == 10.0
        assert len(config.strategies) == 1
        assert config.strategies[0].name == 'MovingAverageCrossover'
    
    def test_load_config_with_env_vars(self, tmp_path, monkeypatch):
        """Test configuration with environment variable substitution."""
        # Set environment variables
        monkeypatch.setenv('TEST_API_KEY', 'test_key_123')
        monkeypatch.setenv('TEST_API_SECRET', 'test_secret_456')
        
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000
            },
            'brokerage': {
                'provider': 'simulated',
                'credentials': {
                    'api_key': '${TEST_API_KEY}',
                    'api_secret': '${TEST_API_SECRET}'
                }
            },
            'data': {
                'provider': 'yfinance'
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {}
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load()
        
        assert config.brokerage.api_key == 'test_key_123'
        assert config.brokerage.api_secret == 'test_secret_456'
    
    def test_load_config_missing_file(self):
        """Test loading configuration from non-existent file."""
        loader = ConfigLoader('nonexistent.yaml')
        
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            loader.load()
    
    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML configuration."""
        config_file = tmp_path / "config.yaml"
        
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError, match="Failed to parse YAML"):
            loader.load()
    
    def test_load_config_empty_file(self, tmp_path):
        """Test loading empty configuration file."""
        config_file = tmp_path / "config.yaml"
        config_file.touch()
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError, match="Configuration file is empty"):
            loader.load()
    
    def test_validate_live_mode_requires_credentials(self, tmp_path):
        """Test that live mode requires brokerage credentials."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'live',  # Live mode
                'initial_capital': 100000
            },
            'brokerage': {
                'provider': 'public',  # Real brokerage
                'credentials': {
                    'api_key': None,  # Missing credentials
                    'api_secret': None
                }
            },
            'data': {
                'provider': 'yfinance'
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {}
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError, match="API key is required"):
            loader.load()
    
    def test_validate_at_least_one_strategy(self, tmp_path):
        """Test that at least one strategy must be enabled."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000
            },
            'brokerage': {
                'provider': 'simulated',
                'credentials': {}
            },
            'data': {
                'provider': 'yfinance'
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': False  # Disabled
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError, match="At least one strategy must be enabled"):
            loader.load()
    
    def test_validate_invalid_trading_mode(self, tmp_path):
        """Test validation of invalid trading mode."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'invalid_mode',
                'initial_capital': 100000
            },
            'brokerage': {
                'provider': 'simulated'
            },
            'data': {
                'provider': 'yfinance'
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {}
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError):
            loader.load()


class TestSystemInitialization:
    """Test trading system initialization."""
    
    def test_system_initialization_simulation_mode(self, tmp_path):
        """Test system initialization in simulation mode."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000,
                'symbols': ['AAPL'],
                'update_interval_seconds': 60,
                'enable_logging': True,
                'log_level': 'INFO'
            },
            'brokerage': {
                'provider': 'simulated',
                'credentials': {},
                'timeout_seconds': 30,
                'max_retries': 3
            },
            'data': {
                'provider': 'yfinance',
                'cache_enabled': True,
                'cache_dir': str(tmp_path / 'cache'),
                'cache_ttl_seconds': 3600,
                'max_retries': 3
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {
                        'fast_period': 10,
                        'slow_period': 30,
                        'symbols': ['AAPL']
                    }
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        system = TradingSystem(str(config_file))
        system.initialize()
        
        assert system.is_initialized()
        assert not system.is_running()
        assert not system.is_live_mode()
        assert system.config.trading.mode == TradingMode.SIMULATION
        assert system.data_provider is not None
        assert system.brokerage is not None
        assert system.portfolio is not None
        assert system.risk_manager is not None
        assert system.order_manager is not None
        assert len(system.strategies) == 1
        assert system.portfolio.cash_balance == 100000
    
    def test_system_initialization_with_invalid_config(self, tmp_path):
        """Test system initialization with invalid configuration."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': -1000  # Invalid negative capital
            },
            'brokerage': {
                'provider': 'simulated'
            },
            'data': {
                'provider': 'yfinance'
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': []
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        system = TradingSystem(str(config_file))
        
        with pytest.raises(TradingSystemError):
            system.initialize()
    
    def test_system_component_wiring(self, tmp_path):
        """Test that all components are properly wired together."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 50000,
                'symbols': ['AAPL', 'GOOGL'],
                'log_level': 'WARNING'
            },
            'brokerage': {
                'provider': 'simulated'
            },
            'data': {
                'provider': 'yfinance',
                'cache_enabled': False
            },
            'risk': {
                'max_position_size_pct': 15.0,
                'max_daily_loss_pct': 3.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {
                        'fast_period': 5,
                        'slow_period': 20
                    }
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        system = TradingSystem(str(config_file))
        system.initialize()
        
        # Verify components are wired correctly
        assert system.order_manager.broker is system.brokerage
        
        # Verify configuration propagated correctly
        assert system.risk_manager.config.max_position_size_pct == 15.0
        assert system.risk_manager.config.max_daily_loss_pct == 3.0
        
        # Verify portfolio initialized with correct capital
        assert system.portfolio.cash_balance == 50000
        assert system.portfolio.initial_value == 50000
    
    def test_system_status(self, tmp_path):
        """Test system status reporting."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000
            },
            'brokerage': {
                'provider': 'simulated'
            },
            'data': {
                'provider': 'yfinance'
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {}
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        system = TradingSystem(str(config_file))
        
        # Before initialization
        status = system.get_status()
        assert status['status'] == 'not_initialized'
        
        # After initialization
        system.initialize()
        status = system.get_status()
        assert status['status'] == 'initialized'
        assert status['mode'] == 'simulation'
        assert status['strategies'] == 1
        assert status['cash_balance'] == 100000
        assert status['positions'] == 0
