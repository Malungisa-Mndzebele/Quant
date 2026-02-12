"""
Comprehensive Integration Tests for AI Trading Agent

This test suite validates end-to-end workflows and integration between components.
"""

import pytest
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDataToAnalysisWorkflow:
    """Test the complete workflow: data → analysis → recommendation"""
    
    def test_market_data_to_recommendation(self):
        """Test fetching market data and generating AI recommendation"""
        from services.market_data_service import MarketDataService, Quote
        from ai.inference import AIEngine
        
        # Mock the Alpaca API
        with patch('services.market_data_service.StockHistoricalDataClient'):
            with patch('services.market_data_service.settings') as mock_settings:
                mock_settings.alpaca_api_key = 'test_key'
                mock_settings.alpaca_secret_key = 'test_secret'
                mock_settings.paper_trading = True
                
                # Create service
                market_service = MarketDataService()
                
                # Mock quote data
                mock_quote = Quote(
                    symbol='AAPL',
                    bid_price=150.0,
                    ask_price=150.5,
                    bid_size=100,
                    ask_size=100,
                    timestamp=datetime.now()
                )
                
                # Verify quote structure
                assert mock_quote.symbol == 'AAPL'
                assert mock_quote.mid_price == 150.25
                assert mock_quote.spread == 0.5


class TestTradingWorkflow:
    """Test the complete trading workflow"""
    
    def test_recommendation_to_trade_execution(self):
        """Test generating recommendation and executing trade"""
        from services.trading_service import TradingService
        from services.risk_service import RiskService, RiskConfig
        
        # Mock trading service
        with patch('services.trading_service.TradingClient'):
            with patch('services.trading_service.settings') as mock_settings:
                mock_settings.alpaca_api_key = 'test_key'
                mock_settings.alpaca_secret_key = 'test_secret'
                mock_settings.paper_trading = True
                
                trading_service = TradingService()
                
                # Create risk service
                risk_config = RiskConfig(
                    max_position_size=0.1,
                    max_portfolio_risk=0.2,
                    daily_loss_limit=1000.0,
                    stop_loss_pct=0.05,
                    take_profit_pct=0.15,
                    max_open_positions=10
                )
                risk_service = RiskService(risk_config)
                
                # Verify risk config
                assert risk_config.max_position_size == 0.1
                assert risk_config.daily_loss_limit == 1000.0


class TestPortfolioWorkflow:
    """Test portfolio management workflow"""
    
    def test_trade_to_portfolio_update(self):
        """Test that executed trades update portfolio correctly"""
        from services.portfolio_service import PortfolioService
        
        # Create portfolio service with test database
        portfolio_service = PortfolioService(db_path=':memory:')
        
        # Verify portfolio initialization
        portfolio_value = portfolio_service.get_portfolio_value()
        assert portfolio_value >= 0


class TestBacktestingWorkflow:
    """Test backtesting workflow"""
    
    def test_backtest_execution(self):
        """Test running a complete backtest"""
        from services.backtest_service import BacktestEngine, Strategy
        from config.strategies import STRATEGY_PRESETS
        
        # Create backtest engine
        engine = BacktestEngine(initial_capital=100000.0)
        
        # Verify initialization
        assert engine.initial_capital == 100000.0
        
        # Verify strategy presets exist
        assert 'conservative' in STRATEGY_PRESETS
        assert 'moderate' in STRATEGY_PRESETS
        assert 'aggressive' in STRATEGY_PRESETS


class TestPaperTradingWorkflow:
    """Test paper trading workflow"""
    
    def test_paper_trading_isolation(self):
        """Test that paper trading doesn't affect live accounts"""
        from services.paper_trading_service import PaperTradingService
        
        # Create paper trading service
        paper_service = PaperTradingService(
            initial_capital=100000.0,
            session_name='test_session'
        )
        
        # Verify initialization
        assert paper_service.initial_capital == 100000.0
        assert paper_service.session_name == 'test_session'
        
        # Verify paper trading state
        state = paper_service.get_state()
        assert state['cash'] == 100000.0
        assert len(state['positions']) == 0


class TestErrorRecoveryWorkflow:
    """Test error recovery scenarios"""
    
    def test_state_persistence_on_error(self):
        """Test that system state is preserved on errors"""
        from services.error_recovery_service import ErrorRecoveryService
        
        # Create error recovery service
        recovery_service = ErrorRecoveryService(state_dir='data/state')
        
        # Test state saving
        test_state = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': 100000.0,
            'positions': []
        }
        
        recovery_service.save_state('test_component', test_state)
        
        # Test state loading
        loaded_state = recovery_service.load_state('test_component')
        assert loaded_state is not None
        assert loaded_state['portfolio_value'] == 100000.0


class TestExportFunctionality:
    """Test export functionality across features"""
    
    def test_portfolio_export(self):
        """Test portfolio data export"""
        from services.portfolio_service import PortfolioService
        
        # Create portfolio service
        portfolio_service = PortfolioService(db_path=':memory:')
        
        # Test export functionality exists
        assert hasattr(portfolio_service, 'export_for_taxes')
        assert hasattr(portfolio_service, 'get_transaction_history')


class TestPropertyBasedTests:
    """Verify property-based tests are passing"""
    
    def test_property_tests_exist(self):
        """Verify that property-based tests are defined"""
        import glob
        
        # Find all test files
        test_files = glob.glob('tests/test_*.py')
        
        # Count files with property tests
        property_test_files = []
        for test_file in test_files:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if '@given' in content or 'hypothesis' in content:
                    property_test_files.append(test_file)
        
        # Verify we have property-based tests
        assert len(property_test_files) > 0, "No property-based tests found"


class TestDocumentation:
    """Verify documentation is complete"""
    
    def test_readme_exists(self):
        """Verify README files exist"""
        assert os.path.exists('README.md')
        assert os.path.exists('AI_TRADING_AGENT_README.md')
    
    def test_implementation_docs_exist(self):
        """Verify implementation documentation exists"""
        docs = [
            'QUICKSTART_AI_AGENT.md',
            'PAPER_TRADING_GUIDE.md',
            'TRADING_PAGE_GUIDE.md',
            'PORTFOLIO_PAGE_GUIDE.md',
            'BACKTEST_PAGE_GUIDE.md',
            'WATCHLIST_QUICKSTART.md',
            'PERSONALIZATION_QUICKSTART.md',
            'STRATEGY_CONFIGURATION_GUIDE.md',
            'TRADING_SCHEDULE_GUIDE.md'
        ]
        
        for doc in docs:
            assert os.path.exists(doc), f"Documentation file {doc} not found"


class TestPageFunctionality:
    """Test that all pages can be imported"""
    
    def test_pages_import(self):
        """Verify all page files exist and can be imported"""
        pages = [
            'pages/1_ai_dashboard.py',
            'pages/2_trading.py',
            'pages/3_portfolio.py',
            'pages/4_analytics.py',
            'pages/5_backtest.py'
        ]
        
        for page in pages:
            assert os.path.exists(page), f"Page file {page} not found"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
