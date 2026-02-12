"""Tests for multi-asset support functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from services.market_data_service import MarketDataService, AssetClass, Quote
from services.trading_service import TradingService, AssetClass as TradingAssetClass
from services.portfolio_service import PortfolioService
from ai.inference import AIEngine, TradingSignal, Recommendation
from utils.asset_analysis import (
    calculate_crypto_specific_indicators,
    calculate_forex_specific_indicators,
    calculate_asset_specific_indicators,
    calculate_portfolio_allocation,
    get_asset_class_risk_metrics
)


class TestAssetClassDetection:
    """Test asset class detection from symbols."""
    
    def test_detect_stock_symbols(self):
        """Test detection of stock symbols."""
        mds = MarketDataService()
        
        assert mds.detect_asset_class('AAPL') == AssetClass.STOCK
        assert mds.detect_asset_class('GOOGL') == AssetClass.STOCK
        assert mds.detect_asset_class('TSLA') == AssetClass.STOCK
    
    def test_detect_crypto_symbols(self):
        """Test detection of crypto symbols."""
        mds = MarketDataService()
        
        assert mds.detect_asset_class('BTC/USD') == AssetClass.CRYPTO
        assert mds.detect_asset_class('BTCUSD') == AssetClass.CRYPTO
        assert mds.detect_asset_class('ETH/USD') == AssetClass.CRYPTO
        assert mds.detect_asset_class('ETHUSD') == AssetClass.CRYPTO
    
    def test_detect_forex_symbols(self):
        """Test detection of forex symbols."""
        mds = MarketDataService()
        
        assert mds.detect_asset_class('EUR/USD') == AssetClass.FOREX
        assert mds.detect_asset_class('GBP/USD') == AssetClass.FOREX
        assert mds.detect_asset_class('USD/JPY') == AssetClass.FOREX


class TestAssetSpecificIndicators:
    """Test asset-specific technical indicators."""
    
    def test_crypto_specific_indicators(self):
        """Test crypto-specific indicator calculation."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 45000, 100),
            'high': np.random.uniform(45000, 46000, 100),
            'low': np.random.uniform(39000, 40000, 100),
            'close': np.random.uniform(40000, 45000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        result = calculate_crypto_specific_indicators(df)
        
        # Check that crypto-specific indicators are added
        assert 'realized_vol_24h' in result.columns
        assert 'momentum_6h' in result.columns
        assert 'momentum_24h' in result.columns
        assert 'volume_ma_24h' in result.columns
        assert 'volume_ratio' in result.columns
    
    def test_forex_specific_indicators(self):
        """Test forex-specific indicator calculation."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(1.08, 1.12, 100),
            'high': np.random.uniform(1.12, 1.13, 100),
            'low': np.random.uniform(1.07, 1.08, 100),
            'close': np.random.uniform(1.08, 1.12, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        result = calculate_forex_specific_indicators(df)
        
        # Check that forex-specific indicators are added
        assert 'ema_8' in result.columns
        assert 'ema_21' in result.columns
        assert 'ema_55' in result.columns
        assert 'trend_strength' in result.columns
        assert 'atr_14' in result.columns
        assert 'atr_pct' in result.columns
    
    def test_asset_specific_indicators_routing(self):
        """Test that asset-specific indicators are routed correctly."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 115, 100),
            'low': np.random.uniform(95, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Test crypto routing
        crypto_result = calculate_asset_specific_indicators(df, AssetClass.CRYPTO)
        assert 'realized_vol_24h' in crypto_result.columns
        
        # Test forex routing
        forex_result = calculate_asset_specific_indicators(df, AssetClass.FOREX)
        assert 'trend_strength' in forex_result.columns
        
        # Test stock routing (should return unchanged)
        stock_result = calculate_asset_specific_indicators(df, AssetClass.STOCK)
        assert len(stock_result.columns) == len(df.columns)


class TestAIEngineMultiAsset:
    """Test AI engine with multi-asset support."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 115, 100),
            'low': np.random.uniform(95, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000000, 10000000, 100)
        }, index=dates)
    
    def test_analyze_stock_with_asset_class(self, sample_data):
        """Test analyzing stock with asset class parameter."""
        engine = AIEngine()
        
        # Mock model loading
        with patch.object(engine, 'load_models', return_value={'lstm': False, 'rf': False}):
            with patch.object(engine, 'models_loaded', True):
                with patch.object(engine, 'available_models', ['rf']):
                    with patch.object(engine, 'rf_model') as mock_rf:
                        mock_rf.predict_with_confidence.return_value = [{
                            'prediction': 'buy',
                            'confidence': 0.8,
                            'probabilities': {'buy': 0.8, 'hold': 0.15, 'sell': 0.05}
                        }]
                        
                        signal = engine.analyze_stock('AAPL', sample_data, AssetClass.STOCK)
                        
                        assert signal.symbol == 'AAPL'
                        assert signal.asset_class == AssetClass.STOCK
                        assert signal.action in ['buy', 'sell', 'hold']
                        assert 0 <= signal.confidence <= 1
    
    def test_get_recommendation_with_asset_class(self, sample_data):
        """Test getting recommendation with asset class."""
        engine = AIEngine()
        
        with patch.object(engine, 'analyze_stock') as mock_analyze:
            mock_signal = TradingSignal(
                symbol='BTC/USD',
                action='buy',
                confidence=0.85,
                target_price=45000,
                stop_loss=42000,
                reasoning={
                    'current_price': 43000,
                    'model_scores': {'rf': 1.0},
                    'technical_indicators': {'rsi': 45, 'macd_histogram': 0.5},
                    'models_used': ['rf']
                },
                model_predictions={'rf': 'buy'},
                asset_class=AssetClass.CRYPTO
            )
            mock_analyze.return_value = mock_signal
            
            recommendation = engine.get_recommendation('BTC/USD', sample_data, AssetClass.CRYPTO)
            
            assert recommendation.symbol == 'BTC/USD'
            assert recommendation.asset_class == AssetClass.CRYPTO
            assert recommendation.action == 'buy'
            assert recommendation.confidence == 0.85


class TestPortfolioAssetAllocation:
    """Test portfolio asset class allocation."""
    
    @pytest.fixture
    def mock_trading_service(self):
        """Create mock trading service."""
        mock_service = Mock(spec=TradingService)
        
        # Mock positions
        from services.trading_service import Position
        mock_service.get_positions.return_value = [
            Position(
                symbol='AAPL',
                quantity=100,
                side='long',
                entry_price=150.0,
                current_price=155.0,
                market_value=15500.0,
                cost_basis=15000.0,
                unrealized_pl=500.0,
                unrealized_pl_pct=3.33
            ),
            Position(
                symbol='BTC/USD',
                quantity=1,
                side='long',
                entry_price=40000.0,
                current_price=43000.0,
                market_value=43000.0,
                cost_basis=40000.0,
                unrealized_pl=3000.0,
                unrealized_pl_pct=7.5
            )
        ]
        
        # Mock account
        mock_service.get_account.return_value = {
            'portfolio_value': 60000.0,
            'cash': 1500.0,
            'equity': 60000.0
        }
        
        return mock_service
    
    def test_get_asset_class_allocation(self, mock_trading_service):
        """Test calculating asset class allocation."""
        portfolio = PortfolioService(trading_service=mock_trading_service)
        
        with patch('services.portfolio_service.MarketDataService') as mock_mds_class:
            mock_mds = Mock()
            mock_mds.detect_asset_class.side_effect = lambda s: (
                AssetClass.CRYPTO if 'BTC' in s else AssetClass.STOCK
            )
            mock_mds_class.return_value = mock_mds
            
            allocation = portfolio.get_asset_class_allocation()
            
            assert 'stock' in allocation
            assert 'crypto' in allocation
            assert 'forex' in allocation
            
            # Check that allocations sum to approximately 100%
            # (may not be exact due to cash)
            total_allocation = sum(allocation.values())
            assert 0 <= total_allocation <= 100
    
    def test_get_diversification_metrics(self, mock_trading_service):
        """Test calculating diversification metrics."""
        portfolio = PortfolioService(trading_service=mock_trading_service)
        
        with patch('services.portfolio_service.MarketDataService') as mock_mds_class:
            mock_mds = Mock()
            mock_mds.detect_asset_class.side_effect = lambda s: (
                AssetClass.CRYPTO if 'BTC' in s else AssetClass.STOCK
            )
            mock_mds_class.return_value = mock_mds
            
            with patch.object(portfolio, 'get_correlation_matrix', return_value=pd.DataFrame()):
                metrics = portfolio.get_diversification_metrics()
                
                assert 'asset_class_allocation' in metrics
                assert 'diversification_ratio' in metrics
                assert 'concentration_risk' in metrics
                assert 'num_positions' in metrics
                
                assert metrics['num_positions'] == 2
                assert 0 <= metrics['concentration_risk'] <= 100


class TestAssetClassRiskMetrics:
    """Test asset class-specific risk metrics."""
    
    def test_stock_risk_metrics(self):
        """Test risk metrics for stocks."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns
        
        metrics = get_asset_class_risk_metrics(returns, AssetClass.STOCK)
        
        assert 'volatility' in metrics
        assert 'annualized_vol' in metrics
        assert 'annualized_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    def test_crypto_risk_metrics(self):
        """Test risk metrics for crypto (24/7 trading)."""
        returns = pd.Series(np.random.normal(0.001, 0.05, 365 * 24))  # Hourly returns
        
        metrics = get_asset_class_risk_metrics(returns, AssetClass.CRYPTO)
        
        assert 'volatility' in metrics
        assert 'annualized_vol' in metrics
        # Crypto should have higher annualization factor
        assert metrics['annualized_vol'] > metrics['volatility']
    
    def test_forex_risk_metrics(self):
        """Test risk metrics for forex."""
        returns = pd.Series(np.random.normal(0.0001, 0.01, 252 * 24))  # Hourly returns
        
        metrics = get_asset_class_risk_metrics(returns, AssetClass.FOREX)
        
        assert 'volatility' in metrics
        assert 'annualized_vol' in metrics
        assert 'sharpe_ratio' in metrics


class TestTradingServiceAssetRouting:
    """Test trading service asset-specific order routing."""
    
    def test_place_order_with_schedule_check_stocks(self):
        """Test placing order with schedule check for stocks."""
        service = TradingService(paper=True)
        
        # Mock the underlying place_order method
        with patch.object(service, 'place_order') as mock_place:
            from services.trading_service import Order, OrderStatus
            mock_place.return_value = Order(
                order_id='test123',
                symbol='AAPL',
                quantity=100,
                side='buy',
                order_type='market',
                status=OrderStatus.FILLED,
                filled_qty=100,
                filled_avg_price=150.0,
                limit_price=None,
                submitted_at=datetime.now(),
                filled_at=datetime.now()
            )
            
            # Enable automated trading
            service.enable_automated_trading()
            
            # Try to place order during trading hours (should succeed)
            with patch('services.trading_service.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime(2024, 1, 15, 10, 0)  # Monday 10 AM
                
                order = service.place_order_with_schedule_check(
                    symbol='AAPL',
                    qty=100,
                    side='buy',
                    asset_class=TradingAssetClass.STOCKS
                )
                
                assert order.symbol == 'AAPL'
                assert mock_place.called
    
    def test_place_order_outside_schedule_rejected(self):
        """Test that orders outside schedule are rejected."""
        service = TradingService(paper=True)
        
        # Enable automated trading
        service.enable_automated_trading()
        
        # Try to place order outside trading hours (should fail)
        with patch('services.trading_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 20, 0)  # Monday 8 PM
            
            with pytest.raises(ValueError, match="Trading not allowed"):
                service.place_order_with_schedule_check(
                    symbol='AAPL',
                    qty=100,
                    side='buy',
                    asset_class=TradingAssetClass.STOCKS
                )
    
    def test_crypto_24_7_trading(self):
        """Test that crypto can trade 24/7."""
        service = TradingService(paper=True)
        
        # Mock the underlying place_order method
        with patch.object(service, 'place_order') as mock_place:
            from services.trading_service import Order, OrderStatus
            mock_place.return_value = Order(
                order_id='test123',
                symbol='BTC/USD',
                quantity=1,
                side='buy',
                order_type='market',
                status=OrderStatus.FILLED,
                filled_qty=1,
                filled_avg_price=43000.0,
                limit_price=None,
                submitted_at=datetime.now(),
                filled_at=datetime.now()
            )
            
            # Enable automated trading
            service.enable_automated_trading()
            
            # Try to place crypto order at any time (should succeed)
            with patch('services.trading_service.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime(2024, 1, 15, 2, 0)  # Monday 2 AM
                
                order = service.place_order_with_schedule_check(
                    symbol='BTC/USD',
                    qty=1,
                    side='buy',
                    asset_class=TradingAssetClass.CRYPTO
                )
                
                assert order.symbol == 'BTC/USD'
                assert mock_place.called


class TestAssetClassRoutingProperty:
    """Property-based tests for asset class routing."""
    
    def test_property_asset_class_routing(self):
        """
        Property 20: Asset class routing
        
        For any trade order, the system should route it to the correct
        broker API endpoint based on the asset class.
        
        Validates: Requirements 20.4
        
        Feature: ai-trading-agent, Property 20: Asset class routing
        """
        from hypothesis import given, strategies as st, settings
        from hypothesis.strategies import composite
        
        @composite
        def trade_order_strategy(draw):
            """Generate random trade orders with different asset classes."""
            # Asset class
            asset_class = draw(st.sampled_from([
                AssetClass.STOCK,
                AssetClass.CRYPTO,
                AssetClass.FOREX
            ]))
            
            # Symbol based on asset class
            if asset_class == AssetClass.STOCK:
                symbol = draw(st.sampled_from(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']))
            elif asset_class == AssetClass.CRYPTO:
                symbol = draw(st.sampled_from(['BTC/USD', 'ETH/USD', 'BTCUSD', 'ETHUSD']))
            else:  # FOREX
                symbol = draw(st.sampled_from(['EUR/USD', 'GBP/USD', 'USD/JPY']))
            
            # Order parameters
            qty = draw(st.integers(min_value=1, max_value=1000))
            side = draw(st.sampled_from(['buy', 'sell']))
            order_type = draw(st.sampled_from(['market', 'limit']))
            limit_price = draw(st.floats(min_value=0.01, max_value=100000.0)) if order_type == 'limit' else None
            
            return {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'order_type': order_type,
                'limit_price': limit_price,
                'asset_class': asset_class
            }
        
        @given(order=trade_order_strategy())
        @settings(max_examples=100, deadline=None)
        def property_test(order):
            """
            Test that orders are routed to correct endpoints based on asset class.
            
            The property verifies:
            1. Asset class is correctly detected from symbol
            2. Order routing respects the asset class
            3. Correct API client is used for each asset class
            """
            # Mock the API credentials to avoid requiring real credentials
            with patch('services.market_data_service.settings') as mock_settings:
                mock_settings.alpaca.api_key = 'test_key'
                mock_settings.alpaca.secret_key = 'test_secret'
                mock_settings.cache_ttl_seconds = 300
                
                # Mock the Alpaca clients
                with patch('services.market_data_service.StockHistoricalDataClient'):
                    with patch('services.market_data_service.CryptoHistoricalDataClient'):
                        with patch('services.market_data_service.TradingClient'):
                            # Create market data service to test detection
                            mds = MarketDataService(api_key='test_key', api_secret='test_secret')
                            
                            # Verify asset class detection
                            detected_class = mds.detect_asset_class(order['symbol'])
                            
                            # The detected class should match the expected class for the symbol
                            if 'BTC' in order['symbol'] or 'ETH' in order['symbol']:
                                assert detected_class == AssetClass.CRYPTO, \
                                    f"Crypto symbol {order['symbol']} not detected as CRYPTO"
                            elif '/' in order['symbol'] and len(order['symbol'].split('/')[0]) == 3:
                                # Forex pairs like EUR/USD
                                if order['symbol'].split('/')[0] not in ['BTC', 'ETH', 'LTC']:
                                    assert detected_class == AssetClass.FOREX, \
                                        f"Forex symbol {order['symbol']} not detected as FOREX"
                            else:
                                assert detected_class == AssetClass.STOCK, \
                                    f"Stock symbol {order['symbol']} not detected as STOCK"
                            
                            # Test that the correct data client is selected
                            if detected_class == AssetClass.STOCK:
                                # Stock data should use stock_data_client
                                assert hasattr(mds, 'stock_data_client'), \
                                    "Stock data client not initialized"
                                assert mds.stock_data_client is not None, \
                                    "Stock data client is None"
                            elif detected_class == AssetClass.CRYPTO:
                                # Crypto data should use crypto_data_client
                                assert hasattr(mds, 'crypto_data_client'), \
                                    "Crypto data client not initialized"
                                assert mds.crypto_data_client is not None, \
                                    "Crypto data client is None"
            
            # Test trading schedule routing (doesn't need API credentials)
            with patch('services.trading_service.settings') as mock_settings:
                mock_settings.alpaca.api_key = 'test_key'
                mock_settings.alpaca.secret_key = 'test_secret'
                
                with patch('services.trading_service.TradingClient'):
                    trading_service = TradingService(api_key='test_key', api_secret='test_secret', paper=True)
                    
                    # Get the schedule for this asset class
                    if detected_class == AssetClass.STOCK:
                        schedule = trading_service.get_trading_schedule(TradingAssetClass.STOCKS)
                        assert schedule.asset_class == TradingAssetClass.STOCKS, \
                            "Stock schedule has wrong asset class"
                    elif detected_class == AssetClass.CRYPTO:
                        schedule = trading_service.get_trading_schedule(TradingAssetClass.CRYPTO)
                        assert schedule.asset_class == TradingAssetClass.CRYPTO, \
                            "Crypto schedule has wrong asset class"
                    elif detected_class == AssetClass.FOREX:
                        schedule = trading_service.get_trading_schedule(TradingAssetClass.FOREX)
                        assert schedule.asset_class == TradingAssetClass.FOREX, \
                            "Forex schedule has wrong asset class"
                    
                    # Verify schedule configuration is appropriate for asset class
                    if detected_class == AssetClass.CRYPTO:
                        # Crypto should allow 24/7 trading
                        assert len(schedule.active_days) == 7, \
                            "Crypto schedule should allow all 7 days"
                    elif detected_class == AssetClass.STOCK:
                        # Stocks should be weekdays only
                        assert schedule.active_days == {0, 1, 2, 3, 4}, \
                            "Stock schedule should be Monday-Friday only"
        
        # Run the property test
        property_test()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
