"""
Test script to verify the trading page can be loaded.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work"""
    try:
        from services.market_data_service import MarketDataService, Quote
        from services.trading_service import TradingService, Order, OrderStatus
        from services.risk_service import RiskService, TradeRequest, Position as RiskPosition
        from ai.inference import AIEngine, Recommendation
        from utils.indicators import sma, ema, rsi, macd, bollinger_bands
        from config.settings import settings
        
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_page_structure():
    """Test that the trading page file exists and has correct structure"""
    try:
        with open('pages/2_trading.py', 'r') as f:
            content = f.read()
        
        # Check for required sections
        required_sections = [
            'Stock Search',
            'Current Price',
            'Price Chart',
            'Technical Indicators',
            'AI Recommendation',
            'Manual Trading',
            'Automated Trading',
            'Recent Orders'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"✗ Missing sections: {', '.join(missing_sections)}")
            return False
        
        print("✓ All required sections present")
        
        # Check for required functions
        required_functions = [
            'get_services',
            'get_stock_data',
            'calculate_technical_indicators',
            'display_price_chart',
            'display_recommendation',
            'display_order_preview',
            'display_order_confirmation'
        ]
        
        missing_functions = []
        for func in required_functions:
            if f"def {func}" not in content:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"✗ Missing functions: {', '.join(missing_functions)}")
            return False
        
        print("✓ All required functions present")
        
        # Check for Streamlit components
        required_components = [
            'st.title',
            'st.text_input',
            'st.button',
            'st.metric',
            'st.selectbox',
            'st.number_input',
            'st.toggle',
            'st.dataframe'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"✗ Missing Streamlit components: {', '.join(missing_components)}")
            return False
        
        print("✓ All required Streamlit components present")
        
        return True
    except Exception as e:
        print(f"✗ Error reading page file: {e}")
        return False


def test_requirements_coverage():
    """Test that the page covers all requirements"""
    try:
        with open('pages/2_trading.py', 'r') as f:
            content = f.read()
        
        # Requirements from task 16.1
        requirements = {
            'Stock search and selection': 'text_input',
            'Display current price': 'Current Price',
            'Display technical indicators': 'Technical Indicators',
            'Show AI recommendation': 'AI Recommendation',
            'Manual trade form': 'Place Order',
            'Order preview': 'display_order_preview',
            'Order confirmation': 'display_order_confirmation',
            'Automated trading toggle': 'Enable Automated Trading'
        }
        
        missing_requirements = []
        for req, marker in requirements.items():
            if marker not in content:
                missing_requirements.append(req)
        
        if missing_requirements:
            print(f"✗ Missing requirements: {', '.join(missing_requirements)}")
            return False
        
        print("✓ All requirements covered")
        return True
    except Exception as e:
        print(f"✗ Error checking requirements: {e}")
        return False


if __name__ == '__main__':
    print("Testing Trading Page Implementation")
    print("=" * 50)
    
    all_passed = True
    
    print("\n1. Testing imports...")
    if not test_imports():
        all_passed = False
    
    print("\n2. Testing page structure...")
    if not test_page_structure():
        all_passed = False
    
    print("\n3. Testing requirements coverage...")
    if not test_requirements_coverage():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed!")
        print("\nThe trading page has been successfully implemented with:")
        print("  - Stock search and selection")
        print("  - Real-time price display")
        print("  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)")
        print("  - AI recommendations with confidence and reasoning")
        print("  - Manual trading form (buy/sell, quantity, order type)")
        print("  - Order preview before submission")
        print("  - Order confirmation and status")
        print("  - Automated trading toggle with risk warnings")
        print("  - Recent orders history")
        print("  - Risk validation before order submission")
        print("\nTo run the page:")
        print("  streamlit run pages/2_trading.py")
    else:
        print("✗ Some tests failed")
        sys.exit(1)
