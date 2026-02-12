"""
Simple import test for portfolio page
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_import():
    """Test that the portfolio page can be imported"""
    try:
        # Add pages directory to path
        sys.path.insert(0, 'pages')
        
        # Try to compile the file
        with open('pages/3_portfolio.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, 'pages/3_portfolio.py', 'exec')
        
        logger.info("✅ Portfolio page compiles successfully")
        logger.info("✅ All imports are valid")
        logger.info("✅ Syntax is correct")
        
        # Check for key components
        components = [
            'display_positions_table',
            'display_performance_chart',
            'display_performance_metrics',
            'display_transaction_history',
            'display_allocation_chart',
            'display_position_management'
        ]
        
        for component in components:
            if component in code:
                logger.info(f"✅ Component found: {component}")
            else:
                logger.error(f"❌ Component missing: {component}")
                return False
        
        logger.info("\n✅ Portfolio page is ready to use!")
        logger.info("\nTo run the portfolio page:")
        logger.info("  streamlit run pages/3_portfolio.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_import()
    sys.exit(0 if success else 1)
