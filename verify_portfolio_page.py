"""
Verification script for Portfolio Page

This script verifies that the portfolio page is properly implemented
and all required functionality is present.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_file_exists():
    """Verify that the portfolio page file exists"""
    file_path = "pages/3_portfolio.py"
    
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        return False
    
    logger.info(f"✅ File exists: {file_path}")
    return True


def verify_imports():
    """Verify that all required imports are present"""
    file_path = "pages/3_portfolio.py"
    
    required_imports = [
        "import streamlit as st",
        "import pandas as pd",
        "import plotly.graph_objects as go",
        "from services.portfolio_service import PortfolioService",
        "from services.trading_service import TradingService",
        "from services.market_data_service import MarketDataService"
    ]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    all_present = True
    for imp in required_imports:
        if imp in content:
            logger.info(f"✅ Import present: {imp}")
        else:
            logger.error(f"❌ Import missing: {imp}")
            all_present = False
    
    return all_present


def verify_required_functions():
    """Verify that all required functions are implemented"""
    file_path = "pages/3_portfolio.py"
    
    required_functions = [
        "def get_services():",
        "def format_currency(",
        "def format_percentage(",
        "def get_period_dates(",
        "def display_positions_table(",
        "def display_performance_chart(",
        "def display_performance_metrics(",
        "def display_transaction_history(",
        "def display_allocation_chart(",
        "def display_position_management("
    ]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    all_present = True
    for func in required_functions:
        if func in content:
            logger.info(f"✅ Function present: {func}")
        else:
            logger.error(f"❌ Function missing: {func}")
            all_present = False
    
    return all_present


def verify_required_features():
    """Verify that all required features are implemented"""
    file_path = "pages/3_portfolio.py"
    
    required_features = [
        "# Display all open positions with P&L",
        "display_positions_table",
        "# Portfolio performance chart",
        "display_performance_chart",
        "# Performance metrics",
        "display_performance_metrics",
        "# Transaction history",
        "display_transaction_history",
        "# Position management",
        "display_position_management",
        "# Portfolio allocation",
        "display_allocation_chart",
        "close_position",
        "Export Transactions"
    ]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    all_present = True
    for feature in required_features:
        if feature in content:
            logger.info(f"✅ Feature present: {feature}")
        else:
            logger.error(f"❌ Feature missing: {feature}")
            all_present = False
    
    return all_present


def verify_page_config():
    """Verify page configuration"""
    file_path = "pages/3_portfolio.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("st.set_page_config", "Page configuration"),
        ("page_title=\"Portfolio", "Page title"),
        ("page_icon=\"💼\"", "Page icon"),
        ("layout=\"wide\"", "Wide layout")
    ]
    
    all_present = True
    for check, description in checks:
        if check in content:
            logger.info(f"✅ {description} configured")
        else:
            logger.error(f"❌ {description} missing")
            all_present = False
    
    return all_present


def verify_syntax():
    """Verify Python syntax"""
    file_path = "pages/3_portfolio.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, file_path, 'exec')
        logger.info("✅ Python syntax is valid")
        return True
    except SyntaxError as e:
        logger.error(f"❌ Syntax error: {e}")
        return False


def main():
    """Run all verification checks"""
    logger.info("=" * 60)
    logger.info("Portfolio Page Verification")
    logger.info("=" * 60)
    
    checks = [
        ("File Exists", verify_file_exists),
        ("Imports", verify_imports),
        ("Functions", verify_required_functions),
        ("Features", verify_required_features),
        ("Page Config", verify_page_config),
        ("Syntax", verify_syntax)
    ]
    
    results = {}
    for name, check_func in checks:
        logger.info(f"\n--- Checking: {name} ---")
        results[name] = check_func()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Verification Summary")
    logger.info("=" * 60)
    
    all_passed = True
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
        if not result:
            all_passed = False
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("✅ All checks passed!")
        logger.info("\nTo run the portfolio page:")
        logger.info("  streamlit run pages/3_portfolio.py")
        return 0
    else:
        logger.error("❌ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
