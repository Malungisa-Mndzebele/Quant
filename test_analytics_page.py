"""Test script to verify analytics page can be imported and basic functions work"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_analytics_imports():
    """Test that analytics page can be imported"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("analytics", "pages/4_analytics.py")
        analytics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analytics)
        print("✓ Analytics page imports successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import analytics page: {e}")
        return False

def test_helper_functions():
    """Test helper functions"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("analytics", "pages/4_analytics.py")
        analytics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analytics)
        
        # Test sample data generation
        predictions_df = analytics.generate_sample_predictions(days=30)
        assert len(predictions_df) == 30, "Should generate 30 days of predictions"
        assert 'predicted' in predictions_df.columns, "Should have predicted column"
        assert 'actual' in predictions_df.columns, "Should have actual column"
        assert 'confidence' in predictions_df.columns, "Should have confidence column"
        print("✓ Sample predictions generation works")
        
        # Test sentiment data generation
        sentiment_data = analytics.generate_sample_sentiment_data(days=30)
        assert len(sentiment_data) == 30, "Should generate 30 days of sentiment"
        assert 'sentiment_score' in sentiment_data[0], "Should have sentiment_score"
        print("✓ Sample sentiment generation works")
        
        return True
    except Exception as e:
        print(f"✗ Helper function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing Analytics Page (pages/4_analytics.py)")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_analytics_imports():
        all_passed = False
    
    # Test helper functions
    if not test_helper_functions():
        all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
