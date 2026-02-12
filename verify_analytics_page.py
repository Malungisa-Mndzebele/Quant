"""Quick verification script for analytics page"""

import sys
import ast

def check_syntax(filepath):
    """Check if Python file has valid syntax"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"✓ {filepath} has valid syntax")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in {filepath}:")
        print(f"  Line {e.lineno}: {e.msg}")
        print(f"  {e.text}")
        return False
    except Exception as e:
        print(f"✗ Error checking {filepath}: {e}")
        return False

def check_required_functions(filepath):
    """Check if required functions are defined"""
    required_functions = [
        'get_services',
        'load_model_metrics',
        'create_accuracy_over_time_chart',
        'create_feature_importance_chart',
        'create_confidence_distribution_chart',
        'create_confusion_matrix_chart',
        'create_sentiment_trend_chart',
        'create_model_comparison_chart',
        'generate_sample_predictions',
        'generate_sample_sentiment_data'
    ]
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        defined_functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        
        missing = set(required_functions) - defined_functions
        if missing:
            print(f"✗ Missing functions: {missing}")
            return False
        else:
            print(f"✓ All required functions are defined")
            return True
    except Exception as e:
        print(f"✗ Error checking functions: {e}")
        return False

def main():
    """Run verification"""
    print("Verifying Analytics Page (pages/4_analytics.py)")
    print("=" * 60)
    
    filepath = "pages/4_analytics.py"
    
    all_passed = True
    
    # Check syntax
    if not check_syntax(filepath):
        all_passed = False
    
    # Check required functions
    if not check_required_functions(filepath):
        all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ Analytics page verification passed!")
        print("\n📊 Analytics Page Features:")
        print("  - AI model performance metrics display")
        print("  - Prediction accuracy over time charts")
        print("  - Feature importance visualization")
        print("  - Model confidence distribution")
        print("  - Confusion matrix heatmap")
        print("  - Recent prediction explanations")
        print("  - Sentiment analysis trends")
        print("\n🚀 To run the analytics page:")
        print("  streamlit run pages/4_analytics.py")
        return 0
    else:
        print("✗ Verification failed")
        return 1

if __name__ == "__main__":
    exit(main())
