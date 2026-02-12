"""Manual test script for sentiment service"""

import os
from services.sentiment_service import SentimentService

# Set a dummy API key for testing (won't fetch real news without valid key)
os.environ['NEWS_API_KEY'] = 'test_key'

def test_sentiment_service_initialization():
    """Test that sentiment service can be initialized"""
    try:
        service = SentimentService(news_api_key='test_key')
        print("✓ Sentiment service initialized successfully")
        print(f"  - Model name: {service.model_name}")
        print(f"  - Cache TTL: {service.cache_ttl} seconds")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize sentiment service: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis on sample text"""
    try:
        service = SentimentService(news_api_key='test_key')
        
        # Test positive sentiment
        positive_text = "The company reported record profits and strong growth prospects"
        result = service.analyze_sentiment(positive_text)
        print(f"\n✓ Analyzed positive text:")
        print(f"  - Score: {result.score:.3f}")
        print(f"  - Label: {result.label}")
        print(f"  - Confidence: {result.confidence:.3f}")
        
        # Test negative sentiment
        negative_text = "The company faces bankruptcy and massive layoffs"
        result = service.analyze_sentiment(negative_text)
        print(f"\n✓ Analyzed negative text:")
        print(f"  - Score: {result.score:.3f}")
        print(f"  - Label: {result.label}")
        print(f"  - Confidence: {result.confidence:.3f}")
        
        return True
    except Exception as e:
        print(f"\n✗ Failed sentiment analysis: {e}")
        return False

def test_cache_functionality():
    """Test caching functionality"""
    try:
        service = SentimentService(news_api_key='test_key')
        
        # Get initial cache stats
        stats = service.get_cache_stats()
        print(f"\n✓ Cache stats:")
        print(f"  - News cache size: {stats['news_cache_size']}")
        print(f"  - Sentiment cache size: {stats['sentiment_cache_size']}")
        print(f"  - Model loaded: {stats['model_loaded']}")
        
        return True
    except Exception as e:
        print(f"\n✗ Failed cache test: {e}")
        return False

if __name__ == "__main__":
    print("Testing Sentiment Service Implementation\n")
    print("=" * 50)
    
    # Run tests
    test_sentiment_service_initialization()
    
    print("\n" + "=" * 50)
    print("\nNote: Full sentiment analysis test requires transformers library")
    print("Run: python -m pip install transformers torch")
    print("\nTo test with real news, set NEWS_API_KEY in .env.ai")
    print("Get your key from: https://newsapi.org/")
