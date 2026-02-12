"""Property-based tests for sentiment analysis service."""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from services.sentiment_service import SentimentService, SentimentScore


# Helper strategies for generating test data
@st.composite
def news_text(draw):
    """Generate realistic news article text."""
    # Generate text with varying lengths and content
    sentences = draw(st.lists(
        st.text(
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Po'),
                min_codepoint=32,
                max_codepoint=126
            ),
            min_size=10,
            max_size=100
        ),
        min_size=1,
        max_size=10
    ))
    return ' '.join(sentences)


class TestSentimentServiceProperties:
    """Property-based tests for SentimentService"""
    
    @given(text=news_text())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
    )
    def test_property_sentiment_score_bounds(self, text):
        """
        Feature: ai-trading-agent, Property 3: Sentiment score bounds
        
        For any news article analyzed, the sentiment score should be a float 
        between -1.0 (most negative) and 1.0 (most positive).
        
        Validates: Requirements 3.2
        """
        # Initialize sentiment service
        service = SentimentService(news_api_key='test_key')
        
        # Mock the FinBERT model to avoid loading actual model
        # We'll simulate realistic model outputs
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Generate realistic sentiment predictions
        # FinBERT outputs probabilities for [negative, neutral, positive]
        import numpy as np
        probabilities = np.random.dirichlet([1, 1, 1])  # Random probabilities that sum to 1
        
        # Create mock outputs
        mock_logits = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        
        # Mock softmax to return our probabilities
        import torch
        with patch('torch.nn.functional.softmax') as mock_softmax:
            mock_softmax.return_value = torch.tensor([probabilities])
            
            # Mock tokenizer
            mock_tokenizer.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            # Mock model
            mock_model.return_value = mock_outputs
            mock_model.eval = MagicMock()
            
            # Inject mocked components
            service.tokenizer = mock_tokenizer
            service.model = mock_model
            service._model_loaded = True
            
            # Analyze sentiment
            result = service.analyze_sentiment(text, use_cache=False)
            
            # Property 1: Result must be a SentimentScore object
            assert isinstance(result, SentimentScore), (
                f"Expected SentimentScore object, got {type(result)}"
            )
            
            # Property 2: Score must be a float
            assert isinstance(result.score, float), (
                f"Sentiment score must be a float, got {type(result.score)}"
            )
            
            # Property 3: Score must be between -1.0 and 1.0 (inclusive)
            assert -1.0 <= result.score <= 1.0, (
                f"Sentiment score {result.score} is out of bounds [-1.0, 1.0]"
            )
            
            # Property 4: Score must not be NaN or infinite
            assert not np.isnan(result.score), (
                "Sentiment score must not be NaN"
            )
            assert not np.isinf(result.score), (
                "Sentiment score must not be infinite"
            )
            
            # Property 5: Label must be one of the valid sentiment labels
            valid_labels = ['negative', 'neutral', 'positive']
            assert result.label in valid_labels, (
                f"Invalid sentiment label '{result.label}', "
                f"must be one of {valid_labels}"
            )
            
            # Property 6: Confidence must be between 0.0 and 1.0
            assert 0.0 <= result.confidence <= 1.0, (
                f"Confidence {result.confidence} is out of bounds [0.0, 1.0]"
            )
            
            # Property 7: Raw scores must contain all three sentiment categories
            assert 'negative' in result.raw_scores, (
                "Raw scores must contain 'negative' category"
            )
            assert 'neutral' in result.raw_scores, (
                "Raw scores must contain 'neutral' category"
            )
            assert 'positive' in result.raw_scores, (
                "Raw scores must contain 'positive' category"
            )
            
            # Property 8: Raw scores must sum to approximately 1.0 (probabilities)
            total_prob = sum(result.raw_scores.values())
            assert 0.99 <= total_prob <= 1.01, (
                f"Raw scores must sum to ~1.0, got {total_prob}"
            )
            
            # Property 9: Each raw score must be between 0.0 and 1.0
            for label, score in result.raw_scores.items():
                assert 0.0 <= score <= 1.0, (
                    f"Raw score for '{label}' ({score}) is out of bounds [0.0, 1.0]"
                )
            
            # Property 10: Sentiment score should be consistent with raw scores
            # Score = -1.0 * negative + 0.0 * neutral + 1.0 * positive
            expected_score = (
                -1.0 * result.raw_scores['negative'] +
                0.0 * result.raw_scores['neutral'] +
                1.0 * result.raw_scores['positive']
            )
            assert abs(result.score - expected_score) < 0.01, (
                f"Sentiment score {result.score} is inconsistent with raw scores "
                f"(expected {expected_score})"
            )
            
            # Property 11: Label should match the highest probability
            max_label = max(result.raw_scores, key=result.raw_scores.get)
            assert result.label == max_label, (
                f"Label '{result.label}' does not match highest probability label '{max_label}'"
            )
            
            # Property 12: Confidence should equal the highest probability
            max_prob = max(result.raw_scores.values())
            assert abs(result.confidence - max_prob) < 0.01, (
                f"Confidence {result.confidence} does not match max probability {max_prob}"
            )


class TestSentimentServiceUnitTests:
    """Unit tests for SentimentService - testing specific functionality."""
    
    # ========== News Fetching Tests ==========
    
    def test_get_news_with_mocked_api_success(self):
        """Test news fetching with successful API response."""
        service = SentimentService(news_api_key='test_key')
        
        # Mock successful API response
        mock_response = {
            'status': 'ok',
            'articles': [
                {
                    'title': 'Apple announces new product',
                    'description': 'Apple Inc. unveiled a new product line today.',
                    'url': 'https://example.com/article1',
                    'source': {'name': 'TechNews'},
                    'publishedAt': '2024-01-15T10:00:00Z',
                    'content': 'Full article content here...'
                },
                {
                    'title': 'Apple stock rises',
                    'description': 'Shares of Apple increased by 5% today.',
                    'url': 'https://example.com/article2',
                    'source': {'name': 'FinanceDaily'},
                    'publishedAt': '2024-01-14T15:30:00Z',
                    'content': None
                }
            ]
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response
            
            articles = service.get_news('AAPL', days=7, use_cache=False)
            
            # Verify results
            assert len(articles) == 2
            assert articles[0].title == 'Apple announces new product'
            assert articles[0].source == 'TechNews'
            assert articles[1].title == 'Apple stock rises'
            assert articles[1].source == 'FinanceDaily'
            
            # Verify API was called correctly
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert 'AAPL' in call_args[1]['params']['q']
            assert call_args[1]['params']['apiKey'] == 'test_key'
    
    def test_get_news_with_empty_results(self):
        """Test news fetching when no articles are found."""
        service = SentimentService(news_api_key='test_key')
        
        mock_response = {
            'status': 'ok',
            'articles': []
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response
            
            articles = service.get_news('UNKNOWN', days=7, use_cache=False)
            
            assert len(articles) == 0
    
    def test_get_news_with_invalid_api_key(self):
        """Test news fetching with invalid API key."""
        service = SentimentService(news_api_key='invalid_key')
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 401
            
            with pytest.raises(ValueError, match="Invalid NewsAPI key"):
                service.get_news('AAPL', days=7, use_cache=False)
    
    def test_get_news_with_rate_limit_exceeded(self):
        """Test news fetching when rate limit is exceeded."""
        service = SentimentService(news_api_key='test_key')
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 429
            
            with pytest.raises(Exception, match="rate limit exceeded"):
                service.get_news('AAPL', days=7, use_cache=False)
    
    def test_get_news_with_invalid_symbol(self):
        """Test news fetching with invalid symbol."""
        service = SentimentService(news_api_key='test_key')
        
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            service.get_news('', days=7)
        
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            service.get_news(None, days=7)
    
    def test_get_news_with_invalid_days(self):
        """Test news fetching with invalid days parameter."""
        service = SentimentService(news_api_key='test_key')
        
        with pytest.raises(ValueError, match="Days must be between 1 and 30"):
            service.get_news('AAPL', days=0)
        
        with pytest.raises(ValueError, match="Days must be between 1 and 30"):
            service.get_news('AAPL', days=31)
    
    def test_get_news_without_api_key(self):
        """Test news fetching without API key returns empty list."""
        service = SentimentService(news_api_key=None)
        
        articles = service.get_news('AAPL', days=7)
        
        assert len(articles) == 0
    
    def test_get_news_caching(self):
        """Test that news results are cached properly."""
        service = SentimentService(news_api_key='test_key')
        
        mock_response = {
            'status': 'ok',
            'articles': [
                {
                    'title': 'Test article',
                    'description': 'Test description',
                    'url': 'https://example.com/test',
                    'source': {'name': 'TestSource'},
                    'publishedAt': '2024-01-15T10:00:00Z'
                }
            ]
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response
            
            # First call - should hit API
            articles1 = service.get_news('AAPL', days=7, use_cache=True)
            assert len(articles1) == 1
            assert mock_get.call_count == 1
            
            # Second call - should use cache
            articles2 = service.get_news('AAPL', days=7, use_cache=True)
            assert len(articles2) == 1
            assert mock_get.call_count == 1  # No additional API call
            
            # Third call with use_cache=False - should hit API again
            articles3 = service.get_news('AAPL', days=7, use_cache=False)
            assert len(articles3) == 1
            assert mock_get.call_count == 2
    
    # ========== Sentiment Analysis Tests ==========
    
    def test_sentiment_analysis_with_empty_text_raises_error(self):
        """Test that sentiment analysis fails with empty text."""
        service = SentimentService(news_api_key='test_key')
        
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            service.analyze_sentiment('')
    
    def test_sentiment_analysis_with_none_text_raises_error(self):
        """Test that sentiment analysis fails with None text."""
        service = SentimentService(news_api_key='test_key')
        
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            service.analyze_sentiment(None)
    
    def test_sentiment_analysis_with_positive_text(self):
        """Test sentiment analysis with clearly positive text."""
        service = SentimentService(news_api_key='test_key')
        
        # Mock FinBERT model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        import torch
        import numpy as np
        
        # Positive sentiment: [0.05, 0.15, 0.80]
        probabilities = np.array([0.05, 0.15, 0.80])
        
        mock_logits = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        
        with patch('torch.nn.functional.softmax') as mock_softmax:
            mock_softmax.return_value = torch.tensor([probabilities])
            
            mock_tokenizer.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            mock_model.return_value = mock_outputs
            mock_model.eval = MagicMock()
            
            service.tokenizer = mock_tokenizer
            service.model = mock_model
            service._model_loaded = True
            
            result = service.analyze_sentiment(
                "Excellent quarterly earnings beat expectations with strong growth",
                use_cache=False
            )
            
            assert result.label == 'positive'
            assert result.score > 0.5
            assert result.confidence == 0.80
            assert 'positive' in result.raw_scores
            assert 'negative' in result.raw_scores
            assert 'neutral' in result.raw_scores
    
    def test_sentiment_analysis_with_negative_text(self):
        """Test sentiment analysis with clearly negative text."""
        service = SentimentService(news_api_key='test_key')
        
        # Mock FinBERT model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        import torch
        import numpy as np
        
        # Negative sentiment: [0.85, 0.10, 0.05]
        probabilities = np.array([0.85, 0.10, 0.05])
        
        mock_logits = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        
        with patch('torch.nn.functional.softmax') as mock_softmax:
            mock_softmax.return_value = torch.tensor([probabilities])
            
            mock_tokenizer.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            mock_model.return_value = mock_outputs
            mock_model.eval = MagicMock()
            
            service.tokenizer = mock_tokenizer
            service.model = mock_model
            service._model_loaded = True
            
            result = service.analyze_sentiment(
                "Company faces bankruptcy amid massive losses and declining revenue",
                use_cache=False
            )
            
            assert result.label == 'negative'
            assert result.score < -0.5
            assert result.confidence == 0.85
    
    def test_sentiment_analysis_caching(self):
        """Test that sentiment analysis results are cached."""
        service = SentimentService(news_api_key='test_key')
        
        # Mock FinBERT model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        import torch
        import numpy as np
        
        probabilities = np.array([0.1, 0.8, 0.1])
        
        mock_logits = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        
        with patch('torch.nn.functional.softmax') as mock_softmax:
            mock_softmax.return_value = torch.tensor([probabilities])
            
            mock_tokenizer.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            mock_model.return_value = mock_outputs
            mock_model.eval = MagicMock()
            
            service.tokenizer = mock_tokenizer
            service.model = mock_model
            service._model_loaded = True
            
            text = "Company reports quarterly results"
            
            # First call - should run model
            result1 = service.analyze_sentiment(text, use_cache=True)
            assert mock_model.call_count == 1
            
            # Second call - should use cache
            result2 = service.analyze_sentiment(text, use_cache=True)
            assert mock_model.call_count == 1  # No additional model call
            assert result2.score == result1.score
            
            # Third call with use_cache=False - should run model again
            result3 = service.analyze_sentiment(text, use_cache=False)
            assert mock_model.call_count == 2
    
    # ========== Aggregate Scoring Tests ==========
    
    def test_get_aggregate_sentiment_with_multiple_articles(self):
        """Test aggregate sentiment calculation with multiple articles."""
        service = SentimentService(news_api_key='test_key')
        
        # Mock news articles
        mock_news_response = {
            'status': 'ok',
            'articles': [
                {
                    'title': 'Positive news',
                    'description': 'Great results',
                    'url': 'https://example.com/1',
                    'source': {'name': 'Source1'},
                    'publishedAt': '2024-01-15T10:00:00Z'
                },
                {
                    'title': 'Negative news',
                    'description': 'Poor performance',
                    'url': 'https://example.com/2',
                    'source': {'name': 'Source2'},
                    'publishedAt': '2024-01-14T10:00:00Z'
                },
                {
                    'title': 'Neutral news',
                    'description': 'Company reports',
                    'url': 'https://example.com/3',
                    'source': {'name': 'Source3'},
                    'publishedAt': '2024-01-13T10:00:00Z'
                }
            ]
        }
        
        # Mock sentiment model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        import torch
        import numpy as np
        
        # Different sentiments for each article
        sentiments = [
            np.array([0.1, 0.2, 0.7]),  # Positive
            np.array([0.7, 0.2, 0.1]),  # Negative
            np.array([0.2, 0.6, 0.2])   # Neutral
        ]
        
        call_count = [0]
        
        def mock_softmax_side_effect(*args, **kwargs):
            result = torch.tensor([sentiments[call_count[0] % len(sentiments)]])
            call_count[0] += 1
            return result
        
        with patch('requests.get') as mock_get, \
             patch('torch.nn.functional.softmax') as mock_softmax:
            
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_news_response
            
            mock_softmax.side_effect = mock_softmax_side_effect
            
            mock_tokenizer.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            mock_model.return_value = MagicMock(logits=MagicMock())
            mock_model.eval = MagicMock()
            
            service.tokenizer = mock_tokenizer
            service.model = mock_model
            service._model_loaded = True
            
            aggregate_score = service.get_aggregate_sentiment('AAPL', days=7, use_cache=False)
            
            # Should be a weighted average (more recent articles weighted higher)
            assert isinstance(aggregate_score, float)
            assert -1.0 <= aggregate_score <= 1.0
    
    def test_get_aggregate_sentiment_with_no_articles(self):
        """Test aggregate sentiment when no articles are found."""
        service = SentimentService(news_api_key='test_key')
        
        mock_response = {
            'status': 'ok',
            'articles': []
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response
            
            aggregate_score = service.get_aggregate_sentiment('UNKNOWN', days=7, use_cache=False)
            
            assert aggregate_score == 0.0
    
    def test_get_aggregate_sentiment_with_all_positive(self):
        """Test aggregate sentiment with all positive articles."""
        service = SentimentService(news_api_key='test_key')
        
        mock_news_response = {
            'status': 'ok',
            'articles': [
                {
                    'title': 'Great news',
                    'description': 'Excellent results',
                    'url': 'https://example.com/1',
                    'source': {'name': 'Source1'},
                    'publishedAt': '2024-01-15T10:00:00Z'
                },
                {
                    'title': 'Amazing growth',
                    'description': 'Record profits',
                    'url': 'https://example.com/2',
                    'source': {'name': 'Source2'},
                    'publishedAt': '2024-01-14T10:00:00Z'
                }
            ]
        }
        
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        import torch
        import numpy as np
        
        # All positive sentiment
        probabilities = np.array([0.05, 0.15, 0.80])
        
        with patch('requests.get') as mock_get, \
             patch('torch.nn.functional.softmax') as mock_softmax:
            
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_news_response
            
            mock_softmax.return_value = torch.tensor([probabilities])
            
            mock_tokenizer.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            mock_model.return_value = MagicMock(logits=MagicMock())
            mock_model.eval = MagicMock()
            
            service.tokenizer = mock_tokenizer
            service.model = mock_model
            service._model_loaded = True
            
            aggregate_score = service.get_aggregate_sentiment('AAPL', days=7, use_cache=False)
            
            # Should be strongly positive
            assert aggregate_score > 0.5
    
    def test_get_aggregate_sentiment_with_invalid_symbol(self):
        """Test aggregate sentiment with invalid symbol."""
        service = SentimentService(news_api_key='test_key')
        
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            service.get_aggregate_sentiment('', days=7)
        
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            service.get_aggregate_sentiment(None, days=7)

