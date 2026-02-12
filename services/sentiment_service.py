"""Sentiment analysis service for news and social media sentiment."""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from functools import lru_cache

import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """News article data"""
    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    content: Optional[str] = None
    sentiment_score: Optional[float] = None


@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    score: float  # -1.0 (negative) to 1.0 (positive)
    label: str  # 'positive', 'negative', or 'neutral'
    confidence: float  # 0.0 to 1.0
    raw_scores: Dict[str, float]  # Raw model outputs


@dataclass
class NewsAlert:
    """Breaking news alert"""
    symbol: str
    headline: str
    sentiment: str
    impact_score: float  # 0.0 to 1.0
    timestamp: datetime
    url: str


class SentimentService:
    """
    Service for analyzing news sentiment using FinBERT.
    
    Fetches news articles from NewsAPI and performs sentiment analysis
    using the FinBERT model fine-tuned for financial text.
    """
    
    def __init__(self, news_api_key: Optional[str] = None):
        """
        Initialize sentiment service.
        
        Args:
            news_api_key: NewsAPI key (defaults to settings)
        """
        self.news_api_key = news_api_key or settings.news_api_key
        
        if not self.news_api_key:
            logger.warning("NewsAPI key not provided - news fetching will be disabled")
        
        # NewsAPI configuration
        self.news_api_base_url = "https://newsapi.org/v2"
        
        # Initialize FinBERT model for sentiment analysis
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self._model_loaded = False
        
        # Cache for news articles and sentiment scores
        self._news_cache: Dict[str, tuple[List[NewsArticle], float]] = {}
        self._sentiment_cache: Dict[str, tuple[SentimentScore, float]] = {}
        
        # Cache TTL in seconds
        self.cache_ttl = settings.cache_ttl_seconds
        
        # Rate limiting for NewsAPI (500 requests per day on free tier)
        self._last_request_time = 0
        self._min_request_interval = 0.2  # 200ms between requests
        
        # Retry configuration
        self._max_retries = 3
        self._retry_delay = 1.0  # seconds
        
        logger.info("Initialized sentiment service")
    
    def _load_model(self):
        """Lazy load the FinBERT model."""
        if not self._model_loaded:
            try:
                logger.info(f"Loading FinBERT model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                
                # Set model to evaluation mode
                self.model.eval()
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                    logger.info("FinBERT model loaded on GPU")
                else:
                    logger.info("FinBERT model loaded on CPU")
                
                self._model_loaded = True
            except Exception as e:
                logger.error(f"Failed to load FinBERT model: {e}")
                raise
    
    def _rate_limit(self):
        """Enforce rate limiting between API requests"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _is_cache_valid(self, cached_time: float) -> bool:
        """Check if cached data is still valid"""
        return time.time() - cached_time < self.cache_ttl
    
    def _retry_on_failure(self, func, *args, **kwargs):
        """
        Retry a function call with exponential backoff on failure.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self._max_retries):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                
                if attempt < self._max_retries - 1:
                    # Exponential backoff
                    delay = self._retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        
        # All retries failed
        logger.error(f"All retry attempts failed: {last_exception}")
        raise last_exception
    
    def get_news(
        self,
        symbol: str,
        days: int = 7,
        use_cache: bool = True
    ) -> List[NewsArticle]:
        """
        Fetch recent news articles for a stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days: Number of days to look back (default: 7)
            use_cache: Whether to use cached data if available
            
        Returns:
            List of NewsArticle objects
            
        Raises:
            ValueError: If symbol is invalid or days is out of range
            Exception: If API request fails
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if days < 1 or days > 30:
            raise ValueError("Days must be between 1 and 30")
        
        if not self.news_api_key:
            logger.warning("NewsAPI key not configured - returning empty news list")
            return []
        
        symbol = symbol.upper()
        
        # Create cache key
        cache_key = f"{symbol}_{days}"
        
        # Check cache
        if use_cache and cache_key in self._news_cache:
            cached_news, cached_time = self._news_cache[cache_key]
            if self._is_cache_valid(cached_time):
                logger.debug(f"Using cached news for {symbol}")
                return cached_news
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching news for {symbol} (last {days} days)")
        
        def fetch_news():
            # Search for news articles mentioning the symbol
            params = {
                'q': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(
                f"{self.news_api_base_url}/everything",
                params=params,
                timeout=10
            )
            
            if response.status_code == 401:
                raise ValueError("Invalid NewsAPI key")
            elif response.status_code == 429:
                raise Exception("NewsAPI rate limit exceeded")
            elif response.status_code != 200:
                raise Exception(f"NewsAPI request failed: {response.status_code} - {response.text}")
            
            data = response.json()
            
            if data.get('status') != 'ok':
                raise Exception(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            
            articles = []
            for article_data in data.get('articles', []):
                try:
                    # Parse published date
                    published_at = datetime.fromisoformat(
                        article_data['publishedAt'].replace('Z', '+00:00')
                    )
                    
                    article = NewsArticle(
                        title=article_data.get('title', ''),
                        description=article_data.get('description', ''),
                        url=article_data.get('url', ''),
                        source=article_data.get('source', {}).get('name', 'Unknown'),
                        published_at=published_at,
                        content=article_data.get('content')
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue
            
            return articles
        
        try:
            articles = self._retry_on_failure(fetch_news)
            
            # Update cache
            self._news_cache[cache_key] = (articles, time.time())
            
            logger.info(f"Fetched {len(articles)} news articles for {symbol}")
            return articles
        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            raise
    
    def analyze_sentiment(self, text: str, use_cache: bool = True) -> SentimentScore:
        """
        Analyze sentiment of text using FinBERT.
        
        Args:
            text: Text to analyze
            use_cache: Whether to use cached results if available
            
        Returns:
            SentimentScore object with sentiment analysis results
            
        Raises:
            ValueError: If text is empty
            Exception: If model inference fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Truncate very long text
        text = text[:512]
        
        # Check cache
        if use_cache and text in self._sentiment_cache:
            cached_sentiment, cached_time = self._sentiment_cache[text]
            if self._is_cache_valid(cached_time):
                logger.debug("Using cached sentiment score")
                return cached_sentiment
        
        # Load model if not already loaded
        self._load_model()
        
        logger.debug(f"Analyzing sentiment for text: {text[:100]}...")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get predictions (FinBERT outputs: negative, neutral, positive)
            predictions = predictions.cpu().numpy()[0]
            
            # Map to sentiment labels
            labels = ['negative', 'neutral', 'positive']
            label_scores = {label: float(score) for label, score in zip(labels, predictions)}
            
            # Get dominant sentiment
            max_idx = np.argmax(predictions)
            sentiment_label = labels[max_idx]
            confidence = float(predictions[max_idx])
            
            # Calculate normalized score (-1 to 1)
            # negative: -1, neutral: 0, positive: 1
            sentiment_score = (
                -1.0 * label_scores['negative'] +
                0.0 * label_scores['neutral'] +
                1.0 * label_scores['positive']
            )
            
            result = SentimentScore(
                score=sentiment_score,
                label=sentiment_label,
                confidence=confidence,
                raw_scores=label_scores
            )
            
            # Update cache
            self._sentiment_cache[text] = (result, time.time())
            
            logger.debug(
                f"Sentiment: {sentiment_label} "
                f"(score: {sentiment_score:.3f}, confidence: {confidence:.3f})"
            )
            
            return result
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            raise
    
    def get_aggregate_sentiment(
        self,
        symbol: str,
        days: int = 7,
        use_cache: bool = True
    ) -> float:
        """
        Get aggregate sentiment score for a stock based on recent news.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days: Number of days to look back (default: 7)
            use_cache: Whether to use cached data if available
            
        Returns:
            Aggregate sentiment score from -1.0 (negative) to 1.0 (positive)
            Returns 0.0 if no news articles are found
            
        Raises:
            ValueError: If symbol is invalid
            Exception: If news fetching or sentiment analysis fails
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.upper()
        
        logger.info(f"Calculating aggregate sentiment for {symbol}")
        
        try:
            # Fetch news articles
            articles = self.get_news(symbol, days=days, use_cache=use_cache)
            
            if not articles:
                logger.info(f"No news articles found for {symbol}")
                return 0.0
            
            # Analyze sentiment for each article
            sentiment_scores = []
            weights = []
            
            for article in articles:
                # Combine title and description for analysis
                text = f"{article.title}. {article.description or ''}"
                
                try:
                    sentiment = self.analyze_sentiment(text, use_cache=use_cache)
                    
                    # Store sentiment in article
                    article.sentiment_score = sentiment.score
                    
                    # Weight more recent articles higher
                    age_days = (datetime.now() - article.published_at.replace(tzinfo=None)).days
                    weight = 1.0 / (1.0 + age_days)  # Exponential decay
                    
                    sentiment_scores.append(sentiment.score)
                    weights.append(weight)
                except Exception as e:
                    logger.warning(f"Failed to analyze article sentiment: {e}")
                    continue
            
            if not sentiment_scores:
                logger.warning(f"No valid sentiment scores for {symbol}")
                return 0.0
            
            # Calculate weighted average
            weights_array = np.array(weights)
            scores_array = np.array(sentiment_scores)
            
            aggregate_score = float(
                np.sum(scores_array * weights_array) / np.sum(weights_array)
            )
            
            logger.info(
                f"Aggregate sentiment for {symbol}: {aggregate_score:.3f} "
                f"(based on {len(sentiment_scores)} articles)"
            )
            
            return aggregate_score
        except Exception as e:
            logger.error(f"Failed to calculate aggregate sentiment for {symbol}: {e}")
            raise
    
    def detect_breaking_news(
        self,
        symbol: str,
        hours: int = 24,
        sentiment_threshold: float = 0.5
    ) -> Optional[NewsAlert]:
        """
        Detect significant breaking news events for a stock.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            hours: Number of hours to look back (default: 24)
            sentiment_threshold: Minimum absolute sentiment score to trigger alert (default: 0.5)
            
        Returns:
            NewsAlert if significant news is detected, None otherwise
            
        Raises:
            ValueError: If symbol is invalid
            Exception: If news fetching or sentiment analysis fails
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if hours < 1 or hours > 168:  # Max 1 week
            raise ValueError("Hours must be between 1 and 168")
        
        symbol = symbol.upper()
        
        logger.info(f"Checking for breaking news for {symbol} (last {hours} hours)")
        
        try:
            # Fetch recent news (convert hours to days, minimum 1 day)
            days = max(1, hours // 24)
            articles = self.get_news(symbol, days=days, use_cache=False)
            
            if not articles:
                return None
            
            # Filter to articles within the specified hours
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_articles = [
                article for article in articles
                if article.published_at.replace(tzinfo=None) >= cutoff_time
            ]
            
            if not recent_articles:
                return None
            
            # Analyze sentiment and find most significant article
            max_impact = 0.0
            breaking_article = None
            breaking_sentiment = None
            
            for article in recent_articles:
                # Combine title and description
                text = f"{article.title}. {article.description or ''}"
                
                try:
                    sentiment = self.analyze_sentiment(text, use_cache=True)
                    
                    # Calculate impact score (absolute sentiment * confidence)
                    impact = abs(sentiment.score) * sentiment.confidence
                    
                    if impact > max_impact and abs(sentiment.score) >= sentiment_threshold:
                        max_impact = impact
                        breaking_article = article
                        breaking_sentiment = sentiment
                except Exception as e:
                    logger.warning(f"Failed to analyze article: {e}")
                    continue
            
            if breaking_article and breaking_sentiment:
                alert = NewsAlert(
                    symbol=symbol,
                    headline=breaking_article.title,
                    sentiment=breaking_sentiment.label,
                    impact_score=max_impact,
                    timestamp=breaking_article.published_at,
                    url=breaking_article.url
                )
                
                logger.info(
                    f"Breaking news detected for {symbol}: "
                    f"{alert.headline} (impact: {alert.impact_score:.3f})"
                )
                
                return alert
            
            logger.debug(f"No significant breaking news for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Failed to detect breaking news for {symbol}: {e}")
            raise
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            symbol: If provided, clear cache only for this symbol.
                   If None, clear all cache.
        """
        if symbol:
            symbol = symbol.upper()
            # Clear news cache
            keys_to_delete = [
                key for key in self._news_cache.keys()
                if key.startswith(symbol)
            ]
            for key in keys_to_delete:
                del self._news_cache[key]
            
            if keys_to_delete:
                logger.debug(f"Cleared {len(keys_to_delete)} news cache entries for {symbol}")
        else:
            # Clear all cache
            self._news_cache.clear()
            self._sentiment_cache.clear()
            logger.info("Cleared all cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'news_cache_size': len(self._news_cache),
            'sentiment_cache_size': len(self._sentiment_cache),
            'cache_ttl_seconds': self.cache_ttl,
            'model_loaded': self._model_loaded
        }
