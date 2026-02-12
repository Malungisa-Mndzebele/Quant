"""
Personalization service for learning user preferences and adapting recommendations.

This module tracks user decisions on AI recommendations and adjusts the
recommendation weights to better align with user preferences over time.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class UserDecision:
    """Record of user decision on a recommendation"""
    decision_id: int
    symbol: str
    recommendation_action: str  # 'buy', 'sell', 'hold'
    recommendation_confidence: float
    user_action: str  # 'accepted', 'rejected', 'modified', 'ignored'
    actual_trade_action: Optional[str]  # Actual action taken if different
    timestamp: datetime
    outcome: Optional[str] = None  # 'profit', 'loss', 'neutral' (filled later)
    outcome_value: Optional[float] = None  # P&L if trade was executed
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PersonalizationMetrics:
    """Metrics showing personalization effectiveness"""
    total_decisions: int
    acceptance_rate: float
    rejection_rate: float
    modification_rate: float
    avg_confidence_accepted: float
    avg_confidence_rejected: float
    personalization_active: bool
    personalization_confidence: float
    improvement_score: float  # How much better personalized vs default
    learned_preferences: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class PersonalizationService:
    """
    Service for learning user preferences and personalizing recommendations.
    
    Tracks user decisions on recommendations, learns patterns, and adjusts
    recommendation weights to better match user preferences.
    """
    
    def __init__(
        self,
        db_path: str = "data/database/personalization.db",
        min_decisions_for_learning: int = 10,
        learning_rate: float = 0.1
    ):
        """
        Initialize personalization service.
        
        Args:
            db_path: Path to SQLite database file
            min_decisions_for_learning: Minimum decisions before personalization activates
            learning_rate: Rate at which to adjust weights (0.0 to 1.0)
        """
        self.db_path = db_path
        self.min_decisions_for_learning = min_decisions_for_learning
        self.learning_rate = learning_rate
        
        # Ensure database directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Preference weights (learned from user behavior)
        self.preference_weights = {
            'confidence_threshold': 0.6,  # Minimum confidence user typically accepts
            'risk_preference': 0.5,  # 0=conservative, 1=aggressive
            'action_bias': {  # User's tendency to accept each action type
                'buy': 0.5,
                'sell': 0.5,
                'hold': 0.5
            },
            'indicator_weights': {  # Which indicators user values most
                'rsi': 1.0,
                'macd': 1.0,
                'moving_averages': 1.0,
                'volatility': 1.0
            }
        }
        
        # Load existing preferences
        self._load_preferences()
        
        logger.info(f"Initialized personalization service with database: {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create user_decisions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                recommendation_action TEXT NOT NULL,
                recommendation_confidence REAL NOT NULL,
                user_action TEXT NOT NULL,
                actual_trade_action TEXT,
                timestamp TEXT NOT NULL,
                outcome TEXT,
                outcome_value REAL,
                notes TEXT
            )
        """)
        
        # Create preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                preference_key TEXT PRIMARY KEY,
                preference_value TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        
        # Create personalization_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS personalization_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_decisions INTEGER NOT NULL,
                acceptance_rate REAL NOT NULL,
                confidence_threshold REAL NOT NULL,
                risk_preference REAL NOT NULL,
                improvement_score REAL NOT NULL
            )
        """)
        
        # Create indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_decisions_symbol 
            ON user_decisions(symbol)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_decisions_timestamp 
            ON user_decisions(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_decisions_user_action 
            ON user_decisions(user_action)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Personalization database initialized successfully")
    
    def record_decision(
        self,
        symbol: str,
        recommendation_action: str,
        recommendation_confidence: float,
        user_action: str,
        actual_trade_action: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Record a user decision on a recommendation.
        
        Args:
            symbol: Stock symbol
            recommendation_action: AI recommended action ('buy', 'sell', 'hold')
            recommendation_confidence: Confidence score (0.0 to 1.0)
            user_action: User's response ('accepted', 'rejected', 'modified', 'ignored')
            actual_trade_action: Actual action taken if different from recommendation
            notes: Optional notes about the decision
            
        Returns:
            Decision ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_decisions (
                symbol, recommendation_action, recommendation_confidence,
                user_action, actual_trade_action, timestamp, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            recommendation_action,
            recommendation_confidence,
            user_action,
            actual_trade_action,
            datetime.now().isoformat(),
            notes
        ))
        
        decision_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(
            f"Recorded decision {decision_id}: {symbol} - "
            f"{recommendation_action} -> {user_action}"
        )
        
        # Update preferences if we have enough data
        self._update_preferences()
        
        return decision_id
    
    def update_decision_outcome(
        self,
        decision_id: int,
        outcome: str,
        outcome_value: Optional[float] = None
    ):
        """
        Update the outcome of a decision after trade completion.
        
        Args:
            decision_id: ID of the decision to update
            outcome: Outcome ('profit', 'loss', 'neutral')
            outcome_value: P&L value if applicable
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE user_decisions
            SET outcome = ?, outcome_value = ?
            WHERE decision_id = ?
        """, (outcome, outcome_value, decision_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated decision {decision_id} outcome: {outcome}")
        
        # Re-learn preferences with updated outcomes
        self._update_preferences()
    
    def get_decisions(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_action: Optional[str] = None
    ) -> List[UserDecision]:
        """
        Retrieve user decisions with optional filtering.
        
        Args:
            symbol: Filter by symbol
            start_date: Filter by start date
            end_date: Filter by end date
            user_action: Filter by user action
            
        Returns:
            List of UserDecision objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM user_decisions WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if user_action:
            query += " AND user_action = ?"
            params.append(user_action)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        decisions = []
        for row in rows:
            decision = UserDecision(
                decision_id=row[0],
                symbol=row[1],
                recommendation_action=row[2],
                recommendation_confidence=row[3],
                user_action=row[4],
                actual_trade_action=row[5],
                timestamp=datetime.fromisoformat(row[6]),
                outcome=row[7],
                outcome_value=row[8],
                notes=row[9]
            )
            decisions.append(decision)
        
        return decisions
    
    def _update_preferences(self):
        """
        Update preference weights based on user decisions.
        
        This method analyzes user behavior patterns and adjusts weights
        to better match user preferences.
        """
        decisions = self.get_decisions()
        
        if len(decisions) < self.min_decisions_for_learning:
            logger.debug(
                f"Not enough decisions for learning "
                f"({len(decisions)}/{self.min_decisions_for_learning})"
            )
            return
        
        # Analyze acceptance patterns
        accepted = [d for d in decisions if d.user_action == 'accepted']
        rejected = [d for d in decisions if d.user_action == 'rejected']
        
        if not accepted:
            logger.debug("No accepted decisions to learn from")
            return
        
        # Learn confidence threshold
        accepted_confidences = [d.recommendation_confidence for d in accepted]
        rejected_confidences = [d.recommendation_confidence for d in rejected] if rejected else [0.0]
        
        avg_accepted_conf = np.mean(accepted_confidences)
        avg_rejected_conf = np.mean(rejected_confidences) if rejected_confidences else 0.0
        
        # Update confidence threshold (weighted average)
        new_threshold = avg_accepted_conf * 0.9  # Slightly below average accepted
        self.preference_weights['confidence_threshold'] = (
            (1 - self.learning_rate) * self.preference_weights['confidence_threshold'] +
            self.learning_rate * new_threshold
        )
        
        # Learn action bias
        action_counts = defaultdict(lambda: {'accepted': 0, 'total': 0})
        for d in decisions:
            action = d.recommendation_action
            action_counts[action]['total'] += 1
            if d.user_action == 'accepted':
                action_counts[action]['accepted'] += 1
        
        for action in ['buy', 'sell', 'hold']:
            if action_counts[action]['total'] > 0:
                acceptance_rate = action_counts[action]['accepted'] / action_counts[action]['total']
                # Update with learning rate
                self.preference_weights['action_bias'][action] = (
                    (1 - self.learning_rate) * self.preference_weights['action_bias'][action] +
                    self.learning_rate * acceptance_rate
                )
        
        # Learn risk preference from outcomes
        if any(d.outcome is not None for d in decisions):
            profitable_decisions = [d for d in decisions if d.outcome == 'profit']
            risky_decisions = [d for d in decisions if d.recommendation_confidence < 0.7]
            
            if risky_decisions:
                risky_profitable = [d for d in profitable_decisions if d.recommendation_confidence < 0.7]
                risk_success_rate = len(risky_profitable) / len(risky_decisions)
                
                # If user is successful with risky trades, increase risk preference
                if risk_success_rate > 0.6:
                    self.preference_weights['risk_preference'] = min(
                        1.0,
                        self.preference_weights['risk_preference'] + self.learning_rate * 0.1
                    )
                elif risk_success_rate < 0.4:
                    self.preference_weights['risk_preference'] = max(
                        0.0,
                        self.preference_weights['risk_preference'] - self.learning_rate * 0.1
                    )
        
        # Save updated preferences
        self._save_preferences()
        
        # Record history
        self._record_history()
        
        logger.info(
            f"Updated preferences: confidence_threshold={self.preference_weights['confidence_threshold']:.2f}, "
            f"risk_preference={self.preference_weights['risk_preference']:.2f}"
        )
    
    def _save_preferences(self):
        """Save current preferences to database"""
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for key, value in self.preference_weights.items():
            cursor.execute("""
                INSERT OR REPLACE INTO preferences (preference_key, preference_value, last_updated)
                VALUES (?, ?, ?)
            """, (key, json.dumps(value), timestamp))
        
        conn.commit()
        conn.close()
    
    def _load_preferences(self):
        """Load preferences from database"""
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT preference_key, preference_value FROM preferences")
        rows = cursor.fetchall()
        conn.close()
        
        for key, value_json in rows:
            try:
                self.preference_weights[key] = json.loads(value_json)
            except json.JSONDecodeError:
                logger.warning(f"Failed to load preference: {key}")
    
    def _record_history(self):
        """Record current personalization state to history"""
        decisions = self.get_decisions()
        
        if not decisions:
            return
        
        accepted = [d for d in decisions if d.user_action == 'accepted']
        acceptance_rate = len(accepted) / len(decisions) if decisions else 0.0
        
        # Calculate improvement score
        improvement_score = self._calculate_improvement_score()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO personalization_history (
                timestamp, total_decisions, acceptance_rate,
                confidence_threshold, risk_preference, improvement_score
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            len(decisions),
            acceptance_rate,
            self.preference_weights['confidence_threshold'],
            self.preference_weights['risk_preference'],
            improvement_score
        ))
        
        conn.commit()
        conn.close()
    
    def _calculate_improvement_score(self) -> float:
        """
        Calculate how much personalization has improved recommendations.
        
        Returns:
            Improvement score (0.0 to 1.0, where 1.0 is maximum improvement)
        """
        decisions = self.get_decisions()
        
        if len(decisions) < self.min_decisions_for_learning:
            return 0.0
        
        # Compare recent acceptance rate to early acceptance rate
        recent_decisions = decisions[:20]  # Last 20 decisions
        early_decisions = decisions[-20:] if len(decisions) >= 40 else decisions  # First 20
        
        recent_acceptance = sum(1 for d in recent_decisions if d.user_action == 'accepted') / len(recent_decisions)
        early_acceptance = sum(1 for d in early_decisions if d.user_action == 'accepted') / len(early_decisions)
        
        # Improvement is the increase in acceptance rate
        improvement = recent_acceptance - early_acceptance
        
        # Normalize to 0-1 range (assume max improvement is 0.3 or 30%)
        normalized_improvement = max(0.0, min(1.0, (improvement + 0.3) / 0.6))
        
        return normalized_improvement
    
    def get_personalization_metrics(self) -> PersonalizationMetrics:
        """
        Get metrics showing personalization effectiveness.
        
        Returns:
            PersonalizationMetrics object
        """
        decisions = self.get_decisions()
        
        if not decisions:
            return PersonalizationMetrics(
                total_decisions=0,
                acceptance_rate=0.0,
                rejection_rate=0.0,
                modification_rate=0.0,
                avg_confidence_accepted=0.0,
                avg_confidence_rejected=0.0,
                personalization_active=False,
                personalization_confidence=0.0,
                improvement_score=0.0,
                learned_preferences={}
            )
        
        # Calculate rates
        accepted = [d for d in decisions if d.user_action == 'accepted']
        rejected = [d for d in decisions if d.user_action == 'rejected']
        modified = [d for d in decisions if d.user_action == 'modified']
        
        acceptance_rate = len(accepted) / len(decisions)
        rejection_rate = len(rejected) / len(decisions)
        modification_rate = len(modified) / len(decisions)
        
        # Average confidences
        avg_confidence_accepted = np.mean([d.recommendation_confidence for d in accepted]) if accepted else 0.0
        avg_confidence_rejected = np.mean([d.recommendation_confidence for d in rejected]) if rejected else 0.0
        
        # Personalization status
        personalization_active = len(decisions) >= self.min_decisions_for_learning
        personalization_confidence = min(1.0, len(decisions) / (self.min_decisions_for_learning * 3))
        
        # Improvement score
        improvement_score = self._calculate_improvement_score()
        
        return PersonalizationMetrics(
            total_decisions=len(decisions),
            acceptance_rate=acceptance_rate,
            rejection_rate=rejection_rate,
            modification_rate=modification_rate,
            avg_confidence_accepted=float(avg_confidence_accepted),
            avg_confidence_rejected=float(avg_confidence_rejected),
            personalization_active=personalization_active,
            personalization_confidence=personalization_confidence,
            improvement_score=improvement_score,
            learned_preferences=self.preference_weights.copy()
        )
    
    def adjust_recommendation(
        self,
        recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adjust a recommendation based on learned preferences.
        
        Args:
            recommendation: Original recommendation dictionary
            
        Returns:
            Adjusted recommendation dictionary
        """
        decisions = self.get_decisions()
        
        # Don't adjust if not enough data
        if len(decisions) < self.min_decisions_for_learning:
            return recommendation
        
        adjusted = recommendation.copy()
        
        # Adjust confidence based on action bias
        action = recommendation.get('action', 'hold')
        action_bias = self.preference_weights['action_bias'].get(action, 0.5)
        
        # Increase confidence if user typically accepts this action
        if action_bias > 0.6:
            adjusted['confidence'] = min(1.0, recommendation['confidence'] * (1 + (action_bias - 0.5)))
        # Decrease confidence if user typically rejects this action
        elif action_bias < 0.4:
            adjusted['confidence'] = recommendation['confidence'] * action_bias * 2
        
        # Adjust risk level based on risk preference
        risk_levels = {'low': 0, 'medium': 1, 'high': 2}
        current_risk = risk_levels.get(recommendation.get('risk_level', 'medium'), 1)
        risk_pref = self.preference_weights['risk_preference']
        
        # If user prefers higher risk and current risk is low, suggest medium
        if risk_pref > 0.7 and current_risk == 0:
            adjusted['risk_level'] = 'medium'
        # If user prefers lower risk and current risk is high, suggest medium
        elif risk_pref < 0.3 and current_risk == 2:
            adjusted['risk_level'] = 'medium'
        
        # Add personalization indicator
        adjusted['personalized'] = True
        adjusted['personalization_confidence'] = min(1.0, len(decisions) / (self.min_decisions_for_learning * 3))
        
        return adjusted
    
    def reset_personalization(self):
        """
        Reset personalization to default state.
        
        Clears learned preferences but keeps decision history for reference.
        """
        # Reset to default weights
        self.preference_weights = {
            'confidence_threshold': 0.6,
            'risk_preference': 0.5,
            'action_bias': {
                'buy': 0.5,
                'sell': 0.5,
                'hold': 0.5
            },
            'indicator_weights': {
                'rsi': 1.0,
                'macd': 1.0,
                'moving_averages': 1.0,
                'volatility': 1.0
            }
        }
        
        # Save reset preferences
        self._save_preferences()
        
        logger.info("Personalization reset to default state")
    
    def get_personalization_history(
        self,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get personalization history over time.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with personalization metrics over time
        """
        conn = sqlite3.connect(self.db_path)
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = """
            SELECT timestamp, total_decisions, acceptance_rate,
                   confidence_threshold, risk_preference, improvement_score
            FROM personalization_history
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date,))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def export_decisions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Export user decisions to DataFrame.
        
        Args:
            start_date: Start date for export
            end_date: End date for export
            
        Returns:
            DataFrame with all decisions
        """
        decisions = self.get_decisions(start_date=start_date, end_date=end_date)
        
        if not decisions:
            return pd.DataFrame()
        
        data = [d.to_dict() for d in decisions]
        df = pd.DataFrame(data)
        
        return df


def create_personalization_service(
    db_path: str = "data/database/personalization.db"
) -> PersonalizationService:
    """
    Create personalization service with default configuration.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Configured PersonalizationService instance
    """
    return PersonalizationService(
        db_path=db_path,
        min_decisions_for_learning=10,
        learning_rate=0.1
    )
