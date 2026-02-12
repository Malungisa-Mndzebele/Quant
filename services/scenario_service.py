"""
Scenario testing service for simulating different market conditions.

This module provides functionality to test portfolio performance under
various market scenarios (bull, bear, volatile, stable) by applying
historical patterns to current positions.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MarketScenario(str, Enum):
    """Types of market scenarios"""
    BULL = "bull"  # Strong upward trend
    BEAR = "bear"  # Strong downward trend
    VOLATILE = "volatile"  # High volatility, no clear trend
    STABLE = "stable"  # Low volatility, sideways movement
    CRASH = "crash"  # Sudden severe downturn
    RECOVERY = "recovery"  # Recovery from downturn


@dataclass
class Position:
    """Position information for scenario testing"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float


@dataclass
class ScenarioParameters:
    """Parameters defining a market scenario"""
    scenario_type: MarketScenario
    duration_days: int
    price_change_pct: float  # Expected price change over duration
    volatility_multiplier: float  # Multiplier for historical volatility
    correlation_factor: float  # How correlated assets move (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['scenario_type'] = self.scenario_type.value
        return data


@dataclass
class ScenarioResult:
    """Results from scenario simulation"""
    scenario_type: MarketScenario
    scenario_name: str
    duration_days: int
    initial_portfolio_value: float
    projected_portfolio_value: float
    projected_return: float
    projected_return_pct: float
    projected_max_drawdown: float
    projected_max_drawdown_pct: float
    projected_volatility: float
    value_at_risk_95: float  # 95% VaR
    probability_of_loss: float
    position_impacts: List[Dict[str, Any]]
    risk_score: float  # 0-100, higher is riskier
    suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['scenario_type'] = self.scenario_type.value
        return data


class ScenarioService:
    """
    Service for scenario testing and stress testing portfolios.
    
    Simulates different market conditions and projects portfolio performance
    under various scenarios.
    """
    
    # Predefined scenario parameters
    SCENARIO_PRESETS = {
        MarketScenario.BULL: ScenarioParameters(
            scenario_type=MarketScenario.BULL,
            duration_days=90,
            price_change_pct=15.0,
            volatility_multiplier=0.8,
            correlation_factor=0.7
        ),
        MarketScenario.BEAR: ScenarioParameters(
            scenario_type=MarketScenario.BEAR,
            duration_days=90,
            price_change_pct=-20.0,
            volatility_multiplier=1.5,
            correlation_factor=0.85
        ),
        MarketScenario.VOLATILE: ScenarioParameters(
            scenario_type=MarketScenario.VOLATILE,
            duration_days=60,
            price_change_pct=0.0,
            volatility_multiplier=2.5,
            correlation_factor=0.5
        ),
        MarketScenario.STABLE: ScenarioParameters(
            scenario_type=MarketScenario.STABLE,
            duration_days=90,
            price_change_pct=2.0,
            volatility_multiplier=0.5,
            correlation_factor=0.3
        ),
        MarketScenario.CRASH: ScenarioParameters(
            scenario_type=MarketScenario.CRASH,
            duration_days=30,
            price_change_pct=-35.0,
            volatility_multiplier=3.0,
            correlation_factor=0.95
        ),
        MarketScenario.RECOVERY: ScenarioParameters(
            scenario_type=MarketScenario.RECOVERY,
            duration_days=120,
            price_change_pct=25.0,
            volatility_multiplier=1.2,
            correlation_factor=0.6
        )
    }
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize scenario service.
        
        Args:
            random_seed: Random seed for reproducibility (optional)
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.info("Initialized scenario service")
    
    def run_scenario(
        self,
        scenario_type: MarketScenario,
        positions: List[Position],
        cash: float,
        historical_volatility: Optional[Dict[str, float]] = None,
        custom_parameters: Optional[ScenarioParameters] = None
    ) -> ScenarioResult:
        """
        Run scenario simulation on current portfolio.
        
        Args:
            scenario_type: Type of market scenario to simulate
            positions: Current portfolio positions
            cash: Available cash
            historical_volatility: Historical volatility for each symbol (optional)
            custom_parameters: Custom scenario parameters (optional)
            
        Returns:
            ScenarioResult with projected performance
            
        Raises:
            ValueError: If scenario type is invalid
        """
        if scenario_type not in self.SCENARIO_PRESETS and not custom_parameters:
            raise ValueError(f"Invalid scenario type: {scenario_type}")
        
        # Get scenario parameters
        params = custom_parameters or self.SCENARIO_PRESETS[scenario_type]
        
        logger.info(
            f"Running {scenario_type.value} scenario: "
            f"{params.price_change_pct:+.1f}% over {params.duration_days} days"
        )
        
        # Calculate initial portfolio value
        initial_value = cash + sum(pos.market_value for pos in positions)
        
        # Use default volatility if not provided
        if historical_volatility is None:
            historical_volatility = {pos.symbol: 0.25 for pos in positions}  # 25% annual vol
        
        # Simulate price paths for each position
        position_impacts = []
        projected_values = []
        
        for pos in positions:
            # Get historical volatility for this symbol
            vol = historical_volatility.get(pos.symbol, 0.25)
            
            # Simulate price path
            simulated_prices = self._simulate_price_path(
                initial_price=pos.current_price,
                expected_return_pct=params.price_change_pct,
                volatility=vol * params.volatility_multiplier,
                duration_days=params.duration_days,
                num_simulations=1000
            )
            
            # Calculate statistics
            final_prices = simulated_prices[:, -1]
            mean_final_price = np.mean(final_prices)
            std_final_price = np.std(final_prices)
            percentile_5 = np.percentile(final_prices, 5)
            percentile_95 = np.percentile(final_prices, 95)
            
            # Calculate projected P&L
            mean_pl = (mean_final_price - pos.current_price) * pos.quantity
            mean_pl_pct = ((mean_final_price - pos.current_price) / pos.current_price) * 100
            
            # Calculate worst case (5th percentile)
            worst_case_pl = (percentile_5 - pos.current_price) * pos.quantity
            worst_case_pl_pct = ((percentile_5 - pos.current_price) / pos.current_price) * 100
            
            # Calculate best case (95th percentile)
            best_case_pl = (percentile_95 - pos.current_price) * pos.quantity
            best_case_pl_pct = ((percentile_95 - pos.current_price) / pos.current_price) * 100
            
            # Probability of loss
            prob_loss = np.mean(final_prices < pos.current_price)
            
            position_impact = {
                'symbol': pos.symbol,
                'current_price': pos.current_price,
                'quantity': pos.quantity,
                'current_value': pos.market_value,
                'projected_price': mean_final_price,
                'projected_value': mean_final_price * pos.quantity,
                'projected_pl': mean_pl,
                'projected_pl_pct': mean_pl_pct,
                'worst_case_price': percentile_5,
                'worst_case_pl': worst_case_pl,
                'worst_case_pl_pct': worst_case_pl_pct,
                'best_case_price': percentile_95,
                'best_case_pl': best_case_pl,
                'best_case_pl_pct': best_case_pl_pct,
                'probability_of_loss': prob_loss,
                'price_std': std_final_price
            }
            
            position_impacts.append(position_impact)
            projected_values.append(mean_final_price * pos.quantity)
        
        # Calculate portfolio-level metrics
        projected_portfolio_value = cash + sum(projected_values)
        projected_return = projected_portfolio_value - initial_value
        projected_return_pct = (projected_return / initial_value) * 100 if initial_value > 0 else 0
        
        # Calculate maximum drawdown (using Monte Carlo simulations)
        max_drawdown, max_drawdown_pct = self._calculate_projected_drawdown(
            positions=positions,
            params=params,
            historical_volatility=historical_volatility
        )
        
        # Calculate portfolio volatility
        portfolio_volatility = self._calculate_portfolio_volatility(
            positions=positions,
            historical_volatility=historical_volatility,
            volatility_multiplier=params.volatility_multiplier
        )
        
        # Calculate Value at Risk (95% confidence)
        var_95 = self._calculate_value_at_risk(
            initial_value=initial_value,
            projected_return_pct=projected_return_pct,
            volatility=portfolio_volatility,
            confidence=0.95
        )
        
        # Calculate probability of loss
        prob_loss = self._calculate_probability_of_loss(
            projected_return_pct=projected_return_pct,
            volatility=portfolio_volatility
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            projected_return_pct=projected_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            volatility=portfolio_volatility,
            var_95=var_95,
            initial_value=initial_value
        )
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            scenario_type=scenario_type,
            position_impacts=position_impacts,
            risk_score=risk_score,
            projected_return_pct=projected_return_pct,
            max_drawdown_pct=max_drawdown_pct
        )
        
        result = ScenarioResult(
            scenario_type=scenario_type,
            scenario_name=self._get_scenario_name(scenario_type),
            duration_days=params.duration_days,
            initial_portfolio_value=initial_value,
            projected_portfolio_value=projected_portfolio_value,
            projected_return=projected_return,
            projected_return_pct=projected_return_pct,
            projected_max_drawdown=max_drawdown,
            projected_max_drawdown_pct=max_drawdown_pct,
            projected_volatility=portfolio_volatility,
            value_at_risk_95=var_95,
            probability_of_loss=prob_loss,
            position_impacts=position_impacts,
            risk_score=risk_score,
            suggestions=suggestions
        )
        
        logger.info(
            f"Scenario complete: {scenario_type.value} - "
            f"Projected return: {projected_return_pct:+.2f}%, "
            f"Risk score: {risk_score:.1f}"
        )
        
        return result
    
    def _simulate_price_path(
        self,
        initial_price: float,
        expected_return_pct: float,
        volatility: float,
        duration_days: int,
        num_simulations: int = 1000
    ) -> np.ndarray:
        """
        Simulate price paths using geometric Brownian motion.
        
        Args:
            initial_price: Starting price
            expected_return_pct: Expected return over duration (percentage)
            volatility: Annual volatility
            duration_days: Duration in days
            num_simulations: Number of simulation paths
            
        Returns:
            Array of shape (num_simulations, duration_days) with simulated prices
        """
        # Convert to daily parameters
        dt = 1 / 252  # Daily time step (252 trading days per year)
        drift = (expected_return_pct / 100) / duration_days  # Daily drift
        daily_vol = volatility * np.sqrt(dt)
        
        # Generate random returns
        random_returns = np.random.normal(
            loc=drift,
            scale=daily_vol,
            size=(num_simulations, duration_days)
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + random_returns, axis=1)
        
        # Calculate prices
        prices = initial_price * cumulative_returns
        
        return prices
    
    def _calculate_projected_drawdown(
        self,
        positions: List[Position],
        params: ScenarioParameters,
        historical_volatility: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Calculate projected maximum drawdown using Monte Carlo simulation.
        
        Args:
            positions: Current positions
            params: Scenario parameters
            historical_volatility: Historical volatility for each symbol
            
        Returns:
            Tuple of (max_drawdown_dollars, max_drawdown_percent)
        """
        if not positions:
            return 0.0, 0.0
        
        # Simulate portfolio value over time
        num_simulations = 1000
        portfolio_paths = np.zeros((num_simulations, params.duration_days))
        
        for pos in positions:
            vol = historical_volatility.get(pos.symbol, 0.25)
            price_paths = self._simulate_price_path(
                initial_price=pos.current_price,
                expected_return_pct=params.price_change_pct,
                volatility=vol * params.volatility_multiplier,
                duration_days=params.duration_days,
                num_simulations=num_simulations
            )
            
            # Add position value to portfolio
            portfolio_paths += price_paths * pos.quantity
        
        # Calculate drawdown for each simulation
        max_drawdowns = []
        for i in range(num_simulations):
            path = portfolio_paths[i, :]
            running_max = np.maximum.accumulate(path)
            drawdown = path - running_max
            max_drawdowns.append(abs(np.min(drawdown)))
        
        # Use median drawdown as projection
        projected_max_dd = np.median(max_drawdowns)
        
        # Calculate percentage
        initial_value = sum(pos.market_value for pos in positions)
        projected_max_dd_pct = (projected_max_dd / initial_value) * 100 if initial_value > 0 else 0
        
        return projected_max_dd, projected_max_dd_pct
    
    def _calculate_portfolio_volatility(
        self,
        positions: List[Position],
        historical_volatility: Dict[str, float],
        volatility_multiplier: float
    ) -> float:
        """
        Calculate projected portfolio volatility.
        
        Args:
            positions: Current positions
            historical_volatility: Historical volatility for each symbol
            volatility_multiplier: Scenario volatility multiplier
            
        Returns:
            Projected annual portfolio volatility
        """
        if not positions:
            return 0.0
        
        # Calculate portfolio value
        total_value = sum(pos.market_value for pos in positions)
        
        if total_value == 0:
            return 0.0
        
        # Calculate weighted average volatility (simplified - assumes no correlation)
        weighted_vol_squared = 0.0
        
        for pos in positions:
            weight = pos.market_value / total_value
            vol = historical_volatility.get(pos.symbol, 0.25) * volatility_multiplier
            weighted_vol_squared += (weight * vol) ** 2
        
        portfolio_vol = np.sqrt(weighted_vol_squared)
        
        return portfolio_vol
    
    def _calculate_value_at_risk(
        self,
        initial_value: float,
        projected_return_pct: float,
        volatility: float,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            initial_value: Initial portfolio value
            projected_return_pct: Projected return percentage
            volatility: Portfolio volatility
            confidence: Confidence level (default: 0.95)
            
        Returns:
            VaR in dollars (positive number represents potential loss)
        """
        # Convert to daily parameters
        daily_return = projected_return_pct / 100 / 252
        daily_vol = volatility / np.sqrt(252)
        
        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        
        # Calculate VaR
        var_return = daily_return + z_score * daily_vol
        var_dollars = abs(initial_value * var_return)
        
        return var_dollars
    
    def _calculate_probability_of_loss(
        self,
        projected_return_pct: float,
        volatility: float
    ) -> float:
        """
        Calculate probability of portfolio loss.
        
        Args:
            projected_return_pct: Projected return percentage
            volatility: Portfolio volatility
            
        Returns:
            Probability of loss (0-1)
        """
        from scipy import stats
        
        # Calculate z-score for break-even
        z_score = -projected_return_pct / (volatility * 100)
        
        # Calculate probability
        prob_loss = stats.norm.cdf(z_score)
        
        return prob_loss
    
    def _calculate_risk_score(
        self,
        projected_return_pct: float,
        max_drawdown_pct: float,
        volatility: float,
        var_95: float,
        initial_value: float
    ) -> float:
        """
        Calculate overall risk score (0-100).
        
        Higher score indicates higher risk.
        
        Args:
            projected_return_pct: Projected return percentage
            max_drawdown_pct: Maximum drawdown percentage
            volatility: Portfolio volatility
            var_95: Value at Risk (95%)
            initial_value: Initial portfolio value
            
        Returns:
            Risk score (0-100)
        """
        score = 0.0
        
        # Factor 1: Drawdown risk (0-35 points)
        # Higher drawdown = higher risk
        drawdown_score = min(max_drawdown_pct / 50 * 35, 35)  # 50% DD = max points
        score += drawdown_score
        
        # Factor 2: Volatility risk (0-25 points)
        # Higher volatility = higher risk
        vol_score = min(volatility / 0.5 * 25, 25)  # 50% vol = max points
        score += vol_score
        
        # Factor 3: VaR risk (0-25 points)
        # Higher VaR relative to portfolio = higher risk
        var_pct = (var_95 / initial_value) * 100 if initial_value > 0 else 0
        var_score = min(var_pct / 20 * 25, 25)  # 20% VaR = max points
        score += var_score
        
        # Factor 4: Return/risk ratio (0-15 points)
        # Negative or low return relative to risk = higher risk
        if volatility > 0:
            sharpe_like = projected_return_pct / (volatility * 100)
            if sharpe_like < 0:
                score += 15  # Negative expected return = max points
            elif sharpe_like < 0.5:
                score += 15 * (1 - sharpe_like / 0.5)
        
        return min(score, 100.0)
    
    def _generate_suggestions(
        self,
        scenario_type: MarketScenario,
        position_impacts: List[Dict[str, Any]],
        risk_score: float,
        projected_return_pct: float,
        max_drawdown_pct: float
    ) -> List[str]:
        """
        Generate portfolio adjustment suggestions based on scenario results.
        
        Args:
            scenario_type: Type of scenario
            position_impacts: Impact on each position
            risk_score: Overall risk score
            projected_return_pct: Projected return percentage
            max_drawdown_pct: Maximum drawdown percentage
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # High risk suggestions
        if risk_score >= 70:
            suggestions.append(
                f"⚠️ High risk detected (score: {risk_score:.1f}/100). "
                "Consider reducing position sizes or adding hedges."
            )
        
        # Drawdown suggestions
        if max_drawdown_pct > 25:
            suggestions.append(
                f"📉 Projected max drawdown of {max_drawdown_pct:.1f}% is significant. "
                "Consider setting tighter stop-losses or reducing exposure."
            )
        
        # Position-specific suggestions
        high_risk_positions = [
            p for p in position_impacts
            if p['probability_of_loss'] > 0.6 or p['worst_case_pl_pct'] < -30
        ]
        
        if high_risk_positions:
            symbols = [p['symbol'] for p in high_risk_positions[:3]]
            suggestions.append(
                f"🎯 High-risk positions in {scenario_type.value} scenario: {', '.join(symbols)}. "
                "Consider reducing exposure or hedging."
            )
        
        # Scenario-specific suggestions
        if scenario_type == MarketScenario.BEAR:
            if projected_return_pct < -15:
                suggestions.append(
                    "🐻 Portfolio is vulnerable to bear market. "
                    "Consider defensive positions, cash allocation, or inverse ETFs."
                )
        
        elif scenario_type == MarketScenario.VOLATILE:
            suggestions.append(
                "📊 High volatility scenario. Consider options strategies "
                "(straddles, strangles) to profit from volatility."
            )
        
        elif scenario_type == MarketScenario.CRASH:
            if projected_return_pct < -25:
                suggestions.append(
                    "💥 Portfolio highly vulnerable to market crash. "
                    "Strongly consider protective puts or reducing leverage."
                )
        
        elif scenario_type == MarketScenario.BULL:
            if projected_return_pct < 10:
                suggestions.append(
                    "🐂 Portfolio may underperform in bull market. "
                    "Consider increasing exposure to growth stocks or momentum strategies."
                )
        
        # Diversification suggestions
        if len(position_impacts) < 3:
            suggestions.append(
                f"🔀 Limited diversification ({len(position_impacts)} positions). "
                "Consider adding more positions to reduce concentration risk."
            )
        
        # Positive feedback
        if risk_score < 40 and projected_return_pct > 5:
            suggestions.append(
                f"✅ Portfolio shows resilience in {scenario_type.value} scenario "
                f"with acceptable risk (score: {risk_score:.1f}/100)."
            )
        
        return suggestions
    
    def _get_scenario_name(self, scenario_type: MarketScenario) -> str:
        """Get human-readable scenario name"""
        names = {
            MarketScenario.BULL: "Bull Market (Strong Uptrend)",
            MarketScenario.BEAR: "Bear Market (Strong Downtrend)",
            MarketScenario.VOLATILE: "High Volatility (Choppy Market)",
            MarketScenario.STABLE: "Stable Market (Low Volatility)",
            MarketScenario.CRASH: "Market Crash (Severe Downturn)",
            MarketScenario.RECOVERY: "Market Recovery (Post-Crash Rally)"
        }
        return names.get(scenario_type, scenario_type.value)
    
    def compare_scenarios(
        self,
        positions: List[Position],
        cash: float,
        scenarios: Optional[List[MarketScenario]] = None,
        historical_volatility: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compare portfolio performance across multiple scenarios.
        
        Args:
            positions: Current portfolio positions
            cash: Available cash
            scenarios: List of scenarios to compare (defaults to all presets)
            historical_volatility: Historical volatility for each symbol (optional)
            
        Returns:
            DataFrame with comparison metrics
        """
        if scenarios is None:
            scenarios = list(self.SCENARIO_PRESETS.keys())
        
        results = []
        
        for scenario_type in scenarios:
            result = self.run_scenario(
                scenario_type=scenario_type,
                positions=positions,
                cash=cash,
                historical_volatility=historical_volatility
            )
            
            results.append({
                'Scenario': result.scenario_name,
                'Duration (Days)': result.duration_days,
                'Projected Return': f"${result.projected_return:,.2f}",
                'Return %': f"{result.projected_return_pct:+.2f}%",
                'Max Drawdown %': f"{result.projected_max_drawdown_pct:.2f}%",
                'Volatility': f"{result.projected_volatility:.1%}",
                'VaR (95%)': f"${result.value_at_risk_95:,.2f}",
                'Prob. of Loss': f"{result.probability_of_loss:.1%}",
                'Risk Score': f"{result.risk_score:.1f}/100"
            })
        
        logger.info(f"Compared {len(scenarios)} scenarios")
        
        return pd.DataFrame(results)
    
    def get_scenario_parameters(self, scenario_type: MarketScenario) -> ScenarioParameters:
        """
        Get parameters for a scenario type.
        
        Args:
            scenario_type: Type of scenario
            
        Returns:
            ScenarioParameters
            
        Raises:
            ValueError: If scenario type is invalid
        """
        if scenario_type not in self.SCENARIO_PRESETS:
            raise ValueError(f"Invalid scenario type: {scenario_type}")
        
        return self.SCENARIO_PRESETS[scenario_type]
