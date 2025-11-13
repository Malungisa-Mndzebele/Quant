"""Strategy module for trading strategies."""

from .base import Strategy, StrategyManager
from .moving_average_crossover import MovingAverageCrossover

__all__ = ['Strategy', 'StrategyManager', 'MovingAverageCrossover']
