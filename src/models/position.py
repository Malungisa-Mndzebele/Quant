"""Position model."""

from dataclasses import dataclass


@dataclass
class Position:
    """Represents a stock position in the portfolio."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float = None

    def __post_init__(self):
        """Validate position and set defaults."""
        if self.current_price is None:
            self.current_price = self.entry_price
        
        if self.quantity == 0:
            raise ValueError("Position quantity cannot be zero")
        
        if self.entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {self.entry_price}")
        
        if self.current_price <= 0:
            raise ValueError(f"Current price must be positive, got {self.current_price}")

    def update_price(self, price: float) -> None:
        """Update the current price of the position."""
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        self.current_price = price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def market_value(self) -> float:
        """Calculate current market value of the position."""
        return self.current_price * abs(self.quantity)

    @property
    def cost_basis(self) -> float:
        """Calculate the cost basis of the position."""
        return self.entry_price * abs(self.quantity)

    @property
    def return_pct(self) -> float:
        """Calculate return percentage."""
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
