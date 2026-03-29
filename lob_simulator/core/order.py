"""
core/order.py
=============
Defines the fundamental building block of any LOB: the Order.

Design philosophy:
  - A dataclass keeps the definition concise and gives us __repr__ for free.
  - We use a simple string-based Side ('buy'/'sell') and OrderType ('limit'/'market')
    rather than integers, so the code reads like English.
  - `price` is Optional[float] because market orders have no fixed price — they
    are instructions to trade immediately at whatever the market offers.
  - `timestamp` is a float (seconds) matching the simulator's continuous time model.

WHY NOT inherit from dict or NamedTuple?
  - dict: no type safety, no attribute access syntax.
  - NamedTuple: immutable, cannot update `quantity` in-place during partial fills.
  - dataclass: mutable, type-annotated, clean repr — perfect fit.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import itertools

# Module-level counter so every Order ever created gets a unique ID, even
# across multiple OrderBook instances in one process.
_order_id_counter = itertools.count(1)


def next_order_id() -> int:
    """Return a monotonically-increasing unique order ID."""
    return next(_order_id_counter)


@dataclass
class Order:
    """
    Represents a single order in the limit order book.

    Attributes
    ----------
    order_id   : Unique integer identifier.
    side       : 'buy' or 'sell'.
    order_type : 'limit' or 'market'.
    price      : Limit price (None for market orders).
    quantity   : Remaining unfilled quantity (decremented on partial fills).
    timestamp  : Arrival time in simulation seconds.
    """

    order_id: int
    side: str              # 'buy' | 'sell'
    order_type: str        # 'limit' | 'market'
    price: Optional[float]
    quantity: int
    timestamp: float

    # ------------------------------------------------------------------ #
    # Convenience constructors                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def limit(
        cls,
        side: str,
        price: float,
        quantity: int,
        timestamp: float,
    ) -> "Order":
        """Create a limit order with an auto-assigned ID."""
        return cls(
            order_id=next_order_id(),
            side=side,
            order_type="limit",
            price=price,
            quantity=quantity,
            timestamp=timestamp,
        )

    @classmethod
    def market(
        cls,
        side: str,
        quantity: int,
        timestamp: float,
    ) -> "Order":
        """Create a market order with an auto-assigned ID."""
        return cls(
            order_id=next_order_id(),
            side=side,
            order_type="market",
            price=None,
            quantity=quantity,
            timestamp=timestamp,
        )

    def __repr__(self) -> str:
        price_str = f"{self.price:.4f}" if self.price is not None else "MKT"
        return (
            f"Order(id={self.order_id}, {self.side.upper()} {self.order_type.upper()}, "
            f"qty={self.quantity}, px={price_str}, t={self.timestamp:.4f})"
        )
