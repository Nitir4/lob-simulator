"""
core/trade.py
=============
Records an execution event — two orders crossing in the book.

Design philosophy:
  - A Trade is an immutable record of something that *happened*.  We model it
    as a frozen dataclass so it cannot be accidentally mutated after creation.
  - Capturing both sides (buy_order_id, sell_order_id) lets us audit the full
    execution history: who traded with whom, at what time, at what price.
  - `price` is always the *resting* order's limit price (price improvement
    flows to the aggressive side — standard exchange behaviour).

WHY a separate Trade class?
  After a fill, the Order object's quantity changes.  If we only stored the
  modified Order we would lose the history of each individual execution.
  Trade gives us a clean, append-only audit log.
"""

from __future__ import annotations
from dataclasses import dataclass
import itertools

_trade_id_counter = itertools.count(1)


def next_trade_id() -> int:
    return next(_trade_id_counter)


@dataclass(frozen=True)
class Trade:
    """
    Immutable record of a single matched execution.

    Attributes
    ----------
    trade_id     : Unique identifier.
    buy_order_id : ID of the buy-side order involved.
    sell_order_id: ID of the sell-side order involved.
    price        : Execution price (resting order's limit price).
    quantity     : Number of units exchanged.
    timestamp    : Simulation time of execution.
    """

    trade_id: int
    buy_order_id: int
    sell_order_id: int
    price: float
    quantity: int
    timestamp: float

    @classmethod
    def create(
        cls,
        buy_order_id: int,
        sell_order_id: int,
        price: float,
        quantity: int,
        timestamp: float,
    ) -> "Trade":
        return cls(
            trade_id=next_trade_id(),
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
        )

    def __repr__(self) -> str:
        return (
            f"Trade(id={self.trade_id}, buy={self.buy_order_id}, "
            f"sell={self.sell_order_id}, qty={self.quantity}, "
            f"px={self.price:.4f}, t={self.timestamp:.4f})"
        )
