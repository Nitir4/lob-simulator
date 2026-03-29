"""
simulation/events.py
====================
Defines the Event dataclass and EventType enum.

DESIGN PHILOSOPHY
-----------------
An Event represents an *external intention* — what a market participant
wants to do — before it is converted into an Order and processed by the
LOB.  Keeping this separation clean means:

  - The event generator only speaks the language of "what happens next"
  - The simulator translates events into Orders
  - The OrderBook only knows about Orders (no coupling to the simulator)

This is a classic layered design: each module has one clear responsibility.

EVENT TYPES
-----------
LIMIT_BUY  : Place a resting buy order at a specific price below the ask.
LIMIT_SELL : Place a resting sell order at a specific price above the bid.
MARKET_BUY : Buy at whatever the best ask is (immediate execution intent).
MARKET_SELL: Sell at whatever the best bid is (immediate execution intent).

WHY mix limit and market arrivals?
  Real markets have both patient traders (limit orders, providing liquidity)
  and impatient traders (market orders, consuming liquidity).  The ratio
  between them is a key market microstructure parameter that affects spread,
  depth, and volatility.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import itertools


class EventType(Enum):
    LIMIT_BUY   = auto()
    LIMIT_SELL  = auto()
    MARKET_BUY  = auto()
    MARKET_SELL = auto()
    CANCEL      = auto()   # resting order lifetime expiry


_event_id_counter = itertools.count(1)


@dataclass
class Event:
    """
    A single market event with an arrival time and intent.

    Attributes
    ----------
    event_id     : Unique identifier.
    event_type   : One of EventType.
    arrival_time : Simulation time at which this event occurs.
    price        : Intended limit price (None for market events).
    quantity     : Order size (number of shares / units).
    """

    event_id: int
    event_type: EventType
    arrival_time: float
    price: Optional[float]
    quantity: int

    @classmethod
    def create(
        cls,
        event_type: EventType,
        arrival_time: float,
        price: Optional[float],
        quantity: int,
    ) -> "Event":
        return cls(
            event_id=next(_event_id_counter),
            event_type=event_type,
            arrival_time=arrival_time,
            price=price,
            quantity=quantity,
        )

    def is_market(self) -> bool:
        return self.event_type in (EventType.MARKET_BUY, EventType.MARKET_SELL)

    def side(self) -> str:
        return "buy" if self.event_type in (EventType.LIMIT_BUY, EventType.MARKET_BUY) else "sell"

    def __repr__(self) -> str:
        px = f"{self.price:.4f}" if self.price is not None else "MKT"
        return (
            f"Event(id={self.event_id}, {self.event_type.name}, "
            f"qty={self.quantity}, px={px}, t={self.arrival_time:.4f})"
        )
