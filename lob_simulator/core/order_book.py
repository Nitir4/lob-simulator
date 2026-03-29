"""
core/order_book.py  (PHASE 2 REVISION)
========================================
Changes in this revision:

1. add_limit_order() now returns Tuple[List[Trade], Optional[Order]].
   The Optional[Order] is the rested remainder (None if fully filled).
   This eliminates the O(n) set-difference in Simulator.

2. get_imbalance() — Cont-Larrard (2013) queue imbalance:
     I = (V_bid - V_ask) / (V_bid + V_ask)
   V_bid, V_ask are total volume at the BEST level only.
   Returns 0.0 if either side is empty.

3. get_best_level_volume(side) added as helper.

MATCHING ENGINE (formal):
  Price priority: SortedDict levels in best-first order.
  Time priority:  deque popleft() → FIFO within each level.
  Execution price: ALWAYS resting_order.price (passive pricing rule).
  Crossing: buy crosses if limit_price >= ask_price (or market).
            sell crosses if limit_price <= bid_price (or market).
  Partial fill: reduce quantities in-place;
                rest unfilled limit remainder; discard market remainder.

DATA STRUCTURE JUSTIFICATION:
  SortedDict preferred over heapq because cancellations require
  true O(log n) deletion. heapq's lazy deletion is O(n) amortized
  for get_best_bid() when stale entries accumulate at the top.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from sortedcontainers import SortedDict

from lob_simulator.core.order import Order
from lob_simulator.core.trade import Trade


class OrderBook:
    """
    A price-time priority limit order book.

    INVARIANTS (checked by assert_invariants()):
      1. best_ask > best_bid when both exist (no crossed book).
      2. Every price level has a non-empty deque (no ghost levels).
      3. Every order in the book is in _order_index, and vice versa.
      4. All resting order quantities are >= 1.

    Public API
    ----------
    add_limit_order(order)        -> Tuple[List[Trade], Optional[Order]]
    process_market_order(order)   -> List[Trade]
    cancel_order(order_id)        -> bool
    get_best_bid()                -> Optional[float]
    get_best_ask()                -> Optional[float]
    get_mid_price()               -> Optional[float]
    get_spread()                  -> Optional[float]
    get_imbalance()               -> float
    get_best_level_volume(side)   -> int
    snapshot()                    -> dict
    assert_invariants()           -> None
    """

    def __init__(self) -> None:
        # Ascending SortedDict: best_bid = keys()[-1], best_ask = keys()[0]
        self._bids: SortedDict = SortedDict()
        self._asks: SortedDict = SortedDict()
        # order_id → Order: O(1) lookup for cancel
        self._order_index: Dict[int, Order] = {}

    # ================================================================== #
    # Public interface                                                     #
    # ================================================================== #

    def add_limit_order(
        self, order: Order
    ) -> Tuple[List[Trade], Optional[Order]]:
        """
        Submit a limit order.

        Returns
        -------
        (trades, rested_order) where rested_order is the Order object
        as placed in the book (with reduced quantity after partial fills),
        or None if fully matched and not rested.

        WHY return the rested order?
          The Simulator needs to schedule a cancel_time for any newly
          rested order. Previously it computed a set-difference of
          _order_index before/after — O(n). Returning the rested order
          directly makes this O(1).
        """
        assert order.order_type == "limit", \
            "Use process_market_order() for market orders."
        trades = self._match(order)

        rested: Optional[Order] = None
        if order.quantity > 0:
            self._add_to_book(order)
            rested = order

        return trades, rested

    def process_market_order(self, order: Order) -> List[Trade]:
        """
        Submit a market order. Sweeps at best prices; remainder discarded.
        Market orders never rest in the book.
        """
        assert order.order_type == "market"
        return self._match(order)

    def cancel_order(self, order_id: int) -> bool:
        """
        Remove a resting limit order. Returns True if found, False otherwise.
        Complexity: O(log n) for level removal; O(k) for deque.remove().
        """
        if order_id not in self._order_index:
            return False

        order = self._order_index.pop(order_id)
        book = self._bids if order.side == "buy" else self._asks

        level_queue: deque = book[order.price]
        try:
            level_queue.remove(order)
        except ValueError:
            pass  # Already consumed by matching

        if not level_queue:
            del book[order.price]

        return True

    # ── Accessors ──────────────────────────────────────────────────────

    def get_best_bid(self) -> Optional[float]:
        return self._bids.keys()[-1] if self._bids else None

    def get_best_ask(self) -> Optional[float]:
        return self._asks.keys()[0] if self._asks else None

    def get_mid_price(self) -> Optional[float]:
        """Mid = (best_bid + best_ask) / 2. Standard neutral reference price."""
        bid, ask = self.get_best_bid(), self.get_best_ask()
        return (bid + ask) / 2.0 if (bid is not None and ask is not None) else None

    def get_spread(self) -> Optional[float]:
        bid, ask = self.get_best_bid(), self.get_best_ask()
        return (ask - bid) if (bid is not None and ask is not None) else None

    def get_best_level_volume(self, side: str) -> int:
        """
        Total quantity resting at the best price level on a given side.

        'side' = 'buy' → best_bid level; 'sell' → best_ask level.
        Returns 0 if the side is empty.
        """
        if side == "buy":
            if not self._bids:
                return 0
            best = self._bids.keys()[-1]
            return sum(o.quantity for o in self._bids[best])
        else:
            if not self._asks:
                return 0
            best = self._asks.keys()[0]
            return sum(o.quantity for o in self._asks[best])

    def get_imbalance(self) -> float:
        """
        Queue imbalance at the best level (Cont & Larrard 2013):

            I = (V_bid - V_ask) / (V_bid + V_ask)

        where V_bid = total volume at best bid level,
              V_ask = total volume at best ask level.

        I ∈ [-1, 1]:
          I > 0 → bid side heavier → upward price pressure.
          I < 0 → ask side heavier → downward price pressure.
          I = 0 → balanced, no directional signal.

        Returns 0.0 when either side is empty (no signal possible).

        WHY use best level only, not total depth?
          Empirical research (Stoikov 2009, Cont-Larrard 2013) shows
          that best-level imbalance is a strong short-horizon price
          predictor. Depth far from best quotes has much weaker signal.
          Using total depth conflates signal with noise from deep levels.
        """
        v_bid = self.get_best_level_volume("buy")
        v_ask = self.get_best_level_volume("sell")
        total = v_bid + v_ask
        if total == 0:
            return 0.0
        return (v_bid - v_ask) / total

    def snapshot(self) -> dict:
        """Human-readable state snapshot. Top 5 levels each side."""
        bids_view = [
            (px, sum(o.quantity for o in q))
            for px, q in reversed(self._bids.items())
        ]
        asks_view = [
            (px, sum(o.quantity for o in q))
            for px, q in self._asks.items()
        ]
        return {
            "best_bid":   self.get_best_bid(),
            "best_ask":   self.get_best_ask(),
            "mid_price":  self.get_mid_price(),
            "spread":     self.get_spread(),
            "imbalance":  self.get_imbalance(),
            "bids":       bids_view[:5],
            "asks":       asks_view[:5],
        }

    def assert_invariants(self) -> None:
        """
        Check all structural invariants. Raises AssertionError on violation.

        WHY each invariant matters:
          1. No crossed book: matching engine must fill all crossable orders.
             A crossed book means a trade was missed — incorrect simulation.
          2. No empty levels: get_best_bid() reads keys()[-1]. An empty
             deque at that key returns the wrong price silently.
          3. Index ↔ book bijection: ghost entries in _order_index cause
             cancel_order() to report True but do nothing; missing entries
             make orders uncancellable forever.
          4. Positive quantities: zero or negative means overfill bug.
        """
        bid, ask = self.get_best_bid(), self.get_best_ask()
        if bid is not None and ask is not None:
            assert ask > bid, f"Crossed book: bid={bid}, ask={ask}"

        book_ids = set()

        for px, q in self._bids.items():
            assert len(q) > 0, f"Empty bid level at {px}"
            for o in q:
                assert o.quantity >= 1, f"Non-positive qty {o.quantity} in bids"
                book_ids.add(o.order_id)

        for px, q in self._asks.items():
            assert len(q) > 0, f"Empty ask level at {px}"
            for o in q:
                assert o.quantity >= 1, f"Non-positive qty {o.quantity} in asks"
                book_ids.add(o.order_id)

        indexed_ids = set(self._order_index.keys())
        assert book_ids == indexed_ids, (
            f"Index mismatch: in_book_not_indexed={book_ids - indexed_ids}, "
            f"indexed_not_in_book={indexed_ids - book_ids}"
        )

    # ================================================================== #
    # Internal helpers                                                     #
    # ================================================================== #

    def _add_to_book(self, order: Order) -> None:
        book = self._bids if order.side == "buy" else self._asks
        if order.price not in book:
            book[order.price] = deque()
        book[order.price].append(order)
        self._order_index[order.order_id] = order

    def _match(self, aggressive: Order) -> List[Trade]:
        """
        Core matching engine. See module docstring for formal rules.
        """
        trades: List[Trade] = []

        if aggressive.side == "buy":
            opposite_book = self._asks
            price_levels  = list(opposite_book.keys())             # ascending
            def crosses(lim, rest): return lim is None or lim >= rest
        else:
            opposite_book = self._bids
            price_levels  = list(reversed(opposite_book.keys()))   # descending
            def crosses(lim, rest): return lim is None or lim <= rest

        for level_price in price_levels:
            if aggressive.quantity <= 0:
                break
            if not crosses(aggressive.price, level_price):
                break
            if level_price not in opposite_book:
                continue

            level_queue: deque = opposite_book[level_price]

            while level_queue and aggressive.quantity > 0:
                resting = level_queue[0]
                fill_qty = min(aggressive.quantity, resting.quantity)

                if aggressive.side == "buy":
                    buy_id, sell_id = aggressive.order_id, resting.order_id
                else:
                    buy_id, sell_id = resting.order_id, aggressive.order_id

                trade = Trade.create(
                    buy_order_id=buy_id,
                    sell_order_id=sell_id,
                    price=level_price,    # ALWAYS resting price
                    quantity=fill_qty,
                    timestamp=aggressive.timestamp,
                )
                trades.append(trade)

                aggressive.quantity -= fill_qty
                resting.quantity    -= fill_qty

                if resting.quantity == 0:
                    level_queue.popleft()
                    self._order_index.pop(resting.order_id, None)

            if not level_queue:
                del opposite_book[level_price]

        return trades

    def __repr__(self) -> str:
        return (
            f"OrderBook(bids={len(self._bids)} levels, "
            f"asks={len(self._asks)} levels, "
            f"mid={self.get_mid_price()}, I={self.get_imbalance():.2f})"
        )
