"""
simulation/simulator.py  (PHASE 2-5 REVISION)
===============================================
Changes in this revision:

1. O(1) new-order tracking  [Phase 2, Flaw 5 fix]
   BEFORE: set(order_book._order_index.keys()) before/after → O(n) per event.
   AFTER:  add_limit_order() returns (trades, rested_order).
           We use the returned Optional[Order] directly to schedule cancel.
           No set difference, no O(n) copy.

2. State passed to EventGenerator via update_state()  [Phase 3]
   After each event: pass mid-price AND imbalance to generator.
   Generator then uses imbalance to modulate market order probabilities.

3. Volume and imbalance logged in snapshots  [Phase 4]
   Each snapshot now includes:
     imbalance    : I ∈ [-1, 1]
     bid_volume   : Total volume at best bid level
     ask_volume   : Total volume at best ask level
     n_resting    : Total number of resting orders (stationarity monitor)
   This gives the user data to verify stationarity and study book dynamics.

4. Stationarity monitor in summary()  [Phase 1, Flaw 6 fix]
   Reports peak_resting and mean_resting across the simulation.
   If peak > 1000, prints a warning (book may be growing unboundedly).

5. Backward compatibility: Simulator still accepts SimParams; warmup
   uses exponential placement which may cross → handled by add_limit_order.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from lob_simulator.core.order import Order
from lob_simulator.core.order_book import OrderBook
from lob_simulator.core.trade import Trade
from lob_simulator.simulation.events import Event, EventType
from lob_simulator.simulation.event_generator import EventGenerator, SimParams


# ── Simulation History ────────────────────────────────────────────────────

@dataclass
class SimulationHistory:
    """
    Structured result of a simulation run.

    snapshots : One dict per event (arrival or cancellation).
    trades    : All Trade executions.
    params    : SimParams used.
    duration  : Requested horizon T.
    """

    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    trades:    List[Trade]          = field(default_factory=list)
    params:    Optional[SimParams]  = None
    duration:  float                = 0.0

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pip install pandas")
        return pd.DataFrame(self.snapshots)

    def summary(self) -> str:
        n_cancel = sum(1 for s in self.snapshots if s["event_type"] == "CANCEL")
        n_arrive = len(self.snapshots) - n_cancel
        mids = [s["mid_price"] for s in self.snapshots if s["mid_price"] is not None]
        resting_counts = [s["n_resting"] for s in self.snapshots]

        # Stationarity check
        peak_resting = max(resting_counts) if resting_counts else 0
        mean_resting = (sum(resting_counts) / len(resting_counts)
                        if resting_counts else 0)

        lines = [
            "── Simulation Summary ──────────────────────────",
            f"  Duration      : {self.duration:.2f} time units",
            f"  Arrivals      : {n_arrive}",
            f"  Cancellations : {n_cancel}",
            f"  Trades        : {len(self.trades)}",
            f"  Mid (start)   : {mids[0]:.4f}" if mids else "  Mid : N/A",
            f"  Mid (end)     : {mids[-1]:.4f}" if mids else "",
            f"  Mid drift     : {mids[-1]-mids[0]:+.4f}" if len(mids) > 1 else "",
            f"  Peak resting  : {peak_resting}",
            f"  Mean resting  : {mean_resting:.1f}",
        ]
        if peak_resting > 1000:
            lines.append("  ⚠ WARNING: peak_resting > 1000 — book may not be stationary.")
            lines.append("    Increase mu_cancel or reduce arrival_rate.")
        lines.append("────────────────────────────────────────────────")
        return "\n".join(lines)


# ── Simulator ─────────────────────────────────────────────────────────────

class Simulator:
    """
    Event-driven stochastic LOB simulator.

    SCOPE (honest stylized facts captured):
      + Positive bid-ask spread with genuine tightening/widening dynamics.
      + Price-time priority (maker pricing rule).
      + Inside-spread limit orders from mid-anchored Exp placement.
      + State-dependent market order rates (queue imbalance → directional drift).
      + Poisson arrivals + Exponential order lifetimes (cancellations).
      + Right-skewed order sizes (Geometric).

    NOT modelled:
      - Hawkes clustering (self-exciting arrivals after trades).
      - Intraday volume/volatility patterns (U-shape).
      - Long-memory order sign autocorrelation (Lillo-Farmer 2004).
      - Fat-tailed returns (power law).
      - Strategic/informed heterogeneous agents.

    Parameters
    ----------
    params : SimParams — all model parameters.
    seed   : Random seed for reproducibility.
    """

    def __init__(
        self,
        params: Optional[SimParams] = None,
        seed:   Optional[int]       = None,
    ) -> None:
        self.params = params or SimParams()
        self.seed   = seed
        self._reset()

    def _reset(self) -> None:
        self.order_book      = OrderBook()
        self.event_generator = EventGenerator(params=self.params, seed=self.seed)
        self.history         = SimulationHistory(params=self.params)
        self._cancel_heap: List[tuple] = []
        self._pending_cancels: Dict[int, float] = {}

    def run(self, T: float) -> SimulationHistory:
        """
        Run the event-driven simulation from t=0 to t=T.

        Algorithm
        ---------
        1. Warmup: seed book with initial resting orders.
        2. Loop:
           a. Draw next arrival via Exp(arrival_rate).
           b. Process all pending cancellations before next arrival.
           c. Process arrival: submit order to LOB.
           d. If a new order rested: schedule its cancel_time (O(1)).
           e. Update generator state: mid + imbalance.
           f. Log snapshot with imbalance, volumes, n_resting.
        """
        self._reset()
        self.history.duration = T

        self._warmup_book()

        t = 0.0

        while True:
            # 2a. Draw next arrival
            event = self.event_generator.next_event(current_time=t)
            if event.arrival_time > T:
                self._process_cancellations_before(T)
                break

            # 2b. Process cancellations due before this arrival
            self._process_cancellations_before(event.arrival_time)

            t = event.arrival_time

            # 2c. Submit order to LOB
            order = self._event_to_order(event)

            if order.order_type == "limit":
                trades, rested = self.order_book.add_limit_order(order)
            else:
                trades = self.order_book.process_market_order(order)
                rested = None   # market orders never rest

            # 2d. Schedule cancel for newly rested order — O(1)
            if rested is not None:
                cancel_time = self.event_generator.generate_cancel_time(t)
                self._schedule_cancel(rested.order_id, cancel_time)

            # 2e. Update event generator state
            mid       = self.order_book.get_mid_price()
            imbalance = self.order_book.get_imbalance()
            self.event_generator.update_state(mid=mid, imbalance=imbalance)

            # 2f. Log snapshot
            snap = self._build_snapshot(t, event.event_type.name, trades)
            self.history.snapshots.append(snap)
            self.history.trades.extend(trades)

        return self.history

    # ── Private helpers ────────────────────────────────────────────────

    def _warmup_book(self) -> None:
        """
        Pre-populate 3 symmetric price levels each side.
        Uses direct price construction (not the event generator) to
        guarantee valid initial quotes regardless of depth_lambda.
        """
        p = self.params
        mid = p.ref_price
        half_spread = p.initial_spread / 2

        for i in range(1, 4):
            bid_px = round(
                (mid - half_spread - (i - 1) * p.tick_size) / p.tick_size
            ) * p.tick_size
            ask_px = round(
                (mid + half_spread + (i - 1) * p.tick_size) / p.tick_size
            ) * p.tick_size
            qty = 5 + i * 2

            bid_order = Order.limit("buy",  price=bid_px, quantity=qty, timestamp=0.0)
            ask_order = Order.limit("sell", price=ask_px, quantity=qty, timestamp=0.0)

            _, rested_bid = self.order_book.add_limit_order(bid_order)
            _, rested_ask = self.order_book.add_limit_order(ask_order)

            for rested in (rested_bid, rested_ask):
                if rested is not None:
                    ct = self.event_generator.generate_cancel_time(0.0)
                    self._schedule_cancel(rested.order_id, ct)

        self.event_generator.update_state(
            mid=self.order_book.get_mid_price(),
            imbalance=self.order_book.get_imbalance(),
        )

    def _schedule_cancel(self, order_id: int, cancel_time: float) -> None:
        heapq.heappush(self._cancel_heap, (cancel_time, order_id))
        self._pending_cancels[order_id] = cancel_time

    def _process_cancellations_before(self, t_cutoff: float) -> None:
        """
        Fire all pending cancellations with cancel_time <= t_cutoff.
        Skip orders already filled (no longer in _order_index).
        """
        while self._cancel_heap and self._cancel_heap[0][0] <= t_cutoff:
            cancel_time, order_id = heapq.heappop(self._cancel_heap)

            if order_id in self.order_book._order_index:
                if self.order_book.cancel_order(order_id):
                    self._pending_cancels.pop(order_id, None)
                    snap = self._build_snapshot(cancel_time, "CANCEL", [])
                    self.history.snapshots.append(snap)

    def _event_to_order(self, event: Event) -> Order:
        side = event.side()
        qty  = event.quantity
        t    = event.arrival_time
        if event.is_market():
            return Order.market(side=side, quantity=qty, timestamp=t)
        return Order.limit(side=side, price=event.price, quantity=qty, timestamp=t)

    def _build_snapshot(
        self,
        t: float,
        event_type_name: str,
        trades: List[Trade],
    ) -> Dict[str, Any]:
        """
        Build a state snapshot.  Includes imbalance and resting volume
        so users can study book dynamics and verify stationarity.
        """
        return {
            "time":         t,
            "event_type":   event_type_name,
            "mid_price":    self.order_book.get_mid_price(),
            "best_bid":     self.order_book.get_best_bid(),
            "best_ask":     self.order_book.get_best_ask(),
            "spread":       self.order_book.get_spread(),
            "imbalance":    self.order_book.get_imbalance(),
            "bid_volume":   self.order_book.get_best_level_volume("buy"),
            "ask_volume":   self.order_book.get_best_level_volume("sell"),
            "n_resting":    len(self.order_book._order_index),
            "n_trades":     len(trades),
            "trade_prices": [tr.price for tr in trades],
            "trade_qty":    sum(tr.quantity for tr in trades),
        }
