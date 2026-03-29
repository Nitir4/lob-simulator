"""Event-driven matching, independent cancellation lifetimes and diagnostics.

The constant-rate arrival clock is sampled separately from state-dependent
marks so preceding cancellations update the state first. Queue means distinguish
event sampling from time weighting; neither alone proves stationarity.
"""

from __future__ import annotations

import heapq
import math
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
    initial_snapshot: Optional[Dict[str, Any]] = None

    def time_average_resting(self) -> Optional[float]:
        """Integrate the piecewise-constant queue depth, including the final tail.

        None means no positive horizon or no captured initial state (e.g. an
        old/manually constructed history); it must not be replaced by event mean.
        """
        if self.duration <= 0 or self.initial_snapshot is None:
            return None
        previous_time = 0.0
        previous_depth = self.initial_snapshot['n_resting']
        integral = 0.0
        for snapshot in self.snapshots:
            t = snapshot['time']
            if not math.isfinite(t) or not previous_time <= t <= self.duration:
                raise ValueError('Snapshots must be chronological within the horizon.')
            integral += previous_depth*(t-previous_time)
            previous_time, previous_depth = t, snapshot['n_resting']
        integral += previous_depth*(self.duration-previous_time)
        return integral/self.duration

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

        # Descriptive queue diagnostics, not a stationarity test.
        peak_resting = max(resting_counts+[self.initial_snapshot['n_resting']] if self.initial_snapshot else resting_counts, default=0)
        mean_resting = (sum(resting_counts) / len(resting_counts)
                        if resting_counts else None)
        time_mean = self.time_average_resting()
        initial_mid = self.initial_snapshot.get('mid_price') if self.initial_snapshot else None
        start_mid = initial_mid if initial_mid is not None else (mids[0] if mids else None)
        end_mid = mids[-1] if mids else initial_mid

        lines = [
            "── Simulation Summary ──────────────────────────",
            f"  Duration      : {self.duration:.2f} time units",
            f"  Arrivals      : {n_arrive}",
            f"  Cancellations : {n_cancel}",
            f"  Trades        : {len(self.trades)}",
            f"  Mid (start)   : {start_mid:.4f}" if start_mid is not None else "  Mid : N/A",
            f"  Mid (end)     : {end_mid:.4f}" if end_mid is not None else "",
            f"  Mid drift     : {end_mid-start_mid:+.4f}" if start_mid is not None and end_mid is not None else "",
            f"  Peak resting  : {peak_resting}",
            f"  Mean resting (event): {mean_resting:.1f}" if mean_resting is not None else "  Mean resting (event): unavailable",
            f"  Mean resting (time) : {time_mean:.3f}" if time_mean is not None else "  Mean resting (time) : unavailable",
        ]
        if peak_resting > 1000:
            lines.append("  ⚠ Peak resting > 1000; inspect arrivals, fills and cancellations.")
            lines.append("    A count threshold is not a stationarity test.")
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
      + State-dependent event-type probabilities at a constant total arrival rate.
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
        if not math.isfinite(T) or T < 0:
            raise ValueError('Simulation horizon must be finite and nonnegative.')
        self._reset()
        self.history.duration = T
        self._warmup_book()
        t = 0.0
        while True:
            result = self.advance(t, horizon=T)
            if result is None:
                break
            t = result[0].arrival_time

        return self.history

    def advance(self, current_time: float, horizon: float | None = None):
        """Process cancellations, then one arrival, shared by batch and GUI.

        Returns (event, submitted_order, trades), or None at the horizon.
        Cancellation wins a timestamp tie. Arrival intensity is constant;
        its state-dependent type, price and size are sampled at arrival.
        """
        arrival_time = self.event_generator.next_arrival_time(current_time)
        cutoff = arrival_time if horizon is None else min(arrival_time, horizon)
        self._process_cancellations_before(cutoff)
        if horizon is not None and arrival_time > horizon:
            return None
        event = self.event_generator.event_at(arrival_time)
        order = self._event_to_order(event)
        if order.order_type == 'limit':
            trades, rested = self.order_book.add_limit_order(order)
        else:
            trades, rested = self.order_book.process_market_order(order), None
        if rested is not None:
            self._schedule_cancel(rested.order_id, self.event_generator.generate_cancel_time(arrival_time))
        for trade in trades:
            for order_id in (trade.buy_order_id, trade.sell_order_id):
                if order_id not in self.order_book._order_index:
                    self._update_generator_state()
        self._update_generator_state()
        self.history.snapshots.append(self._build_snapshot(arrival_time, event.event_type.name, trades))
        self.history.trades.extend(trades)
        return event, order, trades

    def _update_generator_state(self):
        self.event_generator.update_state(mid=self.order_book.get_mid_price(),
                                          imbalance=self.order_book.get_imbalance())

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
            bid_px = math.floor(
                (mid - half_spread - (i - 1) * p.tick_size) / p.tick_size
            ) * p.tick_size
            ask_px = math.ceil(
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
        self.history.initial_snapshot = self._build_snapshot(0.0, 'INITIAL', [])

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

            self._pending_cancels.pop(order_id, None)
            if order_id in self.order_book._order_index:
                if self.order_book.cancel_order(order_id):
                    self._update_generator_state()
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
        so users can study book dynamics; stationarity requires separate analysis.
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
