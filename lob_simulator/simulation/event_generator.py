"""
simulation/event_generator.py  (PHASE 2-3 REVISION)
=====================================================
Changes in this revision:

PHASE 2 — Price Placement Fix
------------------------------
OLD (broken): offset ~ Geometric(p)-1 relative to best_bid/ask.
  This CANNOT generate inside-spread orders. Limit buys can only be
  placed AT or BELOW best_bid, never above it. As shown in Phase 1
  critique, this means the event generator never tightens the spread.

NEW (correct): delta ~ Exp(lambda_depth) relative to mid-price.
  For limit buy:  price = mid - delta  (always below mid)
  For limit sell: price = mid + delta  (always above mid)
  delta > 0, so lim_buy < mid < lim_sell always.
  But since mid = (bid + ask) / 2, an order placed at (mid - epsilon)
  with epsilon < half_spread will be INSIDE the spread if:
      epsilon < half_spread, i.e., price > bid.
  P(inside spread) = P(delta < half_spread) = 1 - exp(-lambda × half_spread)
  This is tunable via lambda_depth.

WHY Exponential and not Geometric?
  Geometric is a discrete approximation to the Exponential on a tick grid.
  The Exponential is cleaner analytically and matches the Cont-Stoikov
  (2010) model for limit order placement away from mid.
  We round to nearest tick after sampling.

PHASE 3 — State-Dependent Market Order Rates (Queue Imbalance)
--------------------------------------------------------------
OLD (broken): event type probabilities are CONSTANTS — IID draws.
  This produces zero expected price drift by symmetry.

NEW (correct): market order rates are modulated by queue imbalance I.
  Base probability vector: [p_lb, p_ls, p_mb, p_ms] (sum=1)
  After observing imbalance I ∈ [-1, 1]:
    p_mb' = p_mb * (1 + alpha * I)    market buys more likely when bid heavy
    p_ms' = p_ms * (1 - alpha * I)    market sells more likely when ask heavy
    p_lb, p_ls remain unchanged (limit orders less sensitive to imbalance)
    Renormalize to sum=1.

  WHY market orders more sensitive than limit orders?
    Market orders express URGENT directionality — the trader believes they
    must act NOW and is willing to pay the spread. This urgency is driven
    by either private information or reaction to visible imbalance.
    Limit orders are more patient; their placement IS affected by imbalance
    but less so in a simple model.

  ACADEMIC SOURCE: Cont & Larrard (2013) "How Order Flow Drives Price Changes"
  The prediction P(price increase) = f(I) is the core result of this paper.
  Our state-dependent rates implement the reduced-form intuition.

PARAMETER CHANGES in SimParams:
  - Removed: offset_geom_p (replaced by depth_lambda)
  - Added: depth_lambda (Exponential price placement rate)
  - Added: imbalance_alpha (sensitivity of market orders to imbalance)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from lob_simulator.simulation.events import Event, EventType


@dataclass
class SimParams:
    """
    All tunable parameters for the event generator and simulator.

    Arrival & cancellation
    ----------------------
    arrival_rate : λ_total — total order arrival rate (events per unit time).
    mu_cancel    : μ — cancellation rate per resting order.
                   Mean order lifetime = 1/mu_cancel time units.
                   Stationarity requires: mu_cancel × mean_queue_depth ≈
                   λ_limit × (1 - P_fill). Check that mean_queue_depth
                   stabilizes in simulation output.

    Market microstructure
    ---------------------
    tick_size       : Minimum price increment.
    ref_price       : Starting reference mid-price.
    initial_spread  : Starting bid-ask spread in price units.

    Event mix (must sum to 1.0)
    ---------------------------
    These are BASE probabilities, before imbalance adjustment.
    p0_limit_buy, p0_limit_sell: passive liquidity providers.
    p0_market_buy, p0_market_sell: aggressive liquidity takers.

    Price placement
    ---------------
    depth_lambda : Rate parameter λ for Exp(λ) placement distribution.
                   Distance from mid: delta ~ Exp(depth_lambda).
                   The mean distance from mid = 1/depth_lambda.
                   Larger depth_lambda → orders cluster near mid (narrow book).
                   Smaller depth_lambda → orders spread out (deep book).
                   Recommended: set so mean offset ≈ 1-2 ticks.
                   Example: tick=0.01, depth_lambda=50 → mean offset=0.02 (2 ticks).

    State dependence
    ----------------
    imbalance_alpha : Sensitivity of market order rates to queue imbalance I.
                      alpha=0 → IID arrivals (no state dependence, no drift).
                      alpha=1 → maximum sensitivity.
                      alpha=0.5 → moderate, recommended for academic use.
                      Increases realism but makes the model non-Markovian in
                      the strict sense (rates depend on current book state).

    Order sizes
    -----------
    size_geom_p : Parameter for Geometric(p) order size distribution.
    max_size    : Cap on order sizes.
    """

    arrival_rate:    float = 10.0
    mu_cancel:       float = 2.0

    tick_size:       float = 0.01
    ref_price:       float = 100.0
    initial_spread:  float = 0.04    # 4 ticks

    p0_limit_buy:    float = 0.40
    p0_limit_sell:   float = 0.40
    p0_market_buy:   float = 0.10
    p0_market_sell:  float = 0.10

    depth_lambda:       float = 50.0   # mean offset = 0.02 = 2 ticks
    imbalance_alpha:    float = 0.5    # moderate state dependence
    size_geom_p:        float = 0.5
    max_size:           int   = 20

    def __post_init__(self) -> None:
        total = (self.p0_limit_buy + self.p0_limit_sell
                 + self.p0_market_buy + self.p0_market_sell)
        assert abs(total - 1.0) < 1e-9, f"Base probs must sum to 1, got {total}"
        assert self.mu_cancel > 0, "mu_cancel must be positive"
        assert 0.0 <= self.imbalance_alpha <= 1.0, "alpha must be in [0,1]"
        assert self.depth_lambda > 0, "depth_lambda must be positive"


class EventGenerator:
    """
    Generates market events using:
      - Single Poisson process with rate arrival_rate (interarrival ~ Exp).
      - State-dependent multinomial for event type (imbalance-modulated).
      - Exponential-from-midprice limit order placement (inside-spread possible).

    State is updated by the Simulator after each event via update_state().
    """

    def __init__(
        self,
        params: Optional[SimParams] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.params = params or SimParams()
        self.rng = np.random.default_rng(seed)

        p = self.params
        self._mid: float     = p.ref_price
        self._imbalance: float = 0.0    # I ∈ [-1, 1], updated each step

        self._base_probs = np.array([
            p.p0_limit_buy,
            p.p0_limit_sell,
            p.p0_market_buy,
            p.p0_market_sell,
        ])
        self._event_types = [
            EventType.LIMIT_BUY,
            EventType.LIMIT_SELL,
            EventType.MARKET_BUY,
            EventType.MARKET_SELL,
        ]

    def update_state(
        self,
        mid: Optional[float],
        imbalance: float,
    ) -> None:
        """
        Update the generator's view of current market state.
        Called by Simulator after each event is processed.

        Parameters
        ----------
        mid        : Current mid-price (for price anchoring).
        imbalance  : Current queue imbalance I ∈ [-1, 1].
        """
        if mid is not None:
            self._mid = mid
        self._imbalance = imbalance

    # ── backward-compat alias ────────────────────────────────────────
    def set_quotes(
        self,
        best_bid: Optional[float],
        best_ask: Optional[float],
    ) -> None:
        """Kept for backward compatibility. Prefer update_state()."""
        if best_bid is not None and best_ask is not None:
            self._mid = (best_bid + best_ask) / 2.0

    def next_event(self, current_time: float) -> Event:
        """
        Generate the next arrival event.

        1. Interarrival ~ Exp(arrival_rate) → Poisson process.
        2. Event type from imbalance-modulated probabilities.
        3. Price from Exp(depth_lambda) placement around midprice.
        4. Quantity from Geometric(size_geom_p).
        """
        interarrival = self.rng.exponential(1.0 / self.params.arrival_rate)
        arrival_time = current_time + interarrival

        event_type = self._sample_event_type()
        price      = self._sample_price(event_type)
        quantity   = self._sample_quantity()

        return Event.create(
            event_type=event_type,
            arrival_time=arrival_time,
            price=price,
            quantity=quantity,
        )

    def generate_cancel_time(self, order_timestamp: float) -> float:
        """
        Draw cancel time for a newly rested order.
        Lifetime ~ Exp(mu_cancel) — memoryless, valid Poisson cancel process.
        """
        lifetime = self.rng.exponential(1.0 / self.params.mu_cancel)
        return order_timestamp + lifetime

    # ── Private helpers ────────────────────────────────────────────────

    def _sample_event_type(self) -> EventType:
        """
        Imbalance-modulated event type sampling.

        Base probabilities are adjusted so that market buys are more
        likely when the bid side is heavy (I > 0) and vice versa.

        Adjustment:
          p_mb' = p_mb * (1 + alpha * I)
          p_ms' = p_ms * (1 - alpha * I)

        Limit order probabilities are left unchanged (they respond
        less to imbalance in this simplified model).

        After adjustment, we renormalize to sum=1.

        IMPORTANT: alpha * |I| <= alpha <= 1 ensures p_mb', p_ms' >= 0
        since base probs are >= 0. No clamping needed as long as alpha ∈ [0,1].
        """
        alpha = self.params.imbalance_alpha
        I     = self._imbalance

        probs = self._base_probs.copy()
        # Index 2 = MARKET_BUY, Index 3 = MARKET_SELL
        probs[2] *= (1.0 + alpha * I)
        probs[3] *= (1.0 - alpha * I)

        # Clamp negatives from floating point arithmetic, then renormalize
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()

        idx = self.rng.choice(len(self._event_types), p=probs)
        return self._event_types[idx]

    def _sample_price(self, event_type: EventType) -> Optional[float]:
        """
        Midprice-anchored placement: delta ~ Exp(depth_lambda).

        For limit buy:  price = mid - delta  (always below mid)
        For limit sell: price = mid + delta  (always above mid)

        P(order is inside spread) = P(delta < half_spread)
                                  = 1 - exp(-depth_lambda * half_spread)

        With depth_lambda=50, tick=0.01, spread=0.04:
          half_spread = 0.02
          P(inside spread) = 1 - exp(-50 * 0.02) = 1 - exp(-1) ≈ 0.63

        This means ~63% of limit orders attempt to improve the quote,
        of which roughly half cross and fill immediately, half tighten.
        This is more aggressive than real markets (5-15% inside spread),
        but produces visible spread dynamics. For a more conservative
        calibration, use depth_lambda ≈ 10-20.

        WHY place relative to MID and not best quote?
          If we place relative to best_bid (offset=0,1,2...):
            - Limit buys always land at or below best_bid.
            - The spread can only widen from limit orders.
          If we place relative to MID:
            - Some buys land above best_bid (inside spread) → tighten.
            - Some buys land below best_bid (outside spread) → add depth.
          This is the correct model (Cont-Stoikov 2010).
        """
        p = self.params
        if event_type in (EventType.MARKET_BUY, EventType.MARKET_SELL):
            return None

        delta = self.rng.exponential(scale=1.0 / p.depth_lambda)

        if event_type == EventType.LIMIT_BUY:
            raw = self._mid - delta
        else:
            raw = self._mid + delta

        # Round to nearest tick
        return round(raw / p.tick_size) * p.tick_size

    def _sample_quantity(self) -> int:
        p = self.params
        return min(int(self.rng.geometric(p=p.size_geom_p)), p.max_size)
