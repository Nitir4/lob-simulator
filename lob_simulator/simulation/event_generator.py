"""Constant-rate arrival clock with state-dependent conditional event marks.

Market-order weights respond to best-level imbalance, then all weights are
normalized. Effective type intensities equal arrival_rate times the normalized
probabilities; passive intensities can change too. This is an illustrative
marked Poisson model, not a calibrated reproduction of a published LOB model.
Limit prices use mid-anchored exponential offsets and outward tick rounding.
"""

from __future__ import annotations

import math
import numbers
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
                   Queue-depth behavior also depends on fills and arrivals.
                   A short run or count threshold does not prove stationarity.

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
    imbalance_alpha : Sensitivity of conditional market-order weights to imbalance I.
                      alpha=0 → constant type probabilities; realized paths can drift.
                      alpha=1 → maximum sensitivity.
                      alpha=0.5 → illustrative moderate sensitivity, not calibrated.
                      State-dependent rates remain Markovian in the complete
                      order-book state.

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
        positive = [self.arrival_rate, self.mu_cancel, self.tick_size, self.ref_price,
                    self.initial_spread, self.depth_lambda]
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError('Rates, tick size, reference price, spread and depth must be finite and positive.')
        probs = [self.p0_limit_buy, self.p0_limit_sell, self.p0_market_buy, self.p0_market_sell]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in probs) or not math.isclose(sum(probs), 1., abs_tol=1e-9):
            raise ValueError('Base probabilities must be nonnegative and sum to one.')
        if not math.isfinite(self.imbalance_alpha) or not 0 <= self.imbalance_alpha < 1:
            # At alpha=1 a one-sided market-only mix can have zero total intensity.
            if self.imbalance_alpha != 1 or self.p0_limit_buy+self.p0_limit_sell == 0:
                raise ValueError('Imbalance alpha must be in [0,1], with passive arrivals when alpha is one.')
        if not math.isfinite(self.size_geom_p) or not 0 < self.size_geom_p <= 1:
            raise ValueError('Geometric size probability must be in (0,1].')
        if isinstance(self.max_size, bool) or not isinstance(self.max_size, numbers.Integral) or self.max_size < 1:
            raise ValueError('Maximum size must be a positive integer.')
        if self.ref_price-self.initial_spread/2-3*self.tick_size <= 0:
            raise ValueError('Reference price must accommodate three positive warm-up bid levels.')


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
            if not math.isfinite(mid) or mid <= 0:
                raise ValueError('Mid-price must be finite and positive.')
            self._mid = mid
        if not math.isfinite(imbalance) or not -1 <= imbalance <= 1:
            raise ValueError('Imbalance must be finite and in [-1,1].')
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
        return self.event_at(self.next_arrival_time(current_time))

    def next_arrival_time(self, current_time: float) -> float:
        """Draw only the clock; state-dependent marks are drawn at arrival."""
        if not math.isfinite(current_time) or current_time < 0:
            raise ValueError('Current time must be finite and nonnegative.')
        return current_time + self.rng.exponential(1.0 / self.params.arrival_rate)

    def event_at(self, arrival_time: float) -> Event:
        """Draw an arrival using the state after all preceding cancellations."""
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

        Raw limit-order weights are unchanged. Normalization can change
        their final probabilities and hence their effective intensities.

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

        # Renormalization also changes the final limit-order probabilities.
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()

        idx = self.rng.choice(len(self._event_types), p=probs)
        return self._event_types[idx]

    def _sample_price(self, event_type: EventType) -> Optional[float]:
        """
        Midprice-anchored placement: delta ~ Exp(depth_lambda).

        For limit buy: raw price = mid - delta; round down to a positive tick
        For limit sell: raw price = mid + delta; round up to a tick

        P(order is inside spread) = P(delta < half_spread)
                                  = 1 - exp(-depth_lambda * half_spread)

        With depth_lambda=50, tick=0.01, spread=0.04:
          half_spread = 0.02
          P(inside spread) = 1 - exp(-50 * 0.02) = 1 - exp(-1) ≈ 0.63

        This is the continuous-price probability before outward tick rounding.
        Passive orders remain on their side of the mid and do not deliberately
        cross the opposite quote. No empirical inside-spread frequency is claimed.

        WHY place relative to MID and not best quote?
          If we place relative to best_bid (offset=0,1,2...):
            - Limit buys always land at or below best_bid.
            - The spread can only widen from limit orders.
          If we place relative to MID:
            - Some buys land above best_bid (inside spread) → tighten.
            - Some buys land below best_bid (outside spread) → add depth.
          This is our mid-anchored placement assumption, not an exact paper reproduction.
        """
        p = self.params
        if event_type in (EventType.MARKET_BUY, EventType.MARKET_SELL):
            return None

        delta = self.rng.exponential(scale=1.0 / p.depth_lambda)

        if event_type == EventType.LIMIT_BUY:
            raw = self._mid - delta
        else:
            raw = self._mid + delta

        # Round outward to the tick grid. A deep buy cannot quote a negative
        # price; place it at the minimum positive tick instead.
        ticks = math.floor(raw / p.tick_size) if event_type == EventType.LIMIT_BUY else math.ceil(raw / p.tick_size)
        return max(1, ticks) * p.tick_size

    def _sample_quantity(self) -> int:
        p = self.params
        return min(int(self.rng.geometric(p=p.size_geom_p)), p.max_size)
