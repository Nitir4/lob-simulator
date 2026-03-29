# Stochastic Limit Order Book Simulator

An event-driven, stylized LOB simulator with price-time matching and cancellations.
Built for academic study of market microstructure and optimal execution.

## Project Structure

```
lob_simulator/
├── core/
│   ├── order.py          # Order dataclass (limit / market)
│   ├── trade.py          # Trade execution record (immutable)
│   └── order_book.py     # OrderBook: SortedDict + FIFO + matching engine
├── simulation/
│   ├── events.py         # Event dataclass + EventType enum
│   ├── event_generator.py# Poisson clock + exponential mid-price placement
│   └── simulator.py      # Event-driven simulation loop + history logging
└── tests/
    ├── test_order_book.py # Unit tests: LOB invariants
    └── test_simulator.py  # Integration tests: full simulation
```

## Quick Start

```python
from lob_simulator import Simulator, SimParams

# Run with defaults (arrival_rate=10, ref_price=100, T=60s)
sim = Simulator(seed=42)
history = sim.run(T=60.0)

print(history.summary())

# Convert to DataFrame for analysis
df = history.to_dataframe()
print(df[["time", "mid_price", "spread", "n_trades"]].head(10))
```

## Design Highlights

| Component | Design choice | Why |
|---|---|---|
| Price levels | `SortedDict` (sortedcontainers) | O(log n) insert, O(1) best bid/ask |
| Per-level queue | `collections.deque` | FIFO naturally: append right, pop left |
| Arrivals | Constant Poisson clock, normalized state-dependent marks | Explicit illustrative assumption |
| Price placement | Exponential offset from mid, outward tick rounding | Permits inside-spread passive quotes |
| Simulation loop | Event-driven (not time-stepped) | O(N_events), no empty iterations |

## Running Tests

```bash
python -m pytest lob_simulator/tests/ -v
```

## Dependencies

```
sortedcontainers   # pure-Python, zero-C sorted data structures
numpy              # random number generation
pandas             # (optional) for to_dataframe()
pytest             # (optional) for running tests
```

## Parameters

```python
from lob_simulator import SimParams

params = SimParams(
    arrival_rate   = 10.0,    # events per second
    tick_size      = 0.01,
    ref_price      = 100.0,
    initial_spread = 0.04,    # 4 ticks
    p0_limit_buy  = 0.40,
    p0_limit_sell = 0.40,
    market_buy_prob= 0.10,
    p0_market_sell=0.10,
    mu_cancel      = 2.0,     # independent per-order lifetime rate
    depth_lambda   = 50.0,    # mean mid-price offset = 0.02
    imbalance_alpha= 0.5,     # illustrative conditional mark sensitivity
    size_geom_p    = 0.5,     # order size distribution
    max_size       = 20,
)
sim = Simulator(params=params, seed=0)
```

## What to Build Next

- **Self-exciting arrivals** — Hawkes process for clustering after trades
- **Execution strategies** — TWAP, VWAP, optimal liquidation (Almgren-Chriss)
- **Market impact modelling** — track how your own orders move the price

## October 2026 correctness review

Batch runs and the live GUI now use `Simulator.advance(...)` for the same event path. The simulator samples the next arrival clock, processes all cancellations up to that time (cancellation wins ties), updates the generator state, and only then samples the state-dependent arrival type/price/size. Expired filled-order cancellation records are removed. The GUI previously never processed cancellations.

Orders are validated before matching: finite positive limit prices, positive integral quantities, valid side/type and finite nonnegative timestamps. An ID cannot be reused within a book, including after completion/cancellation. Auto-generated IDs remain globally unique. Resting order objects are mutable for partial matching; callers must not modify their side, price, ID or quantity after submission. Invariant checks verify index/object identity as well as levels and quantities.

Generator parameters are validated explicitly, including nonnegative probabilities and positive rates/ticks/sizes. Alpha=1 requires passive arrivals so the normalized mark distribution cannot collapse to zero in a one-sided book. Warm-up quotes round outward to preserve a positive spread; generated limit prices are rounded outward to ticks and deep bids stop at the minimum positive tick. This positive boundary changes the unconstrained placement distribution at low prices. Conditional mark probabilities depend on the complete book state; total arrival intensity stays constant. Normalizing adjusted market-order weights can also change passive effective intensities.

Regression tests cover cancellation-before-arrival state, end-of-horizon handling, the actual GUI worker with substituted emissions, malformed orders, ID reuse, valid ticks and 500 matching/cancellation operations checked against an independent list-based reference book. Run from the repository root:

```bash
pip install -r requirements.txt
pip install pytest
python -m pytest lob_simulator/tests -q
```

The tests verify implementation invariants and deterministic examples. Short seeded stationarity/drift checks are diagnostics, not a proof of a stationary market model or empirical price-dynamics calibration.

## Queue averages and interpretation

SimulationHistory captures the initial post-warmup state. time_average_resting()
integrates the resting order count over elapsed time, including the initial wait,
zero-duration ties and the tail after the last event. The summary labels both
its event-weighted and time-weighted means. Zero-horizon or older histories
without an initial state return no time mean rather than inventing one. This
counts orders, not their combined share quantity.

The intensity, placement, capped geometric size and cancellation laws are
illustrative assumptions. This is not a calibrated reproduction of Cont–Stoikov
or an empirical claim about all exchanges. A bounded short path or queue-count
threshold does not prove stationarity; compare predetermined ensembles and
report uncertainty when studying such hypotheses.
