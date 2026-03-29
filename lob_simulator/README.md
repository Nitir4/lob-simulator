# Stochastic Limit Order Book Simulator

A professional-grade, event-driven LOB simulator written in clean Python.
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
│   ├── event_generator.py# Poisson arrivals + geometric price placement
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
| Arrivals | Poisson process (Exp(λ) interarrival) | Empirically validated for LOBs |
| Price placement | Geometric offset from best quote | Clusters near spread (realistic) |
| Simulation loop | Event-driven (not time-stepped) | O(N_events), no empty iterations |

## Running Tests

```bash
cd /home/nitir505/Ritin/probability
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
    limit_buy_prob = 0.40,
    limit_sell_prob= 0.40,
    market_buy_prob= 0.10,
    market_sell_prob=0.10,
    offset_geom_p  = 0.5,     # price offset distribution
    size_geom_p    = 0.5,     # order size distribution
    max_size       = 20,
)
sim = Simulator(params=params, seed=0)
```

## What to Build Next

- **Order cancellations** — Poisson cancel events remove resting orders
- **Self-exciting arrivals** — Hawkes process for clustering after trades
- **Execution strategies** — TWAP, VWAP, optimal liquidation (Almgren-Chriss)
- **Market impact modelling** — track how your own orders move the price
