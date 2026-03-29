"""
lob_gui.py
==========
Web-based GUI for the LOB Simulator.

Run: python lob_gui.py
Then open: http://localhost:5000

Features:
- Real-time order book depth visualization
- Mid-price chart
- Trade history feed
- Simulation controls (play/pause/step/reset)
- Parameter adjustment
"""

from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import numpy as np

from lob_simulator import Simulator, SimParams, OrderBook, Order
from lob_simulator.simulation.events import EventType

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lob-simulator-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global simulation state
sim_state = {
    'simulator': None,
    'history': None,
    'running': False,
    'current_time': 0.0,
    'params': SimParams(),
    'seed': 42,
    'book_data': {'bids': [], 'asks': [], 'trades': [], 'mid_history': []},
    'lock': threading.Lock()
}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>LOB Simulator Visualizer</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .header p { opacity: 0.9; }
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        .panel {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .panel h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .control-group label {
            font-size: 0.9em;
            color: #aaa;
        }
        .control-group input[type="range"] {
            width: 100%;
        }
        .control-group span.value {
            color: #667eea;
            font-weight: bold;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.2s;
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-primary:hover { background: #5a6fd6; }
        .btn-danger {
            background: #e74c3c;
            color: white;
        }
        .btn-danger:hover { background: #c0392b; }
        .btn-success {
            background: #27ae60;
            color: white;
        }
        .btn-success:hover { background: #219a52; }
        .order-book {
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
        }
        .book-row {
            display: flex;
            align-items: center;
            height: 24px;
            margin: 2px 0;
        }
        .book-bar {
            height: 100%;
            transition: width 0.3s ease;
        }
        .bid-bar {
            background: linear-gradient(90deg, #27ae60, #2ecc71);
            margin-right: 10px;
        }
        .ask-bar {
            background: linear-gradient(90deg, #e74c3c, #c0392b);
            margin-left: 10px;
        }
        .book-price {
            width: 70px;
            text-align: center;
            font-weight: bold;
        }
        .bid-price { color: #2ecc71; }
        .ask-price { color: #e74c3c; }
        .book-qty {
            width: 50px;
            text-align: right;
            color: #aaa;
        }
        .spread-indicator {
            text-align: center;
            padding: 10px;
            background: #0f3460;
            border-radius: 6px;
            margin: 10px 0;
        }
        .spread-indicator .label {
            font-size: 0.8em;
            color: #888;
        }
        .spread-indicator .value {
            font-size: 1.5em;
            font-weight: bold;
            color: #f39c12;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .stat-box {
            background: #0f3460;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-box .label {
            font-size: 0.8em;
            color: #888;
            margin-bottom: 5px;
        }
        .stat-box .value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        .trades-feed {
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.8em;
        }
        .trade-item {
            padding: 8px;
            margin: 3px 0;
            background: #0f3460;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
        }
        .trade-buy { border-left: 3px solid #27ae60; }
        .trade-sell { border-left: 3px solid #e74c3c; }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-running { background: #27ae60; animation: pulse 1s infinite; }
        .status-paused { background: #f39c12; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Limit Order Book Simulator</h1>
        <p>Real-time visualization of market microstructure dynamics</p>
    </div>

    <div class="main-grid">
        <!-- Left Panel: Controls & Stats -->
        <div class="panel">
            <h2>⚙️ Controls</h2>
            <div class="controls">
                <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                    <button id="btn-start" class="btn btn-success" onclick="toggleSimulation()">
                        <span id="status-dot" class="status-indicator status-paused"></span>
                        <span id="btn-text">Start</span>
                    </button>
                    <button class="btn btn-danger" onclick="resetSimulation()">Reset</button>
                </div>

                <div class="control-group">
                    <label>Arrival Rate (λ): <span class="value" id="val-arrival">10.0</span></label>
                    <input type="range" id="arrival-rate" min="1" max="50" step="1" value="10"
                           oninput="updateParam('arrival_rate', this.value)">
                </div>

                <div class="control-group">
                    <label>Cancel Rate (μ): <span class="value" id="val-cancel">2.0</span></label>
                    <input type="range" id="cancel-rate" min="0.5" max="10" step="0.5" value="2"
                           oninput="updateParam('mu_cancel', this.value)">
                </div>

                <div class="control-group">
                    <label>Imbalance Alpha: <span class="value" id="val-alpha">0.5</span></label>
                    <input type="range" id="alpha" min="0" max="1" step="0.1" value="0.5"
                           oninput="updateParam('imbalance_alpha', this.value)">
                </div>

                <div class="control-group">
                    <label>Depth Lambda: <span class="value" id="val-depth">50</span></label>
                    <input type="range" id="depth" min="10" max="100" step="10" value="50"
                           oninput="updateParam('depth_lambda', this.value)">
                </div>
            </div>

            <h2 style="margin-top: 30px;">📈 Statistics</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="label">Time</div>
                    <div class="value" id="stat-time">0.00</div>
                </div>
                <div class="stat-box">
                    <div class="label">Events</div>
                    <div class="value" id="stat-events">0</div>
                </div>
                <div class="stat-box">
                    <div class="label">Trades</div>
                    <div class="value" id="stat-trades">0</div>
                </div>
                <div class="stat-box">
                    <div class="label">Resting</div>
                    <div class="value" id="stat-resting">0</div>
                </div>
                <div class="stat-box">
                    <div class="label">Imbalance</div>
                    <div class="value" id="stat-imbalance">0.00</div>
                </div>
                <div class="stat-box">
                    <div class="label">Mid Price</div>
                    <div class="value" id="stat-mid">100.00</div>
                </div>
            </div>
        </div>

        <!-- Middle Panel: Order Book & Charts -->
        <div class="panel">
            <h2>📖 Order Book Depth</h2>
            <div class="order-book" id="order-book">
                <!-- Asks (sells) -->
                <div id="asks-container"></div>

                <!-- Spread -->
                <div class="spread-indicator">
                    <div class="label">SPREAD</div>
                    <div class="value" id="spread-value">0.04</div>
                </div>

                <!-- Bids (buys) -->
                <div id="bids-container"></div>
            </div>

            <h2 style="margin-top: 30px;">📉 Price Chart</h2>
            <div class="chart-container">
                <canvas id="priceChart"></canvas>
            </div>
        </div>

        <!-- Right Panel: Recent Trades -->
        <div class="panel">
            <h2>💰 Recent Trades</h2>
            <div class="trades-feed" id="trades-feed">
                <div style="color: #666; text-align: center; padding: 20px;">
                    No trades yet...
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let isRunning = false;
        let priceChart;
        let midPriceHistory = [];
        let timeHistory = [];

        // Initialize chart
        function initChart() {
            const ctx = document.getElementById('priceChart').getContext('2d');
            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Mid Price',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            ticks: {
                                color: '#888',
                                callback: function(value) {
                                    return value.toFixed(2);
                                }
                            },
                            grid: {
                                color: '#333'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#eee' }
                        }
                    }
                }
            });
        }

        function toggleSimulation() {
            isRunning = !isRunning;
            const btnText = document.getElementById('btn-text');
            const statusDot = document.getElementById('status-dot');

            if (isRunning) {
                btnText.textContent = 'Pause';
                statusDot.className = 'status-indicator status-running';
                socket.emit('start_simulation');
            } else {
                btnText.textContent = 'Resume';
                statusDot.className = 'status-indicator status-paused';
                socket.emit('pause_simulation');
            }
        }

        function resetSimulation() {
            isRunning = false;
            document.getElementById('btn-text').textContent = 'Start';
            document.getElementById('status-dot').className = 'status-indicator status-paused';
            midPriceHistory = [];
            timeHistory = [];
            if (priceChart) {
                priceChart.data.labels = [];
                priceChart.data.datasets[0].data = [];
                priceChart.update();
            }
            document.getElementById('trades-feed').innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">No trades yet...</div>';
            socket.emit('reset_simulation');
        }

        function updateParam(param, value) {
            const numValue = parseFloat(value);
            // Map param names to display element IDs
            const idMap = {
                'arrival_rate': 'val-arrival',
                'mu_cancel': 'val-cancel',
                'imbalance_alpha': 'val-alpha',
                'depth_lambda': 'val-depth'
            };
            const displayId = idMap[param];
            if (displayId) {
                document.getElementById(displayId).textContent = numValue;
            }
            socket.emit('update_param', {param: param, value: numValue});
        }

        function updateOrderBook(data) {
            const asksContainer = document.getElementById('asks-container');
            const bidsContainer = document.getElementById('bids-container');
            const spreadValue = document.getElementById('spread-value');

            // Update spread
            spreadValue.textContent = data.spread ? data.spread.toFixed(2) : '—';

            // Update asks (reverse order so best ask is at bottom, closest to spread)
            const asks = data.asks.slice().reverse();
            asksContainer.innerHTML = asks.map(([price, qty]) => `
                <div class="book-row">
                    <div class="book-bar ask-bar" style="width: ${Math.min(qty * 3, 100)}px; margin-left: auto;"></div>
                    <div class="book-price ask-price">${price.toFixed(2)}</div>
                    <div class="book-qty">${qty}</div>
                </div>
            `).join('');

            // Update bids
            bidsContainer.innerHTML = data.bids.map(([price, qty]) => `
                <div class="book-row">
                    <div class="book-bar bid-bar" style="width: ${Math.min(qty * 3, 100)}px;"></div>
                    <div class="book-price bid-price">${price.toFixed(2)}</div>
                    <div class="book-qty">${qty}</div>
                </div>
            `).join('');
        }

        function addTrade(trade) {
            const feed = document.getElementById('trades-feed');
            if (feed.children.length === 1 && feed.children[0].textContent.includes('No trades')) {
                feed.innerHTML = '';
            }

            const tradeEl = document.createElement('div');
            tradeEl.className = `trade-item ${trade.side}`;
            tradeEl.innerHTML = `
                <span>#${trade.id}</span>
                <span>${trade.qty} @ ${trade.price.toFixed(2)}</span>
            `;

            feed.insertBefore(tradeEl, feed.firstChild);

            // Keep only last 20 trades
            while (feed.children.length > 20) {
                feed.removeChild(feed.lastChild);
            }
        }

        function updateStats(data) {
            document.getElementById('stat-time').textContent = data.time.toFixed(2);
            document.getElementById('stat-events').textContent = data.events;
            document.getElementById('stat-trades').textContent = data.trades;
            document.getElementById('stat-resting').textContent = data.resting;
            document.getElementById('stat-imbalance').textContent = data.imbalance.toFixed(2);
            document.getElementById('stat-mid').textContent = data.mid.toFixed(2);
        }

        function updateChart(data) {
            if (!priceChart) return;

            midPriceHistory.push(data.mid);
            timeHistory.push(data.time);

            // Keep last 100 points
            if (midPriceHistory.length > 100) {
                midPriceHistory.shift();
                timeHistory.shift();
            }

            priceChart.data.labels = timeHistory.map(t => t.toFixed(1));
            priceChart.data.datasets[0].data = midPriceHistory;
            priceChart.update('none'); // 'none' for performance
        }

        // Socket events
        socket.on('book_update', (data) => {
            updateOrderBook(data);
            updateStats(data);
            updateChart(data);
            if (data.new_trade) {
                addTrade(data.new_trade);
            }
        });

        socket.on('simulation_reset', () => {
            document.getElementById('order-book').innerHTML = `
                <div id="asks-container"></div>
                <div class="spread-indicator">
                    <div class="label">SPREAD</div>
                    <div class="value" id="spread-value">—</div>
                </div>
                <div id="bids-container"></div>
            `;
        });

        // Initialize
        initChart();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@socketio.on('update_param')
def handle_update_param(data):
    with sim_state['lock']:
        param = data['param']
        value = data['value']
        current_params = sim_state['params']

        if param == 'arrival_rate':
            current_params.arrival_rate = value
        elif param == 'mu_cancel':
            current_params.mu_cancel = value
        elif param == 'imbalance_alpha':
            current_params.imbalance_alpha = value
        elif param == 'depth_lambda':
            current_params.depth_lambda = value


@socketio.on('reset_simulation')
def handle_reset():
    with sim_state['lock']:
        sim_state['running'] = False
        sim_state['current_time'] = 0.0
        sim_state['book_data'] = {'bids': [], 'asks': [], 'trades': [], 'mid_history': []}
        sim_state['history'] = None

        # Create fresh simulator
        params = SimParams(
            arrival_rate=sim_state['params'].arrival_rate,
            mu_cancel=sim_state['params'].mu_cancel,
            imbalance_alpha=sim_state['params'].imbalance_alpha,
            depth_lambda=sim_state['params'].depth_lambda
        )
        sim_state['simulator'] = Simulator(params=params, seed=sim_state['seed'])
        sim_state['history'] = sim_state['simulator'].history

    emit('simulation_reset')


@socketio.on('start_simulation')
def handle_start():
    with sim_state['lock']:
        if sim_state['simulator'] is None:
            handle_reset()
        sim_state['running'] = True

    # Start simulation in background thread
    threading.Thread(target=simulation_loop, daemon=True).start()


@socketio.on('pause_simulation')
def handle_pause():
    with sim_state['lock']:
        sim_state['running'] = False


def simulation_loop():
    """Background thread that runs the simulation."""
    while True:
        with sim_state['lock']:
            if not sim_state['running']:
                break

            sim = sim_state['simulator']
            if sim is None:
                break

            # Run one event
            current_t = sim_state['current_time']

            event, order, trades = sim.advance(current_t)
            t = event.arrival_time
            new_trade = None

            # Check for new trade
            if trades:
                trade = trades[-1]
                side = "buy" if trade.buy_order_id == order.order_id else "sell"
                new_trade = {
                    'id': trade.trade_id,
                    'price': trade.price,
                    'qty': trade.quantity,
                    'side': side
                }

            # Get order book snapshot
            book_snapshot = sim.order_book.snapshot()

            # Get stats
            stats = {
                'time': t,
                'mid': book_snapshot['mid_price'] or 100.0,
                'spread': book_snapshot['spread'],
                'imbalance': book_snapshot['imbalance'],
                'events': len(sim.history.snapshots),
                'trades': len(sim.history.trades),
                'resting': len(sim.order_book._order_index),
                'bids': book_snapshot['bids'][:8],  # Top 8 levels
                'asks': book_snapshot['asks'][:8],
                'new_trade': new_trade
            }

            sim_state['current_time'] = t

        # Emit update (outside lock to avoid holding it during emit)
        socketio.emit('book_update', stats)

        # Small delay to control simulation speed
        time.sleep(0.05)  # 20 updates/second


if __name__ == '__main__':
    print("=" * 60)
    print("LOB Simulator GUI")
    print("=" * 60)
    print("\nOpen your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")

    # Initialize simulator
    with sim_state['lock']:
        sim_state['simulator'] = Simulator(params=sim_state['params'], seed=sim_state['seed'])
        sim_state['history'] = sim_state['simulator'].history

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
