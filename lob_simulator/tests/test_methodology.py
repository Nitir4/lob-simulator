from types import SimpleNamespace
import numpy as np
import pytest

from lob_simulator.simulation.simulator import Simulator, SimulationHistory
from lob_simulator.simulation.event_generator import SimParams, EventGenerator


def test_time_average_includes_initial_state_and_terminal_tail():
    history = SimulationHistory(duration=10., initial_snapshot={'n_resting': 10},
                                snapshots=[{'time': 9., 'n_resting': 1}])
    assert history.time_average_resting() == pytest.approx(9.1)
    assert np.mean([10, 1]) != history.time_average_resting()


def test_simultaneous_events_do_not_add_duration():
    history = SimulationHistory(duration=10., initial_snapshot={'n_resting': 10},
                                snapshots=[{'time': 5., 'n_resting': 1}, {'time': 5., 'n_resting': 8}])
    assert history.time_average_resting() == pytest.approx(9.)


def test_no_events_still_has_a_time_average_but_zero_horizon_does_not():
    history = SimulationHistory(duration=10., initial_snapshot={'n_resting': 6})
    assert history.time_average_resting() == 6.
    history.duration = 0.
    assert history.time_average_resting() is None
    assert SimulationHistory(duration=10.).time_average_resting() is None


def test_batch_history_captures_warmup_and_matches_manual_integral():
    history = Simulator(seed=505).run(10.)
    assert history.initial_snapshot['time'] == 0.
    assert history.initial_snapshot['n_resting'] == 6
    times = np.array([0.]+[s['time'] for s in history.snapshots]+[history.duration])
    depths = np.array([6]+[s['n_resting'] for s in history.snapshots])
    expected = np.dot(depths, np.diff(times))/history.duration
    assert history.time_average_resting() == pytest.approx(expected)
    assert 'Mean resting (time)' in history.summary()


def test_effective_type_probabilities_are_normalized_at_constant_total_rate():
    params = SimParams(p0_limit_buy=.3, p0_limit_sell=.4, p0_market_buy=.2, p0_market_sell=.1)
    generator = EventGenerator(params, seed=505)
    generator.update_state(100., 1.)
    captured = []
    generator.rng = SimpleNamespace(choice=lambda n, p: captured.append(p.copy()) or 0)
    generator._sample_event_type()
    weights = np.array([.3, .4, .3, .05])
    np.testing.assert_allclose(captured[0], weights/weights.sum())
    assert params.arrival_rate == 10.
    assert captured[0][0] != .3  # passive effective probability changes too


def test_invalid_time_order_cannot_produce_a_reported_mean():
    history = SimulationHistory(duration=10., initial_snapshot={'n_resting': 6},
                                snapshots=[{'time': 5., 'n_resting': 3}, {'time': 4., 'n_resting': 4}])
    with pytest.raises(ValueError, match='chronological'): history.time_average_resting()


@pytest.mark.parametrize('horizon', [0., 10.])
def test_empty_event_history_does_not_invent_an_event_mean(horizon):
    history = SimulationHistory(duration=horizon, initial_snapshot={'n_resting': 6, 'mid_price': 100.})
    assert 'Mean resting (event): unavailable' in history.summary()


def test_mid_drift_includes_the_move_before_the_first_recorded_event():
    history = SimulationHistory(duration=10., initial_snapshot={'n_resting': 6, 'mid_price': 100.},
                                snapshots=[{'time': 1., 'n_resting': 5, 'mid_price': 101., 'event_type': 'LIMIT_BUY'}])
    summary = history.summary()
    assert 'Mid (start)   : 100.0000' in summary
    assert 'Mid drift     : +1.0000' in summary
