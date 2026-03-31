"""Deterministic event-clock and book-integrity regressions."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from lob_simulator.core.order import Order
from lob_simulator.core.order_book import OrderBook
from lob_simulator.simulation.simulator import Simulator
from lob_simulator.simulation.event_generator import SimParams,EventGenerator
from lob_simulator.simulation.events import Event,EventType


@pytest.mark.parametrize('field,value',[('quantity',0),('quantity',-1),('quantity',.5),('side','invalid'),('price',np.nan),('price',-1),('timestamp',np.inf)])
def test_bad_orders_cannot_mutate_book(field,value):
    book=OrderBook();maker=Order.limit('sell',100,10,0)
    book.add_limit_order(maker)
    taker=Order.limit('buy',101,5,1);setattr(taker,field,value)
    with pytest.raises(ValueError):book.add_limit_order(taker)
    assert maker.quantity==10
    book.assert_invariants()


def test_duplicate_id_is_rejected_before_matching_or_index_corruption():
    book=OrderBook();bid=Order.limit('buy',99,5,0);ask=Order.limit('sell',100,5,0)
    book.add_limit_order(bid);book.add_limit_order(ask)
    duplicate=Order(bid.order_id,'buy','limit',101,5,1)
    with pytest.raises(ValueError,match='already been used'):book.add_limit_order(duplicate)
    assert ask.quantity==5
    assert book.cancel_order(bid.order_id)
    with pytest.raises(ValueError,match='already been used'):book.add_limit_order(duplicate)
    book.assert_invariants()


@pytest.mark.parametrize('kwargs',[{'arrival_rate':0},{'tick_size':0},{'p0_limit_buy':-.4,'p0_limit_sell':1.2}, {'mu_cancel':np.nan},{'size_geom_p':0},{'max_size':0},{'initial_spread':np.inf}, {'p0_limit_buy':0,'p0_limit_sell':0,'p0_market_buy':1,'p0_market_sell':0,'imbalance_alpha':1}])
def test_bad_simulation_parameters_fail_early(kwargs):
    with pytest.raises(ValueError):SimParams(**kwargs)


def scheduled_simulator():
    sim=Simulator(seed=505)
    bid=Order.limit('buy',100,5,0);low=Order.limit('buy',98,5,0);ask=Order.limit('sell',102,5,0)
    for order in [bid,low,ask]:sim.order_book.add_limit_order(order)
    sim._update_generator_state()
    sim._schedule_cancel(bid.order_id,.5)
    seen=[]
    sim.event_generator.next_arrival_time=lambda t:t+1
    original=sim.event_generator.event_at
    def event_at(t):
        seen.append((t,sim.event_generator._mid,sim.event_generator._imbalance))
        return Event.create(EventType.MARKET_SELL,t,None,1)
    sim.event_generator.event_at=event_at
    return sim,seen


def test_cancellations_update_state_before_sampling_arrival_marks():
    sim,seen=scheduled_simulator()
    result=sim.advance(0)
    assert seen==[(1.,100.,0.)]  # canceled bid=100; remaining bid=98 and ask=102
    assert [s['event_type'] for s in sim.history.snapshots]==['CANCEL','MARKET_SELL']
    assert [s['time'] for s in sim.history.snapshots]==[.5,1.]
    assert result[2][0].price==98
    sim.order_book.assert_invariants()


def test_horizon_processes_cancels_without_drawing_future_order():
    sim,seen=scheduled_simulator()
    assert sim.advance(0,horizon=.75) is None
    assert seen==[]
    assert sim.history.snapshots[0]['time']==.5


def test_expired_filled_order_cancel_is_removed_from_tracking():
    sim=Simulator(seed=505)
    ask=Order.limit('sell',100,1,0)
    sim.order_book.add_limit_order(ask);sim._schedule_cancel(ask.order_id,.5)
    sim.order_book.process_market_order(Order.market('buy',1,.1))
    sim._process_cancellations_before(1)
    assert not sim._pending_cancels and not sim._cancel_heap
    assert not sim.history.snapshots


def test_live_gui_uses_same_advance_and_cancellation_path():
    # Execute only the actual worker, with a disposable emission sink and no web server.
    source=Path(__file__).resolve().parents[2]/'lob_gui.py'
    tree=ast.parse(source.read_text())
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='simulation_loop')
    sim,seen=scheduled_simulator();emitted=[]
    state={'lock':threading.Lock(),'running':True,'simulator':sim,'current_time':0.}
    def emit(name,payload):
        emitted.append((name,payload));state['running']=False
    namespace={'sim_state':state,'socketio':SimpleNamespace(emit=emit),'time':SimpleNamespace(sleep=lambda t:None)}
    exec(compile(ast.Module(body=[worker],type_ignores=[]),str(source),'exec'),namespace)
    namespace['simulation_loop']()
    assert seen==[(1.,100.,0.)]
    assert emitted[0][1]['events']==2
    assert emitted[0][1]['resting']==2


def test_deep_price_draws_stay_positive_and_tick_aligned():
    generator=EventGenerator(SimParams(depth_lambda=.001),seed=505)
    for kind in [EventType.LIMIT_BUY,EventType.LIMIT_SELL]:
        for _ in range(100):
            price=generator._sample_price(kind)
            assert price>0 and price/generator.params.tick_size==pytest.approx(round(price/generator.params.tick_size))


def test_narrow_initial_spread_does_not_cross_warmup_book():
    sim=Simulator(SimParams(initial_spread=.0001),seed=505)
    sim.run(0)
    sim.order_book.assert_invariants()
    assert sim.order_book.get_spread()>0


@pytest.mark.parametrize('horizon',[-1,np.inf,np.nan])
def test_invalid_horizons_rejected(horizon):
    with pytest.raises(ValueError):Simulator().run(horizon)


def test_matching_against_independent_list_book():
    rng=np.random.default_rng(505);book=OrderBook();resting=[];next_id=100000
    for sequence in range(500):
        if resting and rng.random()<.2:
            chosen=resting[int(rng.integers(len(resting)))];assert book.cancel_order(chosen['id'])
            resting.remove(chosen)
        else:
            next_id+=1;side='buy' if rng.random()<.5 else 'sell'
            is_market=rng.random()<.2;price=None if is_market else float(rng.integers(95,106));quantity=int(rng.integers(1,10))
            expected=[];remaining=quantity
            matches=sorted((r for r in resting if r['side']!=side),key=lambda r:((r['price'] if side=='buy' else -r['price']),r['seq']))
            for maker in matches:
                crosses=is_market or (price>=maker['price'] if side=='buy' else price<=maker['price'])
                if not crosses or remaining==0:break
                fill=min(remaining,maker['qty']);remaining-=fill;maker['qty']-=fill
                expected.append((maker['price'],fill,maker['id']))
                if maker['qty']==0:resting.remove(maker)
            order=Order(next_id,side,'market' if is_market else 'limit',price,quantity,float(sequence))
            actual=book.process_market_order(order) if is_market else book.add_limit_order(order)[0]
            assert [(t.price,t.quantity,t.sell_order_id if side=='buy' else t.buy_order_id) for t in actual]==expected
            if not is_market and remaining:
                resting.append(dict(id=next_id,side=side,price=price,qty=remaining,seq=sequence))
        assert {k:o.quantity for k,o in book._order_index.items()}=={r['id']:r['qty'] for r in resting}
        book.assert_invariants()
