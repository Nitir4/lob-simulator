"""
tests/test_order_book.py  (PHASE 4 REVISION)
=============================================
Added tests for:
  - get_imbalance(): correct formula and edge cases
  - add_limit_order() returns correct Tuple
  - Positive quantity invariant in assert_invariants()
  - Total volume consistency after partial fill
All original tests preserved.
"""

import unittest
from lob_simulator.core.order import Order
from lob_simulator.core.order_book import OrderBook


def make_book_with_spread() -> OrderBook:
    book = OrderBook()
    book.add_limit_order(Order.limit("buy",  99.0,  10, timestamp=1.0))
    book.add_limit_order(Order.limit("sell", 101.0, 10, timestamp=1.1))
    book.assert_invariants()
    return book


# ── Original tests ────────────────────────────────────────────────────────

class TestEmptyBook(unittest.TestCase):
    def setUp(self): self.book = OrderBook()
    def test_best_bid_is_none(self): self.assertIsNone(self.book.get_best_bid())
    def test_best_ask_is_none(self): self.assertIsNone(self.book.get_best_ask())
    def test_mid_price_is_none(self): self.assertIsNone(self.book.get_mid_price())
    def test_spread_is_none(self): self.assertIsNone(self.book.get_spread())
    def test_invariants_on_empty_book(self): self.book.assert_invariants()

    def test_market_order_on_empty_book(self):
        trades = self.book.process_market_order(
            Order.market("buy", 10, timestamp=1.0))
        self.assertEqual(trades, [])
        self.book.assert_invariants()

    def test_imbalance_on_empty_book(self):
        """Imbalance must be 0.0 when book is empty — no signal."""
        self.assertAlmostEqual(self.book.get_imbalance(), 0.0)


class TestLimitOrderResting(unittest.TestCase):
    def setUp(self): self.book = OrderBook()

    def test_limit_buy_rests_in_bids(self):
        _, rested = self.book.add_limit_order(
            Order.limit("buy", 99.0, 5, timestamp=1.0))
        self.assertIsNotNone(rested)   # should have rested
        self.assertEqual(self.book.get_best_bid(), 99.0)
        self.book.assert_invariants()

    def test_fully_filled_order_not_rested(self):
        """add_limit_order should return rested=None when fully filled."""
        self.book.add_limit_order(Order.limit("sell", 100.0, 5, timestamp=1.0))
        _, rested = self.book.add_limit_order(
            Order.limit("buy", 100.0, 5, timestamp=2.0))
        self.assertIsNone(rested, "Fully filled order must return rested=None")
        self.book.assert_invariants()

    def test_spread_computed_correctly(self):
        self.book.add_limit_order(Order.limit("buy",  99.0,  5, timestamp=1.0))
        self.book.add_limit_order(Order.limit("sell", 101.0, 5, timestamp=1.1))
        self.assertAlmostEqual(self.book.get_spread(), 2.0)
        self.assertAlmostEqual(self.book.get_mid_price(), 100.0)
        self.book.assert_invariants()


class TestFullFill(unittest.TestCase):
    def setUp(self):
        self.book = OrderBook()
        self.book.add_limit_order(Order.limit("sell", 100.0, 10, timestamp=1.0))

    def test_crossing_limit_buy_fully_fills(self):
        _, rested = self.book.add_limit_order(
            Order.limit("buy", 101.0, 10, timestamp=2.0))
        self.assertIsNone(rested)   # fully filled
        self.assertIsNone(self.book.get_best_ask())
        self.book.assert_invariants()

    def test_execution_at_resting_price(self):
        """Formal rule: trade_price = resting_order.price always."""
        trades, _ = self.book.add_limit_order(
            Order.limit("buy", 101.0, 10, timestamp=2.0))
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0].price, 100.0)


class TestPartialFill(unittest.TestCase):
    def setUp(self):
        self.book = OrderBook()
        self.book.add_limit_order(Order.limit("sell", 100.0, 5, timestamp=1.0))

    def test_partial_fill_rests_remainder(self):
        trades, rested = self.book.add_limit_order(
            Order.limit("buy", 100.0, 10, timestamp=2.0))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].quantity, 5)
        self.assertIsNotNone(rested)
        self.assertEqual(rested.quantity, 5)
        self.book.assert_invariants()

    def test_fill_quantity_invariant(self):
        """
        Total volume invariant: after partial fill,
        filled_qty + rested_qty == original_order_qty.
        """
        original_qty = 8
        order = Order.limit("buy", 100.0, original_qty, timestamp=2.0)
        trades, rested = self.book.add_limit_order(order)
        filled_qty = sum(tr.quantity for tr in trades)
        remaining  = rested.quantity if rested else 0
        self.assertEqual(filled_qty + remaining, original_qty)


class TestFIFOPriority(unittest.TestCase):
    def setUp(self):
        self.book = OrderBook()
        self.early = Order.limit("sell", 100.0, 5, timestamp=1.0)
        self.late  = Order.limit("sell", 100.0, 5, timestamp=2.0)
        self.book.add_limit_order(self.early)
        self.book.add_limit_order(self.late)

    def test_earlier_order_fills_first(self):
        trades = self.book.process_market_order(Order.market("buy", 5, 3.0))
        self.assertEqual(trades[0].sell_order_id, self.early.order_id)
        self.book.assert_invariants()

    def test_both_fill_in_order(self):
        trades = self.book.process_market_order(Order.market("buy", 10, 3.0))
        self.assertEqual(trades[0].sell_order_id, self.early.order_id)
        self.assertEqual(trades[1].sell_order_id, self.late.order_id)


class TestMultiLevelSweep(unittest.TestCase):
    def setUp(self):
        self.book = OrderBook()
        for px in [100.0, 100.5, 101.0]:
            self.book.add_limit_order(Order.limit("sell", px, 3, timestamp=1.0))

    def test_sweeps_two_levels(self):
        trades = self.book.process_market_order(Order.market("buy", 6, 2.0))
        self.assertEqual(len(trades), 2)
        self.assertAlmostEqual(trades[0].price, 100.0)
        self.assertAlmostEqual(trades[1].price, 100.5)
        self.assertAlmostEqual(self.book.get_best_ask(), 101.0)
        self.book.assert_invariants()


class TestCancelOrder(unittest.TestCase):
    def setUp(self): self.book = OrderBook()

    def test_cancel_resting_bid(self):
        order = Order.limit("buy", 99.0, 10, timestamp=1.0)
        self.book.add_limit_order(order)
        self.assertTrue(self.book.cancel_order(order.order_id))
        self.assertIsNone(self.book.get_best_bid())
        self.book.assert_invariants()

    def test_cancel_nonexistent_returns_false(self):
        self.assertFalse(self.book.cancel_order(999999))

    def test_cancel_filled_order_returns_false(self):
        resting = Order.limit("sell", 100.0, 5, timestamp=1.0)
        self.book.add_limit_order(resting)
        self.book.process_market_order(Order.market("buy", 5, 2.0))
        self.assertFalse(self.book.cancel_order(resting.order_id))


# ── New Phase 4 tests ────────────────────────────────────────────────────

class TestQueueImbalance(unittest.TestCase):
    """
    Tests for get_imbalance() implementing the Cont-Larrard formula:
      I = (V_bid - V_ask) / (V_bid + V_ask)

    WHY test this carefully?
      The imbalance is now used to modulate market order rates.
      An incorrect imbalance calculation would systematically bias the
      price dynamics in the wrong direction — a subtle but critical bug.
    """

    def test_empty_book_imbalance_is_zero(self):
        book = OrderBook()
        self.assertAlmostEqual(book.get_imbalance(), 0.0)

    def test_balanced_book_imbalance_is_zero(self):
        """Equal volumes at best bid and ask → I = 0."""
        book = OrderBook()
        book.add_limit_order(Order.limit("buy",  99.0, 5, timestamp=1.0))
        book.add_limit_order(Order.limit("sell", 101.0, 5, timestamp=1.1))
        self.assertAlmostEqual(book.get_imbalance(), 0.0)

    def test_heavy_bid_side_positive_imbalance(self):
        """More volume at bid → I > 0 → upward price pressure."""
        book = OrderBook()
        book.add_limit_order(Order.limit("buy",  99.0,  10, timestamp=1.0))
        book.add_limit_order(Order.limit("sell", 101.0, 2,  timestamp=1.1))
        I = book.get_imbalance()
        self.assertGreater(I, 0.0)
        expected = (10 - 2) / (10 + 2)
        self.assertAlmostEqual(I, expected)

    def test_heavy_ask_side_negative_imbalance(self):
        """More volume at ask → I < 0 → downward price pressure."""
        book = OrderBook()
        book.add_limit_order(Order.limit("buy",  99.0,  2,  timestamp=1.0))
        book.add_limit_order(Order.limit("sell", 101.0, 10, timestamp=1.1))
        I = book.get_imbalance()
        self.assertLess(I, 0.0)
        expected = (2 - 10) / (2 + 10)
        self.assertAlmostEqual(I, expected)

    def test_one_sided_book_imbalance(self):
        """
        Only bids, no asks: V_ask=0, so I = (V_bid - 0) / (V_bid + 0) = 1.0.
        This is mathematically correct — a book with only bids has maximum
        upward pressure. The get_imbalance() returns 0.0 only when BOTH
        sides are empty (no signal). With a one-sided book, the signal
        is maximum (extreme imbalance).

        Corrected: the test was wrong, not the formula.
        """
        book = OrderBook()
        book.add_limit_order(Order.limit("buy", 99.0, 10, timestamp=1.0))
        # V_bid=10, V_ask=0 → I = (10-0)/(10+0) = 1.0
        self.assertAlmostEqual(book.get_imbalance(), 1.0)


    def test_imbalance_uses_best_level_only(self):
        """
        Deep book levels should NOT affect imbalance.
        Only the best bid and best ask volumes matter.
        """
        book = OrderBook()
        # Best bid level: 10 units
        book.add_limit_order(Order.limit("buy",  99.0, 10, timestamp=1.0))
        # Deep bid level: 100 units (should be ignored by get_imbalance)
        book.add_limit_order(Order.limit("buy",  98.0, 100, timestamp=1.1))
        # Best ask level: 10 units
        book.add_limit_order(Order.limit("sell", 101.0, 10, timestamp=1.2))
        # Expected: I = (10-10)/(10+10) = 0
        self.assertAlmostEqual(book.get_imbalance(), 0.0)


class TestInsideSpreadOrders(unittest.TestCase):
    """
    Tests verifying that inside-spread orders tighten the spread.
    These test the ORDER BOOK's response to such orders —
    the event generator tests verify these orders are actually generated.
    """

    def setUp(self):
        self.book = make_book_with_spread()  # bid=99, ask=101, spread=2

    def test_inside_spread_buy_tightens(self):
        """Limit buy at 100 (inside 99-101 spread) → new best bid, spread=1."""
        _, rested = self.book.add_limit_order(
            Order.limit("buy", 100.0, 5, timestamp=2.0))
        self.assertIsNotNone(rested)
        self.assertAlmostEqual(self.book.get_best_bid(), 100.0)
        self.assertAlmostEqual(self.book.get_spread(), 1.0)
        self.book.assert_invariants()

    def test_crossing_limit_buy_fills(self):
        """Limit buy at 101 crosses the resting ask → fills immediately."""
        trades, rested = self.book.add_limit_order(
            Order.limit("buy", 101.0, 5, timestamp=2.0))
        self.assertGreater(len(trades), 0)
        self.book.assert_invariants()


class TestEmptyLevelCleanup(unittest.TestCase):
    def test_empty_ask_level_removed(self):
        book = OrderBook()
        book.add_limit_order(Order.limit("sell", 100.0, 3, timestamp=1.0))
        book.process_market_order(Order.market("buy", 3, 2.0))
        self.assertNotIn(100.0, book._asks)
        book.assert_invariants()

    def test_empty_bid_level_removed_after_cancel(self):
        book = OrderBook()
        o = Order.limit("buy", 99.0, 5, timestamp=1.0)
        book.add_limit_order(o)
        book.cancel_order(o.order_id)
        self.assertNotIn(99.0, book._bids)
        book.assert_invariants()


class TestInvariantsUnderStress(unittest.TestCase):
    def test_invariants_hold_after_200_random_ops(self):
        import random
        rng = random.Random(42)
        book = OrderBook()
        resting_ids = []

        for i in range(200):
            op, t = rng.randint(0, 3), float(i)

            if op == 0:
                px = round(rng.uniform(98.0, 100.5), 2)
                _, rested = book.add_limit_order(
                    Order.limit("buy", px, rng.randint(1, 5), t))
                if rested: resting_ids.append(rested.order_id)
            elif op == 1:
                px = round(rng.uniform(99.5, 102.0), 2)
                _, rested = book.add_limit_order(
                    Order.limit("sell", px, rng.randint(1, 5), t))
                if rested: resting_ids.append(rested.order_id)
            elif op == 2:
                side = rng.choice(["buy", "sell"])
                book.process_market_order(Order.market(side, rng.randint(1, 3), t))
            else:
                if resting_ids:
                    oid = rng.choice(resting_ids)
                    book.cancel_order(oid)
                    resting_ids = [x for x in resting_ids if x != oid]

            book.assert_invariants()


if __name__ == "__main__":
    unittest.main()
