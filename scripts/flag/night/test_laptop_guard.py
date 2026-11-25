"""Unit tests for laptop_guard.py.   python3 scripts/flag/night/test_laptop_guard.py"""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laptop_guard as g  # noqa: E402

DEADLINE = 1795627800          # 2026-11-25 12:30 pm ET
T0 = DEADLINE - 12 * 3600      # 12:30 am ET


def verdicts(pods, now, budget=38.0, **kw):
    d, s = g.decide(pods, now, DEADLINE, budget, **kw)
    return {p.name: (v, r) for p, v, r in d}, s


class Guard(unittest.TestCase):
    def test_parse_time(self):
        self.assertEqual(g.parse_time("1795627800"), 1795627800.0)
        self.assertEqual(g.parse_time("2026-11-25T12:30:00-05:00"), 1795627800.0)
        self.assertEqual(g.parse_time("2026-11-25T17:30:00Z"), 1795627800.0)
        with self.assertRaises(ValueError):
            g.parse_time("2026-11-25T12:30:00")           # no offset: refuse to guess

    def test_parse_pod(self):
        p = g.parse_pod("A,abc123,1795600000,2.19")
        self.assertEqual((p.name, p.id, p.start, p.price, p.max_hours, p.stopped_at), ("A", "abc123", 1795600000.0, 2.19, None, None))
        p = g.parse_pod("B,xyz,2026-11-25T01:00:00-05:00,2.19,9")
        self.assertEqual(p.max_hours, 9.0)
        p = g.parse_pod("B,xyz,1795600000,2.19,,1795603600")
        self.assertIsNone(p.max_hours)
        self.assertEqual(p.stopped_at, 1795603600.0)
        for bad in ("A,abc,1795600000", "A,,1795600000,2.19", "A,abc,1795600000,0", "A,abc,1,2,3,4,5"):
            with self.assertRaises(ValueError):
                g.parse_pod(bad)

    def test_spend_arithmetic(self):
        a, b = g.Pod("A", "a", T0, 2.19), g.Pod("B", "b", T0 + 1800, 2.19)
        v, s = verdicts([a, b], T0 + 4 * 3600)
        self.assertAlmostEqual(s["total"], 2.19 * 4 + 2.19 * 3.5)
        self.assertAlmostEqual(s["rate"], 4.38)
        self.assertEqual(v, {"A": ("OK", "running"), "B": ("OK", "running")})
        self.assertAlmostEqual(s["budget_out_at"], T0 + 4 * 3600 + (38 - s["total"]) / 4.38 * 3600)

    def test_budget_stops_all_running_with_lookahead(self):
        a, b = g.Pod("A", "a", T0, 2.19), g.Pod("B", "b", T0, 2.19)
        hit = 38 / 4.38 * 3600                               # seconds until $38 at two pods
        v, _ = verdicts([a, b], T0 + hit - 6 * 60)           # 6 min before, 5 min look-ahead: still fine
        self.assertEqual({x[0] for x in v.values()}, {"OK"})
        v, _ = verdicts([a, b], T0 + hit - 4 * 60)           # would cross before the next check
        self.assertEqual(v, {"A": ("STOP", "budget"), "B": ("STOP", "budget")})
        v, _ = verdicts([a, b], T0 + hit - 4 * 60, poll_min=0)
        self.assertEqual({x[0] for x in v.values()}, {"OK"})
        v, _ = verdicts([a, b], T0 + hit - 4 * 60, poll_min=0, spent=1.0)
        self.assertEqual({x[1] for x in v.values()}, {"budget"})

    def test_deadline(self):
        a = g.Pod("A", "a", DEADLINE - 3600, 2.19)
        self.assertEqual(verdicts([a], DEADLINE - 1)[0]["A"], ("OK", "running"))
        self.assertEqual(verdicts([a], DEADLINE)[0]["A"], ("STOP", "deadline"))
        self.assertEqual(verdicts([a], DEADLINE + 999)[0]["A"], ("STOP", "deadline"))

    def test_stopped_pod_counts_but_is_not_stopped_again(self):
        a = g.Pod("A", "a", T0, 2.19, stopped_at=T0 + 2 * 3600)
        b = g.Pod("B", "b", T0, 2.19)
        v, s = verdicts([a, b], T0 + 5 * 3600)
        self.assertAlmostEqual(s["total"], 2.19 * 2 + 2.19 * 5)
        self.assertAlmostEqual(s["rate"], 2.19)
        self.assertEqual(v["A"], ("OK", "not running"))
        v, _ = verdicts([a, b], DEADLINE + 1)
        self.assertEqual(v, {"A": ("OK", "not running"), "B": ("STOP", "deadline")})
        v, s = verdicts([a], DEADLINE + 1)                   # nothing running: nothing to stop, even over budget
        self.assertEqual(v["A"][0], "OK")
        v, _ = verdicts([a], T0 + 5 * 3600, budget=1.0)
        self.assertEqual(v["A"][0], "OK")

    def test_max_hours_is_per_pod(self):
        a, b = g.Pod("A", "a", T0, 2.19), g.Pod("B", "b", T0, 2.19, max_hours=3)
        v, _ = verdicts([a, b], T0 + 2.9 * 3600)
        self.assertEqual(v, {"A": ("OK", "running"), "B": ("OK", "running")})
        v, _ = verdicts([a, b], T0 + 2.95 * 3600)            # 3 min before the limit, 5 min look-ahead
        self.assertEqual(v, {"A": ("OK", "running"), "B": ("STOP", "max_hours")})

    def test_pod_not_started_yet(self):
        a = g.Pod("A", "a", T0 + 3600, 2.19)
        v, s = verdicts([a], T0)
        self.assertEqual(s["total"], 0.0)
        self.assertEqual(v["A"], ("OK", "not running"))

    def test_report_lines_and_exit_code(self):
        a, b = g.Pod("A", "podaaa", T0, 2.19), g.Pod("B", "podbbb", T0, 2.19, stopped_at=T0 + 3600)
        buf = io.StringIO()
        self.assertTrue(g.report([a, b], DEADLINE, DEADLINE, 38.0, 0.0, 5.0, out=buf))
        lines = buf.getvalue().splitlines()
        self.assertEqual(lines[0], "STOP podaaa name=A reason=deadline")
        self.assertTrue(lines[1].startswith("OK podbbb name=B not_running"))
        self.assertIn("deadline 12:30 pm ET Nov 25", lines[2])
        self.assertEqual(g.et(T0), "12:30 am ET Nov 25")
        argv = ["--pod", "A,podaaa,%d,2.19" % T0, "--deadline", str(DEADLINE), "--budget", "38"]
        old, sys.stdout = sys.stdout, io.StringIO()
        try:
            self.assertEqual(g.main(argv + ["--now", str(T0 + 3600)]), 0)
            self.assertEqual(g.main(argv + ["--now", "2026-11-25T12:30:00-05:00"]), 3)
        finally:
            sys.stdout = old


if __name__ == "__main__":
    unittest.main(verbosity=1)
