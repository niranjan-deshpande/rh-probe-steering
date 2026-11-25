"""Unit tests for watchdog.py (decision logic, credential lookup, key hygiene).   python3 scripts/flag/night/test_watchdog.py"""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import watchdog as w  # noqa: E402

H = 3600.0
KEY = "rpa_SECRETSECRETSECRET"


def decide(now, start=0.0, last_active=None, deadline=100 * H, max_hours=9.0, idle_min=20.0, stop_file=False):
    return w.decide(now, start, now if last_active is None else last_active, deadline, max_hours, idle_min, stop_file)


class Decide(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(decide(1 * H), (False, "ok"))

    def test_deadline(self):
        self.assertEqual(decide(5 * H - 1, deadline=5 * H), (False, "ok"))
        self.assertEqual(decide(5 * H, deadline=5 * H), (True, "deadline"))

    def test_max_hours(self):
        self.assertEqual(decide(9 * H - 1), (False, "ok"))
        self.assertEqual(decide(9 * H), (True, "max_hours"))
        self.assertEqual(decide(10 * H, start=2 * H), (False, "ok"))      # counted from the pod's start

    def test_idle(self):
        self.assertEqual(decide(2 * H, last_active=2 * H - 19.9 * 60), (False, "ok"))
        self.assertEqual(decide(2 * H, last_active=2 * H - 20 * 60), (True, "idle"))

    def test_stop_file_and_priority(self):
        self.assertEqual(decide(1 * H, stop_file=True), (True, "stop_file"))
        self.assertEqual(decide(50 * H, deadline=5 * H, last_active=0.0)[1], "deadline")
        self.assertEqual(decide(50 * H, last_active=0.0)[1], "max_hours")

    def test_limits_can_be_disabled(self):
        self.assertEqual(w.decide(1e9, 0.0, 0.0, None, None, None, False), (False, "ok"))


class LastActive(unittest.TestCase):
    def test_all_quiet_keeps_the_clock(self):
        self.assertEqual(w.update_last_active(100.0, 500.0, False, 0.0, None), 100.0)
        self.assertEqual(w.update_last_active(100.0, 500.0, False, 4.9, 90.0), 100.0)
        self.assertEqual(w.update_last_active(100.0, 500.0, False, None, None), 100.0)   # unreadable GPU is not activity

    def test_each_signal_restarts_the_clock(self):
        self.assertEqual(w.update_last_active(100.0, 500.0, True, 0.0, None), 500.0)
        self.assertEqual(w.update_last_active(100.0, 500.0, False, 5.0, None), 500.0)
        self.assertEqual(w.update_last_active(100.0, 500.0, False, 0.0, 450.0), 450.0)

    def test_future_mtime_cannot_buy_time(self):
        self.assertEqual(w.update_last_active(100.0, 500.0, False, 0.0, 9e9), 500.0)

    def test_idle_sequence(self):
        """A stage runs for 30 min, then nothing: the stop comes 20 min after the last sign of life."""
        last, t, reasons = 0.0, 0.0, []
        for minute in range(0, 60):
            t = minute * 60.0
            busy = minute < 30
            last = w.update_last_active(last, t, busy, 80.0 if busy else 0.0, t if busy else None)
            reasons.append(w.decide(t, 0.0, last, 100 * H, 9.0, 20.0, False))
        self.assertTrue(all(not s for s, _ in reasons[:49]))
        self.assertEqual(reasons[49], (True, "idle"))        # last active at minute 29, so minute 49


class Probes(unittest.TestCase):
    def test_credentials_prefer_environment_then_pid1(self):
        self.assertEqual(w.credentials({"RUNPOD_API_KEY": "k1", "RUNPOD_POD_ID": "p1"}, {"RUNPOD_API_KEY": "k2"}), ("k1", "p1"))
        self.assertEqual(w.credentials({}, {"RUNPOD_API_KEY": "k2", "RUNPOD_POD_ID": "p2"}), ("k2", "p2"))
        self.assertEqual(w.credentials({}, {}), ("", ""))

    def test_read_pid1_env(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"A=1\0RUNPOD_POD_ID=abc\0RUNPOD_API_KEY=x=y\0junk\0")
        try:
            self.assertEqual(w.read_pid1_env(f.name), {"A": "1", "RUNPOD_POD_ID": "abc", "RUNPOD_API_KEY": "x=y"})
            self.assertEqual(w.read_pid1_env(f.name + ".missing"), {})
        finally:
            os.unlink(f.name)

    def test_newest_mtime(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(w.newest_mtime(d))
            self.assertIsNone(w.newest_mtime(os.path.join(d, "missing")))
            os.makedirs(os.path.join(d, "sub"))
            for name, m in (("a.log", 1000.0), ("sub/b.log", 2000.0)):
                p = os.path.join(d, name)
                open(p, "w").close()
                os.utime(p, (m, m))
            self.assertEqual(w.newest_mtime(d), 2000.0)

    def test_stage_alive_from_running_dir(self):
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "running")
            os.makedirs(run)
            empty_proc = os.path.join(d, "proc")
            os.makedirs(empty_proc)
            self.assertFalse(w.stage_alive(d, proc=empty_proc))
            open(os.path.join(run, "server.pid"), "w").write(str(os.getpgid(0)))     # the server is not a stage
            open(os.path.join(run, "H1.timeout"), "w").write("timeout")
            open(os.path.join(run, "junk"), "w").write("not a number")
            self.assertFalse(w.stage_alive(d, proc=empty_proc))
            open(os.path.join(run, "DEAD"), "w").write("2147483646")
            self.assertFalse(w.stage_alive(d, proc=empty_proc))
            open(os.path.join(run, "H1"), "w").write(str(os.getpgid(0)))             # our own group: alive
            self.assertTrue(w.stage_alive(d, proc=empty_proc))

    def test_stage_alive_from_command_lines(self):
        with tempfile.TemporaryDirectory() as d:
            proc = os.path.join(d, "proc")
            for pid, cmd in (("11", b"python3\0watchdog.py\0--note\0stage.sh"), ("12", b"sleep\0100")):
                os.makedirs(os.path.join(proc, pid))
                open(os.path.join(proc, pid, "cmdline"), "wb").write(cmd)
            self.assertFalse(w.stage_alive(d, proc=proc))
            os.makedirs(os.path.join(proc, "13"))
            open(os.path.join(proc, "13", "cmdline"), "wb").write(b"bash\0/workspace/rlp/scripts/flag/night/stage.sh\0H1\0")
            self.assertTrue(w.stage_alive(d, proc=proc))


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.text.encode()


class Stop(unittest.TestCase):
    def test_request_shape_and_success(self):
        seen = []

        def opener(req, timeout):
            seen.append(req)
            return FakeResponse('{"data":{"podStop":{"id":"pod123"}}}')
        ok, msg = w.stop_pod(KEY, "pod123", opener)
        self.assertTrue(ok)
        self.assertNotIn(KEY, msg)
        r = seen[0]
        self.assertEqual(len(seen), 1)
        self.assertEqual(r.full_url, "https://api.runpod.io/graphql")           # the key is not in the URL
        self.assertEqual(r.get_header("Authorization"), "Bearer " + KEY)
        self.assertEqual(json.loads(r.data), {"query": 'mutation{podStop(input:{podId:"pod123"}){id}}'})
        self.assertEqual(r.get_method(), "POST")

    def test_falls_back_to_query_parameter_and_never_leaks(self):
        seen = []

        def opener(req, timeout):
            seen.append(req.full_url)
            if len(seen) == 1:
                raise urllib.error.HTTPError(req.full_url, 401, "no", {}, None)
            raise OSError("cannot reach %s" % req.full_url)                     # an error text that contains the key
        ok, msg = w.stop_pod(KEY, "pod123", opener)
        self.assertFalse(ok)
        self.assertEqual(len(seen), 2)
        self.assertTrue(seen[1].endswith("?api_key=" + KEY))
        self.assertNotIn(KEY, msg)
        self.assertIn("<key>", msg)

    def test_graphql_error_is_not_success(self):
        ok, msg = w.stop_pod(KEY, "pod123", lambda req, timeout: FakeResponse('{"errors":[{"message":"denied %s"}]}' % KEY))
        self.assertFalse(ok)
        self.assertNotIn(KEY, msg)

    def test_missing_or_odd_credentials(self):
        called = []
        self.assertFalse(w.stop_pod("", "pod123", lambda *a, **k: called.append(1))[0])
        self.assertFalse(w.stop_pod(KEY, "", lambda *a, **k: called.append(1))[0])
        self.assertFalse(w.stop_pod(KEY, 'x"){id}}', lambda *a, **k: called.append(1))[0])
        self.assertEqual(called, [])


class DryRun(unittest.TestCase):
    def run_main(self, root, *args):
        buf = io.StringIO()
        old = dict(os.environ)
        os.environ.update(RUNPOD_API_KEY=KEY, RUNPOD_POD_ID="pod123")
        real = w.stop_pod
        w.stop_pod = lambda *a, **k: self.fail("dry run called the API")
        try:
            with contextlib.redirect_stdout(buf):
                rc = w.main(["--root", root, "--dry-run", "--once", *args])
        finally:
            w.stop_pod = real
            os.environ.clear()
            os.environ.update(old)
        return rc, buf.getvalue()

    def test_dry_run_prints_and_does_not_stop(self):
        with tempfile.TemporaryDirectory() as d:
            now = time.time()
            rc, out = self.run_main(d, "--deadline", str(now + 3600), "--max-hours", "9", "--start-epoch", str(now - 60))
            self.assertEqual(rc, 0)
            self.assertIn("reason=ok", out)
            self.assertNotIn("DRY_RUN would stop", out)
            hb = json.load(open(os.path.join(d, "watchdog", "heartbeat.json")))
            self.assertEqual((hb["stop"], hb["reason"], hb["pod"], hb["dry_run"]), (False, "ok", "pod123", True))

            for extra, reason in ((["--deadline", str(now - 1), "--max-hours", "9"], "deadline"),
                                  (["--deadline", str(now + 3600), "--max-hours", "0.01"], "max_hours")):
                rc, out = self.run_main(d, "--start-epoch", str(now - 3600), *extra)
                self.assertEqual(rc, 0)
                self.assertIn("DRY_RUN would stop pod pod123 now (reason: %s)" % reason, out)
            open(os.path.join(d, "STOP"), "w").close()
            rc, out = self.run_main(d, "--deadline", str(now + 3600), "--max-hours", "9", "--start-epoch", str(now))
            self.assertIn("(reason: stop_file)", out)
            self.assertFalse(os.path.exists(os.path.join(d, "watchdog", "stopped")))
            text = out + open(os.path.join(d, "watchdog", "watchdog.log")).read() + open(os.path.join(d, "watchdog", "heartbeat.json")).read()
            self.assertNotIn(KEY, text)
            self.assertIn("key=present", text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
