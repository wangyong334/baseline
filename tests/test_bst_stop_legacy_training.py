"""Only one same-user, same-checkout, exact-output trainer may receive SIGTERM."""
from dataclasses import replace
import io
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

from tools.bst import stop_legacy_training as S


class StopLegacyTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.out = self.root/S.DEFAULT_OUT
        self.process = S.Process(123, 1000, 1000, 40001, "R", self.root,
            ("/env/bin/python", "-u", "tools/bst/train.py", "--config", "configs/bst_shared.json", "--out", S.DEFAULT_OUT))

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_scope_and_absolute_output_are_accepted(self):
        self.assertTrue(S.matches(self.process, self.root, self.out, 1000))
        absolute = replace(self.process, argv=("python3.8", "-X", "dev", "-u",
            str(self.root/"tools/bst/train.py"), "--out="+str(self.out)))
        self.assertTrue(S.matches(absolute, self.root, self.out, 1000))

    def test_other_user_checkout_task_and_python_command_are_excluded(self):
        cases = [replace(self.process, real_uid=99), replace(self.process, effective_uid=99),
                 replace(self.process, cwd=self.root/"other"),
                 replace(self.process, argv=("bash", "tools/final/gpu_pool.sh")),
                 replace(self.process, argv=("python", "tools/final/gpu_hold.py")),
                 replace(self.process, argv=("python", "-c", "tools/bst/train.py", "--out", S.DEFAULT_OUT)),
                 replace(self.process, argv=("python", "-m", "tools.bst.train", "--out", S.DEFAULT_OUT)),
                 replace(self.process, argv=("python", "other.py", "tools/bst/train.py", "--out", S.DEFAULT_OUT)),
                 replace(self.process, argv=self.process.argv[:-1]+("log/different",)),
                 replace(self.process, state="Z")]
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(S.matches(case, self.root, self.out, 1000))

    def test_ambiguous_out_and_multiple_matches_are_refused(self):
        for suffix in (("--out",), ("--out", "--config"), ("--out", S.DEFAULT_OUT, "--out="+S.DEFAULT_OUT)):
            process = replace(self.process, argv=("python", "tools/bst/train.py")+suffix)
            with self.assertRaises(S.StopRefused):
                S.matches(process, self.root, self.out, 1000)
        with mock.patch.object(S, "owned_pids", return_value=[123, 124]):
            with self.assertRaisesRegex(S.StopRefused, "Multiple"):
                S.find_matches(self.root, self.out, 1000,
                               reader=lambda pid: replace(self.process, pid=pid))

    def test_changed_identity_before_signal_is_never_killed(self):
        for changed in (replace(self.process, start_ticks=40002),
                        replace(self.process, argv=self.process.argv[:-1]+("log/other",)),
                        replace(self.process, effective_uid=99)):
            with mock.patch.object(S.os, "kill") as kill, \
                 mock.patch.object(S, "_pidfd_available", return_value=False):
                with self.assertRaisesRegex(S.StopRefused, "identity"):
                    S.stop_one(self.process, self.root, self.out, 1000, reader=lambda pid: changed)
                kill.assert_not_called()

    def test_exactly_one_sigterm_then_zombie_confirms_exit(self):
        reader = mock.Mock(side_effect=[replace(self.process, state="S"), replace(self.process, state="Z")])
        with mock.patch.object(S.os, "kill") as kill, \
             mock.patch.object(S, "_pidfd_available", return_value=False):
            result = S.stop_one(self.process, self.root, self.out, 1000, reader=reader)
        kill.assert_called_once_with(123, signal.SIGTERM)
        self.assertIn("EXIT_CONFIRMED", result)

    def test_already_exited_and_reused_pid_are_safe(self):
        with mock.patch.object(S.os, "kill") as kill, \
             mock.patch.object(S, "_pidfd_available", return_value=False):
            result = S.stop_one(self.process, self.root, self.out, 1000, reader=lambda pid: None)
            kill.assert_not_called()
            self.assertIn("ALREADY_EXITED", result)
            reader = mock.Mock(side_effect=[self.process, replace(self.process, start_ticks=90000)])
            result = S.stop_one(self.process, self.root, self.out, 1000, reader=reader)
            kill.assert_called_once_with(123, signal.SIGTERM)
            self.assertIn("another process", result)

    def test_timeout_never_escalates_to_sigkill(self):
        now = [0.]
        def sleep(seconds):
            now[0] += seconds
        with mock.patch.object(S.os, "kill") as kill, \
             mock.patch.object(S, "_pidfd_available", return_value=False):
            with self.assertRaisesRegex(S.StopRefused, "no SIGKILL"):
                S.stop_one(self.process, self.root, self.out, 1000, timeout=1,
                           reader=lambda pid: self.process, clock=lambda: now[0], sleeper=sleep)
            kill.assert_called_once_with(123, signal.SIGTERM)
        self.assertLessEqual(now[0], 1.)

    def test_pidfd_sender_uses_bound_handle_and_closes_it(self):
        with mock.patch.object(S, "_pidfd_available", return_value=True), \
             mock.patch.object(S.os, "pidfd_open", return_value=999, create=True) as opened, \
             mock.patch.object(S.signal, "pidfd_send_signal", create=True) as sent, \
             mock.patch.object(S.os, "close") as closed, mock.patch.object(S.os, "kill") as kill:
            self.assertTrue(S.send_term_checked(self.process, self.root, self.out, 1000,
                                               reader=lambda pid: self.process))
        opened.assert_called_once_with(123, 0)
        sent.assert_called_once_with(999, signal.SIGTERM, None, 0)
        closed.assert_called_once_with(999)
        kill.assert_not_called()

    def test_default_cli_is_read_only_and_no_match_is_not_whole_job_success(self):
        with mock.patch.object(S.sys, "platform", "linux"), mock.patch.object(S, "ROOT", self.root), \
             mock.patch.object(S.os, "geteuid", return_value=1000, create=True), \
             mock.patch.object(S, "find_matches", return_value=[self.process]), \
             mock.patch.object(S, "stop_one") as stop, mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(S.main([]), 0)
            stop.assert_not_called()
            self.assertIn("DRY_RUN", output.getvalue())
        with mock.patch.object(S.sys, "platform", "linux"), mock.patch.object(S, "ROOT", self.root), \
             mock.patch.object(S.os, "geteuid", return_value=1000, create=True), \
             mock.patch.object(S, "find_matches", return_value=[]), \
             mock.patch.object(S, "stop_one") as stop, mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(S.main(["--execute"]), 0)
            stop.assert_not_called()
            self.assertIn("does not prove", output.getvalue())

    def test_proc_stat_parentheses_and_output_scope(self):
        fields = ["S"] + ["0"] * 18 + ["70012"] + ["0"] * 4
        self.assertEqual(S.stat_identity("123 (python (worker)) " + " ".join(fields)), ("S", 70012))
        self.assertEqual(S.resolve_output(self.root, S.DEFAULT_OUT), self.out)
        for output in ("..", ".", str(self.root.parent/"outside")):
            with self.assertRaises(S.StopRefused):
                S.resolve_output(self.root, output)

    def test_proc_reader_preserves_nul_arguments_and_recognizes_zombie(self):
        proc = self.root/"proc"/"123"
        proc.mkdir(parents=True)
        (proc/"status").write_text("Name:\tpython\nUid:\t1000\t1000\t1000\t1000\n")
        fields = ["R"] + ["0"] * 18 + [str(self.process.start_ticks)] + ["0"] * 4
        (proc/"stat").write_text("123 (python (worker)) " + " ".join(fields))
        (proc/"cmdline").write_bytes(b"\0".join(word.encode() for word in self.process.argv) + b"\0")
        with mock.patch.object(S.os, "readlink", return_value=str(self.root)):
            self.assertEqual(S.read_process(123, self.root/"proc"), self.process)
        fields[0] = "Z"
        (proc/"stat").write_text("123 (python) " + " ".join(fields))
        with mock.patch.object(S.os, "readlink", side_effect=AssertionError("zombie cwd must not be read")):
            zombie = S.read_process(123, self.root/"proc")
        self.assertEqual(zombie.state, "Z")
        self.assertEqual(zombie.start_ticks, self.process.start_ticks)
        self.assertIsNone(S.read_process(999, self.root/"proc"))

    def test_execute_is_explicit_and_exit_confirmation_failure_is_clear(self):
        with mock.patch.object(S.sys, "platform", "linux"), mock.patch.object(S, "ROOT", self.root), \
             mock.patch.object(S.os, "geteuid", return_value=1000, create=True), \
             mock.patch.object(S, "find_matches", return_value=[self.process]), \
             mock.patch.object(S, "stop_one", return_value="EXIT_CONFIRMED") as stop, \
             mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(S.main(["--execute", "--timeout", "5"]), 0)
            stop.assert_called_once_with(self.process, self.root, self.out, 1000, 5.)
        with mock.patch.object(S.os, "kill") as kill, \
             mock.patch.object(S, "_pidfd_available", return_value=False):
            reader = mock.Mock(side_effect=[self.process, S.StopRefused("permission changed")])
            with self.assertRaisesRegex(S.StopRefused, "SIGTERM sent, but exit inspection failed"):
                S.stop_one(self.process, self.root, self.out, 1000, reader=reader)
            kill.assert_called_once_with(123, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
