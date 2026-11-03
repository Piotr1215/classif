import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("judge_gpu_guard", ROOT / "judge.py")
judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(judge)


class GpuGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.lock_path = Path(self.tmp.name) / "gpu.lock"

    def test_local_temperature_below_limit_acquires_host_wide_lock(self):
        with mock.patch.object(judge, "GPU_LOCK_PATH", str(self.lock_path)), \
             mock.patch.object(judge, "local_gpu_temperature", return_value=79):
            lock = judge.acquire_local_inference(1)
        self.assertIsNotNone(lock)
        lock.close()

    def test_hot_or_unreadable_gpu_times_out_without_acquiring(self):
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        with mock.patch.object(judge, "GPU_LOCK_PATH", str(self.lock_path)), \
             mock.patch.object(judge, "local_gpu_temperature", return_value=80), \
             mock.patch.object(judge.time, "monotonic", side_effect=lambda: now[0]), \
             mock.patch.object(judge.time, "sleep", side_effect=sleep):
            self.assertIsNone(judge.acquire_local_inference(1))
        now[0] = 0.0
        with mock.patch.object(judge, "GPU_LOCK_PATH", str(self.lock_path)), \
             mock.patch.object(judge, "local_gpu_temperature", return_value=None), \
             mock.patch.object(judge.time, "monotonic", side_effect=lambda: now[0]), \
             mock.patch.object(judge.time, "sleep", side_effect=sleep):
            self.assertIsNone(judge.acquire_local_inference(1))

    def test_remote_host_does_not_use_laptop_gpu_guard(self):
        interfaces = mock.Mock(stdout='[ {"addr_info": [{"local": "192.0.2.10"}]} ]')
        remote = mock.Mock(stdout="198.51.100.5 STREAM remote.example\n")
        with mock.patch.object(judge.subprocess, "run", side_effect=[interfaces, remote]):
            self.assertFalse(judge.is_local_gpu_host("remote.example:11434"))

    def test_loopback_aliases_use_laptop_gpu_guard(self):
        self.assertTrue(judge.is_local_gpu_host("localhost:11434"))
        self.assertTrue(judge.is_local_gpu_host("127.0.0.1:11434"))

    def test_non_loopback_interface_address_is_local(self):
        ip_output = ('[{"ifname":"eth0","addr_info":[{"local":"192.0.2.10"}]},'
                     '{"ifname":"tailscale0","addr_info":[{"local":"100.64.10.20","family":"inet"}]}]')
        with mock.patch.object(judge.subprocess, "run", return_value=mock.Mock(stdout=ip_output)):
            self.assertTrue(judge.is_local_gpu_host("100.64.10.20:11434"))

    def test_hostname_resolving_to_local_interface_is_local(self):
        ip_output = '[{"ifname":"tailscale0","addr_info":[{"local":"100.64.10.20"}]}]'
        resolve_output = mock.Mock(stdout="100.64.10.20 STREAM local.example\n")
        with mock.patch.object(judge.subprocess, "run", side_effect=[mock.Mock(stdout=ip_output), resolve_output]):
            self.assertTrue(judge.is_local_gpu_host("local.example:11434"))

    def test_dns_resolution_is_bounded_and_fails_closed(self):
        ip_output = '[{"ifname":"tailscale0","addr_info":[{"local":"100.91.4.22"}]}]'
        timeout = judge.subprocess.TimeoutExpired("getent", 1)
        with mock.patch.object(judge.subprocess, "run", side_effect=[mock.Mock(stdout=ip_output), timeout]) as run:
            self.assertTrue(judge.is_local_gpu_host("slow-dns.example:11434"))
        self.assertTrue(run.call_args_list[-1].kwargs["timeout"] <= 1.0)

    def test_missing_interface_tool_fails_closed_for_unknown_alias(self):
        with mock.patch.object(judge.subprocess, "run", side_effect=FileNotFoundError("ip")):
            self.assertTrue(judge.is_local_gpu_host("local.example:11434"))

    def test_empty_interface_list_fails_closed(self):
        with mock.patch.object(judge.subprocess, "run", return_value=mock.Mock(stdout="[]")):
            self.assertTrue(judge.is_local_gpu_host("192.0.2.10:11434"))

    def test_cpu_only_host_does_not_require_nvidia_smi(self):
        with mock.patch.object(judge.shutil, "which", return_value=None), \
             mock.patch.object(judge.os.path, "exists", return_value=False):
            self.assertEqual(judge.local_gpu_temperature(), 0)

    def test_nvidia_driver_without_sensor_fails_closed(self):
        with mock.patch.object(judge.shutil, "which", return_value=None), \
             mock.patch.object(judge.os.path, "exists", return_value=True):
            self.assertIsNone(judge.local_gpu_temperature())

    def test_lock_test_uses_private_temp_path(self):
        self.assertNotEqual(str(self.lock_path), judge.GPU_LOCK_PATH)

    def test_second_local_inference_fails_immediately_while_lock_is_held(self):
        first = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            import fcntl
            fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
            started = time.monotonic()
            with mock.patch.object(judge, "GPU_LOCK_PATH", str(self.lock_path)):
                self.assertIsNone(judge.acquire_local_inference(10))
            self.assertLess(time.monotonic() - started, 0.2)
        finally:
            os.close(first)

    def test_hot_local_chat_and_embedding_skip_http_requests(self):
        with mock.patch.object(judge, "is_local_gpu_host", return_value=True), \
             mock.patch.object(judge, "acquire_local_inference", return_value=None), \
             mock.patch.object(judge.urllib.request, "urlopen") as urlopen:
            chat = judge.chat("localhost:11434", "model", "sys", "user", {})
            embed = judge.embed(["input"], host="localhost:11434")
        self.assertIn("thermal limit", chat["unscored"])
        self.assertIn("thermal limit", embed["unscored"])
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
