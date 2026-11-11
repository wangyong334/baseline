"""Driver NVML board energy: hardware counter preferred, power integral fallback.

Not process energy, CPU energy, wall-plug energy, or neuromorphic SOP energy.
NVML power on recent GPUs can be a one-second average, even when polled faster.
https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html
"""
import ctypes as C
import os
import threading
import time


def integrate_power(samples):
    valid = [(s["seconds"], s["watts"]) for s in samples if s["watts"] is not None]
    if len(valid) != len(samples) or len(valid) < 2:
        return None
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(valid,valid[1:]))


class NVMLBoard:
    def __init__(self, index):
        self.error = None
        self.handle = C.c_void_p()
        self.uuid = None
        self.lib = None
        for path in (["nvml.dll", os.path.join(os.environ.get("WINDIR", "C:/Windows"), "System32/nvml.dll")]
                     if os.name == "nt" else ["libnvidia-ml.so.1"]):
            try:
                self.lib = C.CDLL(path)
                break
            except OSError as exc:
                self.error = str(exc)
        if self.lib is None:
            return
        try:
            if self.lib.nvmlInit_v2() != 0:
                raise RuntimeError("NVML init failed")
            if self.lib.nvmlDeviceGetHandleByIndex_v2(C.c_uint(index), C.byref(self.handle)) != 0:
                raise RuntimeError("NVML physical GPU index unavailable")
            value = C.create_string_buffer(96)
            if self.lib.nvmlDeviceGetUUID(self.handle, value, C.c_uint(len(value))) != 0:
                raise RuntimeError("NVML UUID unavailable")
            self.uuid, self.error = value.value.decode(), None
        except Exception as exc:
            self.error = str(exc)

    def scalar(self, name, kind):
        if self.error or self.lib is None:
            return None
        try:
            value = kind()
            if getattr(self.lib, name)(self.handle, C.byref(value)) != 0:
                return None
            return value.value
        except (AttributeError, OSError):
            return None

    def joules(self):
        value = self.scalar("nvmlDeviceGetTotalEnergyConsumption", C.c_ulonglong)
        return None if value is None else value/1000.

    def watts(self):
        value = self.scalar("nvmlDeviceGetPowerUsage", C.c_uint)
        return None if value is None else value/1000.


class EnergyInterval:
    def __init__(self, board, interval=.1):
        self.board, self.interval = board, interval
        self.samples, self.stop_event = [], threading.Event()

    def sample(self):
        self.samples.append({"seconds": time.perf_counter()-self.start, "watts": self.board.watts()})

    def begin(self):
        self.start = time.perf_counter()
        self.counter0 = self.board.joules()
        self.sample()
        def poll():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()

    def end(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        counter1 = self.board.joules()
        integral = integrate_power(self.samples)
        counter = None if self.counter0 is None or counter1 is None or counter1 < self.counter0 else counter1-self.counter0
        # A non-advancing counter cannot certify zero energy for active inference.
        if counter == 0:
            counter = None
        return {"board_joules": counter if counter is not None else integral,
                "method": "NVML_total_energy_counter" if counter is not None else
                          "NVML_power_trapezoidal_estimate" if integral is not None else "unavailable",
                "counter_joules": counter, "power_integral_joules": integral,
                "samples": self.samples, "measurement_seconds": time.perf_counter()-self.start,
                "scope": "gross GPU board incl. memory/idle; excludes CPU and wall-plug; not process attributable",
                "power_sampling_caveat": "Recent GPU power queries may average over 1 second; short runs are approximate",
                "nvml_error": self.board.error}
