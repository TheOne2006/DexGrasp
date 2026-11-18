"""Sample total device usage, including native sparse/NCCL allocations outside Torch."""
import ctypes as C
from threading import Event, Thread


class MemoryInfo(C.Structure):
    _fields_ = [('total', C.c_ulonglong), ('free', C.c_ulonglong), ('used', C.c_ulonglong)]


class DeviceMemory:
    def __init__(self, uuid):
        self.lib = C.CDLL('libnvidia-ml.so.1')
        self.check(self.lib.nvmlInit_v2())
        self.handle = C.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.handle)))
        self.peak = 0
        self.stop = Event()
        self.read()
        self.thread = Thread(target=self.poll, daemon=True)
        self.thread.start()

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'NVML memory query failed with code {code}')

    def read(self):
        value = MemoryInfo()
        self.check(self.lib.nvmlDeviceGetMemoryInfo(self.handle, C.byref(value)))
        self.peak = max(self.peak, value.used)
        self.total = value.total

    def poll(self):
        while not self.stop.wait(.04):
            self.read()

    def close(self):
        self.stop.set()
        self.thread.join()
        self.read()
        self.check(self.lib.nvmlShutdown())
