import contextlib
import os
import threading
from collections import deque

try:
    import nvidia_smi
except Exception:
    nvidia_smi = None

try:
    import pynvml
except Exception:
    pynvml = None


class GPUInfo:
    def __init__(self):
        # Per-device rolling averages
        self.samples_per_second = 10
        self._avg_lock = threading.Lock()
        self._device_avgs = {}
        self._nvml_lock = threading.Lock()
        # Look out for device env var hack
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        self.forced_gpu = cuda_visible_devices is not None
        self.device = 0
        if cuda_visible_devices:
            try:
                self.device = int(cuda_visible_devices.split(",")[0].strip())
            except (TypeError, ValueError):
                self.device = 0
        ui_show_n_gpus_env = os.getenv("AIWORKER_UI_SHOW_N_GPUS")
        self.ui_show_n_gpus = None
        if ui_show_n_gpus_env is not None:
            try:
                self.ui_show_n_gpus = max(1, int(ui_show_n_gpus_env))
            except (TypeError, ValueError):
                self.ui_show_n_gpus = None
        self._nvml_initialized = False

    # Return a value from the given dictionary supporting dot notation
    def get(self, data, key, default=""):
        # Handle nested structures
        path = key.split(".")

        if len(path) == 1:
            # Simple case
            return data.get(key, default)
        # Nested case
        walkdata = data
        for element in path:
            if element in walkdata:
                walkdata = walkdata[element]
            else:
                walkdata = ""
                break
        return walkdata

    def _init_nvml(self):
        if self._nvml_initialized or pynvml is None:
            return self._nvml_initialized
        with self._nvml_lock:
            if self._nvml_initialized or pynvml is None:
                return self._nvml_initialized
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception:
                self._nvml_initialized = False
        return self._nvml_initialized

    def get_num_gpus(self):
        """How many GPUs in this system?"""
        if self.forced_gpu:
            return 1

        if self.ui_show_n_gpus is not None:
            return self.ui_show_n_gpus

        with contextlib.suppress(Exception):
            if nvidia_smi is not None:
                nvsmi = nvidia_smi.getInstance()
                data = nvsmi.DeviceQuery()
                gpus = data.get("gpu", [])
                if isinstance(gpus, dict):
                    gpus = [gpus]
                elif not isinstance(gpus, list):
                    gpus = list(gpus) if gpus else []
                return len(gpus) if gpus else 0

        with contextlib.suppress(Exception):
            if self._init_nvml():
                return int(pynvml.nvmlDeviceGetCount())
        return 0

    def _get_gpu_data(self, device=0):
        with contextlib.suppress(Exception):
            if nvidia_smi is not None:
                nvsmi = nvidia_smi.getInstance()
                data = nvsmi.DeviceQuery()
                gpus = data.get("gpu", [])
                if isinstance(gpus, dict):
                    gpus = [gpus]
                elif not isinstance(gpus, list):
                    gpus = list(gpus) if gpus else []
                if not gpus:
                    return None
                if device >= len(gpus) or device < 0:
                    device = 0
                return gpus[device]

        with contextlib.suppress(Exception):
            if self._init_nvml():
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                try:
                    power = int(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)
                except Exception:
                    power = 0
                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                except Exception:
                    fan = 0
                try:
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="ignore")
                except Exception:
                    name = "unknown"
                try:
                    pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                except Exception:
                    pcie_gen = "?"
                try:
                    pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
                except Exception:
                    pcie_width = "?"
                return {
                    "product_name": name,
                    "pci": {
                        "pci_gpu_link_info": {
                            "pcie_gen": {"current_link_gen": pcie_gen},
                            "link_widths": {"current_link_width": pcie_width},
                        },
                    },
                    "fan_speed": fan,
                    "fan_speed_unit": "%",
                    "fb_memory_usage": {
                        "total": round(mem.total / (1024 * 1024)),
                        "used": round(mem.used / (1024 * 1024)),
                        "free": round(mem.free / (1024 * 1024)),
                    },
                    "utilization": {"gpu_util": util.gpu, "unit": "%"},
                    "temperature": {"gpu_temp": temp, "unit": "C"},
                    "power_readings": {"power_draw": power, "unit": "W"},
                }
        return None

    def _mem(self, raw):
        try:
            return f"{round(float(raw))} MB"
        except (TypeError, ValueError):
            return "0 MB"

    def get_total_vram_mb(self):
        """Get total VRAM in MB as an integer, or 0"""
        value = 0
        device = self.device if self.forced_gpu else 0
        data = self._get_gpu_data(device)
        if data:
            value = self.get(data, "fb_memory_usage.total", "0")
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = 0
        return value

    def get_free_vram_mb(self):
        """Get free VRAM in MB as an integer, or 0"""
        value = 0
        device = self.device if self.forced_gpu else 0
        data = self._get_gpu_data(device)
        if data:
            value = self.get(data, "fb_memory_usage.free", "0")
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = 0
        return value

    def get_info(self, device_id=0):
        # If we're overriding the device using Nvidia driver env var hack
        if self.forced_gpu and device_id == 0:
            device_id = self.device

        data = self._get_gpu_data(device_id)
        if not data:
            return None

        # Calculate averages
        try:
            gpu_util = int(self.get(data, "utilization.gpu_util", 0))
        except (TypeError, ValueError):
            gpu_util = 0

        try:
            gpu_temp = int(self.get(data, "temperature.gpu_temp", 0))
        except (TypeError, ValueError):
            gpu_temp = 0

        try:
            gpu_power = int(self.get(data, "power_readings.power_draw", 0))
        except (TypeError, ValueError):
            gpu_power = 0

        with self._avg_lock:
            max_samples = self.samples_per_second * 60 * 5
            dev = self._device_avgs.setdefault(
                device_id,
                {"load": deque(maxlen=max_samples), "temp": deque(maxlen=max_samples), "power": deque(maxlen=max_samples)},
            )
            dev["load"].append(gpu_util)
            dev["temp"].append(gpu_temp)
            dev["power"].append(gpu_power)
            avg_load = int(sum(dev["load"]) / len(dev["load"])) if dev["load"] else 0
            avg_power = int(sum(dev["power"]) / len(dev["power"])) if dev["power"] else 0
            avg_temp = int(sum(dev["temp"]) / len(dev["temp"])) if dev["temp"] else 0

        return {
            "product": self.get(data, "product_name", "unknown"),
            "pci_gen": self.get(data, "pci.pci_gpu_link_info.pcie_gen.current_link_gen", "?"),
            "pci_width": self.get(data, "pci.pci_gpu_link_info.link_widths.current_link_width", "?"),
            "fan_speed": f"{self.get(data, 'fan_speed')}{self.get(data, 'fan_speed_unit')}",
            "vram_total": f"{self._mem(self.get(data, 'fb_memory_usage.total', '0'))}",
            "vram_used": f"{self._mem(self.get(data, 'fb_memory_usage.used', '0'))}",
            "vram_free": f"{self._mem(self.get(data, 'fb_memory_usage.free', '0'))}",
            "load": f"{gpu_util}{self.get(data, 'utilization.unit')}",
            "temp": f"{gpu_temp}{self.get(data, 'temperature.unit')}",
            "power": f"{gpu_power}{self.get(data, 'power_readings.unit')}",
            "avg_load": f"{avg_load}{self.get(data, 'utilization.unit')}",
            "avg_temp": f"{avg_temp}{self.get(data, 'temperature.unit')}",
            "avg_power": f"{avg_power}{self.get(data, 'power_readings.unit')}",
        }
