"""
Hardware Performance Testing System (CPU & GPU)
Author: Antigravity
Description:
    A high-performance, comprehensive hardware testing, benchmarking, and
    sustained stress testing suite for CPU and GPU.

Features:
    - Direct Native CUDA Driver API integration (zero external overhead)
    - Hardware telemetry via NVML (sub-millisecond GPU temps, power, clocks, utilization)
    - Configurable Timer for sustained Stress Testing (Minutes / Seconds)
    - CPU, GPU, and Combined Maximum Load Stress Testing
    - Advanced Scoring System (CPU Score, GPU Score, System Score, Hardware Tier, Stability Rating)
    - Real-time cross-comparison metrics (CPU vs GPU speedup ratio)
    - Interactive CLI with colored progress and live countdown
    - Modern Interactive Web Dashboard (FastAPI / SSE / REST) with timer controls
    - Standalone HTML & JSON Report generation
"""

import sys
import os
import time
import math
import json
import ctypes
import hashlib
import platform
import argparse
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import psutil

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# =====================================================================
# Telemetry & Hardware Detection
# =====================================================================

class HardwareTelemetry:
    def __init__(self):
        self.has_nvml = False
        self.nvml = None
        self.nvml_handle = None
        self.has_cuda = False
        self.cuda = None
        self.cuda_device = None

        self._init_nvml()
        self._init_cuda()

    def _init_nvml(self):
        try:
            self.nvml = ctypes.CDLL("nvml.dll")
            if self.nvml.nvmlInit() == 0:
                count = ctypes.c_uint()
                self.nvml.nvmlDeviceGetCount(ctypes.byref(count))
                if count.value > 0:
                    self.nvml_handle = ctypes.c_void_p()
                    if self.nvml.nvmlDeviceGetHandleByIndex(0, ctypes.byref(self.nvml_handle)) == 0:
                        self.has_nvml = True
        except Exception:
            self.has_nvml = False

    def _init_cuda(self):
        try:
            self.cuda = ctypes.CDLL("nvcuda.dll")
            if self.cuda.cuInit(0) == 0:
                dev = ctypes.c_int()
                if self.cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0:
                    self.cuda_device = dev
                    self.has_cuda = True
        except Exception:
            self.has_cuda = False

    def get_cpu_info(self):
        cpu_name = platform.processor() or "Generic CPU"
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            winreg.CloseKey(key)
        except Exception:
            pass

        freq = psutil.cpu_freq()
        mem = psutil.virtual_memory()

        return {
            "name": cpu_name,
            "architecture": platform.machine(),
            "physical_cores": psutil.cpu_count(logical=False) or 1,
            "logical_cores": psutil.cpu_count(logical=True) or 1,
            "current_freq_mhz": round(freq.current, 1) if freq else 0,
            "max_freq_mhz": round(freq.max, 1) if freq and freq.max else 0,
            "total_ram_gb": round(mem.total / (1024 ** 3), 2),
            "available_ram_gb": round(mem.available / (1024 ** 3), 2),
        }

    def get_gpu_info(self):
        if not self.has_nvml:
            if HAS_OPENCV and cv2.ocl.haveOpenCL():
                dev = cv2.ocl.Device.getDefault()
                return {
                    "available": True,
                    "name": dev.name(),
                    "vendor": dev.vendorName(),
                    "driver_version": dev.version(),
                    "total_vram_mb": 0,
                    "backend": "OpenCL",
                    "cuda_capable": False
                }
            return {"available": False, "name": "None / Integrated Graphics"}

        name_buf = ctypes.create_string_buffer(96)
        self.nvml.nvmlDeviceGetName(self.nvml_handle, name_buf, 96)
        gpu_name = name_buf.value.decode("utf-8")

        class nvmlMemory_t(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

        mem = nvmlMemory_t()
        self.nvml.nvmlDeviceGetMemoryInfo(self.nvml_handle, ctypes.byref(mem))

        driver_buf = ctypes.create_string_buffer(64)
        driver_ver = "Unknown"
        try:
            if self.nvml.nvmlSystemGetDriverVersion(driver_buf, 64) == 0:
                driver_ver = driver_buf.value.decode("utf-8")
        except Exception:
            pass

        return {
            "available": True,
            "name": gpu_name,
            "driver_version": driver_ver,
            "total_vram_mb": round(mem.total / (1024 ** 2)),
            "free_vram_mb": round(mem.free / (1024 ** 2)),
            "backend": "Native CUDA Driver API + NVML",
            "cuda_capable": self.has_cuda
        }

    def get_live_metrics(self):
        mem = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        freq = psutil.cpu_freq()

        gpu_metrics = {
            "available": False,
            "gpu_util_pct": 0,
            "vram_used_mb": 0,
            "vram_total_mb": 0,
            "vram_pct": 0,
            "temp_c": 0,
            "power_w": 0.0,
            "clock_mhz": 0
        }

        if self.has_nvml and self.nvml_handle:
            class nvmlUtilization_t(ctypes.Structure):
                _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

            class nvmlMemory_t(ctypes.Structure):
                _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

            util = nvmlUtilization_t()
            mem_info = nvmlMemory_t()
            temp = ctypes.c_uint()
            power = ctypes.c_uint()
            clock = ctypes.c_uint()

            if self.nvml.nvmlDeviceGetUtilizationRates(self.nvml_handle, ctypes.byref(util)) == 0:
                gpu_metrics["gpu_util_pct"] = util.gpu

            if self.nvml.nvmlDeviceGetMemoryInfo(self.nvml_handle, ctypes.byref(mem_info)) == 0:
                total_mb = mem_info.total // (1024 ** 2)
                used_mb = mem_info.used // (1024 ** 2)
                gpu_metrics["vram_total_mb"] = total_mb
                gpu_metrics["vram_used_mb"] = used_mb
                gpu_metrics["vram_pct"] = round((used_mb / total_mb * 100), 1) if total_mb else 0

            if self.nvml.nvmlDeviceGetTemperature(self.nvml_handle, 0, ctypes.byref(temp)) == 0:
                gpu_metrics["temp_c"] = temp.value

            if self.nvml.nvmlDeviceGetPowerUsage(self.nvml_handle, ctypes.byref(power)) == 0:
                gpu_metrics["power_w"] = round(power.value / 1000.0, 2)

            if self.nvml.nvmlDeviceGetClockInfo(self.nvml_handle, 0, ctypes.byref(clock)) == 0:
                gpu_metrics["clock_mhz"] = clock.value

            gpu_metrics["available"] = True

        return {
            "cpu_util_pct": cpu_pct,
            "cpu_per_core_pct": cpu_per_core,
            "cpu_freq_mhz": round(freq.current, 1) if freq else 0,
            "ram_used_gb": round((mem.total - mem.available) / (1024 ** 3), 2),
            "ram_total_gb": round(mem.total / (1024 ** 3), 2),
            "ram_pct": mem.percent,
            "gpu": gpu_metrics
        }

    def close(self):
        if self.has_nvml and self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


# =====================================================================
# Scoring & Tier Evaluation System
# =====================================================================

def calculate_hardware_scores(cpu_results, gpu_results, stress_results=None):
    """
    Standardized, balanced hardware scoring system:
      - CPU Score (Points)
      - GPU Score (Points)
      - Total System Benchmark Score (Points)
      - Hardware Performance Tier (S+, S, A, B, C)
      - Stability / Endurance Rating (0 - 100%)
    """
    # 1. CPU Score Calculation
    cpu_single_mflops = cpu_results.get("single_core", {}).get("throughput_mflops", 0)
    cpu_gemm_gflops = cpu_results.get("multi_core_gemm", {}).get("throughput_gflops", 0)
    cpu_ram_bw = cpu_results.get("memory_bandwidth", {}).get("avg_bandwidth_gbps", 0)
    cpu_sha_mbps = cpu_results.get("hashing", {}).get("sha256_mbps", 0)

    cpu_score = round(
        cpu_single_mflops * 2.5 +
        cpu_gemm_gflops * 12.0 +
        cpu_ram_bw * 120.0 +
        cpu_sha_mbps * 1.5
    )

    # 2. GPU Score Calculation
    gpu_tflops = gpu_results.get("cuda_compute", {}).get("throughput_tflops", 0)
    gpu_vram_bw = gpu_results.get("memory_bandwidth", {}).get("device_to_device_vram_gbps", 0)
    gpu_htod = gpu_results.get("memory_bandwidth", {}).get("host_to_device_gbps", 0)
    gpu_dtoh = gpu_results.get("memory_bandwidth", {}).get("device_to_host_gbps", 0)

    if gpu_results.get("available", False):
        gpu_score = round(
            gpu_tflops * 1200.0 +
            gpu_vram_bw * 35.0 +
            (gpu_htod + gpu_dtoh) * 120.0
        )
    else:
        gpu_score = 0

    # 3. Combined System Score
    if gpu_score > 0:
        system_score = round(cpu_score * 0.40 + gpu_score * 0.60)
    else:
        system_score = cpu_score

    # 4. Hardware Tier Badge
    if system_score >= 15000:
        tier = "Tier S+ (Workstation / Flagship Beast)"
        badge_color = "#e056fd"
    elif system_score >= 10000:
        tier = "Tier S (High-End Gaming & Pro Compute)"
        badge_color = "#f0932b"
    elif system_score >= 6500:
        tier = "Tier A (Performance / Mid-Range High)"
        badge_color = "#6ab04c"
    elif system_score >= 3500:
        tier = "Tier B (Mainstream / Everyday Gaming)"
        badge_color = "#22a6b3"
    else:
        tier = "Tier C (Entry Level / Office)"
        badge_color = "#95afc0"

    # 5. Stability & Thermal Rating (if stress test performed)
    stability = None
    if stress_results:
        penalties = 0.0

        # Thermal penalty: Safe < 72°C. Moderate 72-82°C. High > 82°C.
        peak_temp = stress_results.get("gpu_peak_temp_c", 0)
        if peak_temp > 82:
            penalties += (peak_temp - 82) * 2.5 + 10.0
        elif peak_temp > 72:
            penalties += (peak_temp - 72) * 1.0

        # Clock drop penalty (throttling):
        clock_drop_pct = stress_results.get("gpu_clock_drop_pct", 0)
        if clock_drop_pct > 5.0:
            penalties += clock_drop_pct * 1.2

        cpu_drop_pct = stress_results.get("cpu_clock_drop_pct", 0)
        if cpu_drop_pct > 10.0:
            penalties += cpu_drop_pct * 0.8

        stability_score = max(0.0, min(100.0, round(100.0 - penalties, 1)))

        if stability_score >= 95.0:
            rating_label = "Rock Solid (Exceptional Thermal Headroom)"
            stars = "⭐⭐⭐⭐⭐"
        elif stability_score >= 85.0:
            rating_label = "Stable (Normal Thermal Rise, Safe Limits)"
            stars = "⭐⭐⭐⭐"
        elif stability_score >= 70.0:
            rating_label = "Moderate Throttling (Elevated Heat)"
            stars = "⭐⭐⭐"
        else:
            rating_label = "Heavy Throttling (Thermal Limit Reached)"
            stars = "⚠️"

        stability = {
            "score": stability_score,
            "rating_label": rating_label,
            "stars": stars
        }

    return {
        "cpu_score": cpu_score,
        "gpu_score": gpu_score,
        "system_score": system_score,
        "tier": tier,
        "badge_color": badge_color,
        "stability": stability
    }


# =====================================================================
# CPU Benchmark & Sustained Stress Engine
# =====================================================================

def _thread_stress_worker(stop_event, worker_id, results_dict):
    """Thread worker for sustained timed CPU stress testing using GIL-releasing BLAS math."""
    a = np.random.randn(384, 384).astype(np.float32)
    b = np.random.randn(384, 384).astype(np.float32)
    count = 0
    while not stop_event.is_set():
        _ = np.dot(a, b)
        count += 1
    results_dict[worker_id] = count

class CPUBenchmark:
    def __init__(self):
        self.logical_cores = psutil.cpu_count(logical=True) or 1
        self.physical_cores = psutil.cpu_count(logical=False) or 1

    def test_single_core_compute(self, max_iter=300):
        """Single-core Mandelbrot escape-time benchmark."""
        w, h = 800, 800
        t0 = time.perf_counter()
        x = np.linspace(-2.0, 0.5, w, dtype=np.float64)
        y = np.linspace(-1.25, 1.25, h, dtype=np.float64)
        c = x + 1j * y[:, None]
        z = np.zeros_like(c)
        iters_count = np.zeros(c.shape, dtype=np.int32)

        for i in range(max_iter):
            mask = np.abs(z) <= 2.0
            z[mask] = z[mask] * z[mask] + c[mask]
            iters_count[mask] = i

        duration = time.perf_counter() - t0
        total_ops = w * h * max_iter * 6
        mflops = (total_ops / duration) / 1e6

        return {
            "duration_sec": round(duration, 4),
            "throughput_mflops": round(mflops, 2),
            "score": round(mflops * 2.5, 1)
        }

    def test_multi_core_gemm(self, n=2048, iterations=3):
        """Multi-threaded BLAS Matrix Multiplication (GEMM) GFLOPS."""
        a = np.random.randn(n, n).astype(np.float32)
        b = np.random.randn(n, n).astype(np.float32)

        # Warm-up
        _ = np.matmul(a[:256, :256], b[:256, :256])

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = np.matmul(a, b)
            times.append(time.perf_counter() - t0)

        avg_time = sum(times) / len(times)
        total_flops = 2.0 * (n ** 3)
        gflops = (total_flops / avg_time) / 1e9

        return {
            "matrix_size": f"{n}x{n}",
            "iterations": iterations,
            "avg_time_ms": round(avg_time * 1000, 2),
            "throughput_gflops": round(gflops, 2),
            "score": round(gflops * 12.0, 1)
        }

    def test_memory_bandwidth(self, size_mb=256, runs=3):
        """RAM Sequential Read & Write Bandwidth (GB/s)."""
        num_floats = (size_mb * 1024 * 1024) // 4
        data = np.ones(num_floats, dtype=np.float32)

        # Write
        t0 = time.perf_counter()
        for _ in range(runs):
            np.multiply(data, 1.0001, out=data)
        write_time = (time.perf_counter() - t0) / runs
        write_bw = (size_mb * 1024 * 1024 / 1e9) / write_time

        # Read
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = np.sum(data)
        read_time = (time.perf_counter() - t0) / runs
        read_bw = (size_mb * 1024 * 1024 / 1e9) / read_time

        return {
            "buffer_size_mb": size_mb,
            "read_bandwidth_gbps": round(read_bw, 2),
            "write_bandwidth_gbps": round(write_bw, 2),
            "avg_bandwidth_gbps": round((read_bw + write_bw) / 2.0, 2)
        }

    def test_cryptographic_hash(self, chunk_size_mb=32, total_chunks=8):
        """SHA-256 and MD5 throughput in MB/s."""
        data = os.urandom(chunk_size_mb * 1024 * 1024)
        total_bytes = chunk_size_mb * 1024 * 1024 * total_chunks

        t0 = time.perf_counter()
        for _ in range(total_chunks):
            _ = hashlib.sha256(data).digest()
        sha256_time = time.perf_counter() - t0
        sha256_mbps = (total_bytes / (1024 * 1024)) / sha256_time

        t0 = time.perf_counter()
        for _ in range(total_chunks):
            _ = hashlib.md5(data).digest()
        md5_time = time.perf_counter() - t0
        md5_mbps = (total_bytes / (1024 * 1024)) / md5_time

        return {
            "sha256_mbps": round(sha256_mbps, 2),
            "md5_mbps": round(md5_mbps, 2)
        }

    def run_stress_test(self, duration_sec=60, progress_callback=None, stop_event=None):
        """
        Sustained CPU stress test for a user-specified timer duration (seconds / minutes).
        Monitors CPU load %, per-core saturation, frequency drops, and thermal throttling.
        """
        if stop_event is None:
            stop_event = threading.Event()

        results_dict = {}
        workers = []
        for i in range(self.logical_cores):
            t = threading.Thread(
                target=_thread_stress_worker,
                args=(stop_event, i, results_dict)
            )
            t.daemon = True
            workers.append(t)

        start_time = time.perf_counter()
        for t in workers:
            t.start()

        freq_samples = []
        load_samples = []
        init_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 2500.0

        while (time.perf_counter() - start_time) < duration_sec and not stop_event.is_set():
            time.sleep(1.0)
            elapsed = time.perf_counter() - start_time
            remaining = max(0.0, duration_sec - elapsed)

            curr_load = psutil.cpu_percent(interval=None)
            curr_freq = psutil.cpu_freq().current if psutil.cpu_freq() else init_freq
            load_samples.append(curr_load)
            freq_samples.append(curr_freq)

            if progress_callback:
                progress_callback({
                    "elapsed_sec": round(elapsed, 1),
                    "remaining_sec": round(remaining, 1),
                    "cpu_load_pct": curr_load,
                    "cpu_freq_mhz": round(curr_freq, 1)
                })

        stop_event.set()
        for t in workers:
            t.join(timeout=1.0)

        total_gemm = sum(results_dict.values())
        elapsed_total = time.perf_counter() - start_time
        total_flops = total_gemm * 2.0 * (384.0 ** 3)
        sustained_gflops = round((total_flops / elapsed_total) / 1e9, 2) if elapsed_total > 0 else 0

        avg_load = sum(load_samples) / len(load_samples) if load_samples else 100.0
        avg_freq = sum(freq_samples) / len(freq_samples) if freq_samples else init_freq
        min_freq = min(freq_samples) if freq_samples else init_freq

        clock_drop_pct = max(0.0, round(((init_freq - min_freq) / init_freq) * 100.0, 1)) if init_freq > 0 else 0.0
        throttled = clock_drop_pct > 12.0
        stability_score = max(0.0, min(100.0, round(100.0 - (clock_drop_pct * 1.5), 1)))

        return {
            "duration_sec": round(elapsed_total, 2),
            "target_duration_sec": duration_sec,
            "logical_threads_used": self.logical_cores,
            "total_matrix_operations": total_gemm,
            "sustained_gflops": sustained_gflops,
            "avg_cpu_load_pct": round(avg_load, 1),
            "baseline_freq_mhz": round(init_freq, 1),
            "avg_freq_mhz": round(avg_freq, 1),
            "min_freq_mhz": round(min_freq, 1),
            "cpu_clock_drop_pct": clock_drop_pct,
            "thermal_throttling_detected": throttled,
            "stability_score": stability_score
        }

    def run_all(self, verbose_callback=None):
        results = {}
        if verbose_callback:
            verbose_callback("CPU Single-Core Compute (Mandelbrot & Math)...")
        results["single_core"] = self.test_single_core_compute()

        if verbose_callback:
            verbose_callback("CPU Multi-Core GEMM Matrix Multiplication...")
        results["multi_core_gemm"] = self.test_multi_core_gemm()

        if verbose_callback:
            verbose_callback("CPU RAM Memory Bandwidth...")
        results["memory_bandwidth"] = self.test_memory_bandwidth()

        if verbose_callback:
            verbose_callback("CPU Cryptographic Hashing...")
        results["hashing"] = self.test_cryptographic_hash()

        # Score
        results["composite_score"] = round(
            results["single_core"]["score"] * 0.25 +
            results["multi_core_gemm"]["score"] * 0.40 +
            results["memory_bandwidth"]["avg_bandwidth_gbps"] * 20.0 * 0.15 +
            results["hashing"]["sha256_mbps"] * 0.5 * 0.20
        )
        return results


# =====================================================================
# GPU Benchmark & Sustained Stress Engine
# =====================================================================

PTX_COMPUTE_KERNEL = b"""
.version 7.0
.target sm_70
.address_size 64

.visible .entry compute_kernel(
    .param .u64 d_out,
    .param .u32 iters
)
{
    .reg .b32 %r<5>;
    .reg .f32 %f<10>;
    .reg .b64 %rd<4>;
    .reg .pred %p1;

    ld.param.u64 %rd1, [d_out];
    ld.param.u32 %r1, [iters];

    mov.u32 %r2, %tid.x;
    cvt.rn.f32.u32 %f1, %r2;
    mov.f32 %f2, 1.0001;

    mov.u32 %r3, 0;
LOOP:
    fma.rn.f32 %f1, %f1, %f2, 0.0001;
    fma.rn.f32 %f1, %f1, %f2, 0.0001;
    fma.rn.f32 %f1, %f1, %f2, 0.0001;
    fma.rn.f32 %f1, %f1, %f2, 0.0001;
    add.s32 %r3, %r3, 1;
    setp.lt.s32 %p1, %r3, %r1;
    @%p1 bra LOOP;

    mul.wide.u32 %rd2, %r2, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.f32 [%rd3], %f1;
    ret;
}
"""

class GPUBenchmark:
    def __init__(self, telemetry: HardwareTelemetry):
        self.telemetry = telemetry
        self.cuda = telemetry.cuda
        self.has_cuda = telemetry.has_cuda
        self.ctx = None

    def _ensure_context(self):
        if not self.has_cuda or not self.cuda:
            return False
        if self.ctx is None:
            ctx = ctypes.c_void_p()
            res = self.cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, self.telemetry.cuda_device)
            if res != 0:
                return False
            self.ctx = ctx
        return True

    def _release_context(self):
        if self.ctx and self.cuda:
            try:
                self.cuda.cuCtxDestroy_v2(self.ctx)
            except Exception:
                pass
            self.ctx = None

    def test_vram_bandwidth(self, size_mb=256, runs=5):
        if not self._ensure_context():
            return {"error": "CUDA context unavailable"}

        n_bytes = size_mb * 1024 * 1024
        d_ptr1 = ctypes.c_ulonglong()
        d_ptr2 = ctypes.c_ulonglong()

        self.cuda.cuMemAlloc_v2(ctypes.byref(d_ptr1), n_bytes)
        self.cuda.cuMemAlloc_v2(ctypes.byref(d_ptr2), n_bytes)

        h_data = np.ones(n_bytes // 4, dtype=np.float32)
        h_out = np.empty_like(h_data)

        # Warm-up
        self.cuda.cuMemcpyHtoD_v2(d_ptr1, h_data.ctypes.data_as(ctypes.c_void_p), n_bytes)
        self.cuda.cuCtxSynchronize()

        # Host to Device (Upload)
        t0 = time.perf_counter()
        for _ in range(runs):
            self.cuda.cuMemcpyHtoD_v2(d_ptr1, h_data.ctypes.data_as(ctypes.c_void_p), n_bytes)
            self.cuda.cuCtxSynchronize()
        htod_time = (time.perf_counter() - t0) / runs
        htod_gbps = (n_bytes / 1e9) / htod_time

        # Device to Host (Download)
        t0 = time.perf_counter()
        for _ in range(runs):
            self.cuda.cuMemcpyDtoH_v2(h_out.ctypes.data_as(ctypes.c_void_p), d_ptr1, n_bytes)
            self.cuda.cuCtxSynchronize()
        dtoh_time = (time.perf_counter() - t0) / runs
        dtoh_gbps = (n_bytes / 1e9) / dtoh_time

        # Device to Device (VRAM)
        t0 = time.perf_counter()
        for _ in range(runs):
            self.cuda.cuMemcpyDtoD_v2(d_ptr2, d_ptr1, n_bytes)
            self.cuda.cuCtxSynchronize()
        dtod_time = (time.perf_counter() - t0) / runs
        dtod_gbps = (n_bytes / 1e9) / dtod_time

        self.cuda.cuMemFree_v2(d_ptr1)
        self.cuda.cuMemFree_v2(d_ptr2)

        return {
            "buffer_size_mb": size_mb,
            "host_to_device_gbps": round(htod_gbps, 2),
            "device_to_host_gbps": round(dtoh_gbps, 2),
            "device_to_device_vram_gbps": round(dtod_gbps, 2)
        }

    def test_cuda_peak_compute(self, threads=1024, blocks=1000, iters=80000, runs=5):
        if not self._ensure_context():
            return {"error": "CUDA context unavailable"}

        mod = ctypes.c_void_p()
        err = self.cuda.cuModuleLoadData(ctypes.byref(mod), PTX_COMPUTE_KERNEL)
        if err != 0:
            return {"error": f"Failed to load PTX kernel (code {err})"}

        func = ctypes.c_void_p()
        self.cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"compute_kernel")

        total_threads = threads * blocks
        d_out = ctypes.c_ulonglong()
        self.cuda.cuMemAlloc_v2(ctypes.byref(d_out), total_threads * 4)

        c_iters = ctypes.c_uint(iters)
        args = [ctypes.byref(d_out), ctypes.byref(c_iters)]
        arg_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(a, ctypes.c_void_p) for a in args])

        # Warm-up
        self.cuda.cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, arg_ptrs, 0)
        self.cuda.cuCtxSynchronize()

        # Bench
        t0 = time.perf_counter()
        for _ in range(runs):
            self.cuda.cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, arg_ptrs, 0)
        self.cuda.cuCtxSynchronize()
        total_time = time.perf_counter() - t0

        total_flops = runs * total_threads * iters * 8.0
        gflops = (total_flops / total_time) / 1e9
        tflops = gflops / 1000.0

        self.cuda.cuMemFree_v2(d_out)
        self.cuda.cuModuleUnload(mod)

        return {
            "threads_per_block": threads,
            "blocks": blocks,
            "iterations": iters,
            "compute_time_sec": round(total_time, 4),
            "throughput_gflops": round(gflops, 2),
            "throughput_tflops": round(tflops, 2),
            "score": round(gflops * 2.0, 1)
        }

    def test_opencl_vision_throughput(self, img_dim=4096, kernel_size=25, iterations=10):
        if not HAS_OPENCV or not cv2.ocl.haveOpenCL():
            return {"available": False, "note": "OpenCL not enabled in OpenCV"}

        cv2.ocl.setUseOpenCL(True)
        img = np.random.randint(0, 256, (img_dim, img_dim, 3), dtype=np.uint8)
        u_img = cv2.UMat(img)

        # Warmup
        u_out = cv2.GaussianBlur(u_img, (kernel_size, kernel_size), 0)
        _ = u_out.get()

        t0 = time.perf_counter()
        for _ in range(iterations):
            u_out = cv2.GaussianBlur(u_img, (kernel_size, kernel_size), 0)
            _ = u_out.get()
        avg_time = (time.perf_counter() - t0) / iterations

        megapixels = (img_dim * img_dim) / 1e6
        mp_per_sec = megapixels / avg_time
        fps = 1.0 / avg_time

        return {
            "resolution": f"{img_dim}x{img_dim} (4K)",
            "kernel_size": f"{kernel_size}x{kernel_size}",
            "avg_latency_ms": round(avg_time * 1000, 2),
            "fps": round(fps, 1),
            "megapixels_per_sec": round(mp_per_sec, 1)
        }

    def run_stress_test(self, duration_sec=60, progress_callback=None, stop_event=None):
        """
        Sustained GPU stress test for a user-specified timer duration (seconds / minutes).
        Monitors temperature rise, power delivery (Watts), graphics clocks (MHz), and clock throttling.
        """
        if not self._ensure_context():
            return {"error": "CUDA context unavailable"}

        mod = ctypes.c_void_p()
        err = self.cuda.cuModuleLoadData(ctypes.byref(mod), PTX_COMPUTE_KERNEL)
        if err != 0:
            return {"error": "PTX load error"}

        func = ctypes.c_void_p()
        self.cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"compute_kernel")

        threads, blocks = 1024, 1000
        d_out = ctypes.c_ulonglong()
        self.cuda.cuMemAlloc_v2(ctypes.byref(d_out), threads * blocks * 4)

        c_iters = ctypes.c_uint(120000)
        args = [ctypes.byref(d_out), ctypes.byref(c_iters)]
        arg_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(a, ctypes.c_void_p) for a in args])

        # Warm-up launch to bring GPU into P0 boost performance state
        self.cuda.cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, arg_ptrs, 0)
        self.cuda.cuCtxSynchronize()

        initial_metrics = self.telemetry.get_live_metrics()["gpu"]
        start_time = time.perf_counter()

        temp_samples = []
        power_samples = []
        clock_samples = []
        kernel_launches = 0

        last_progress_time = 0

        while (time.perf_counter() - start_time) < duration_sec:
            if stop_event and stop_event.is_set():
                break

            self.cuda.cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, arg_ptrs, 0)
            self.cuda.cuCtxSynchronize()
            kernel_launches += 1

            now = time.perf_counter()
            if (now - last_progress_time) >= 0.8:
                last_progress_time = now
                curr = self.telemetry.get_live_metrics()["gpu"]
                temp_samples.append(curr.get("temp_c", 0))
                power_samples.append(curr.get("power_w", 0))
                clock_samples.append(curr.get("clock_mhz", 0))

                elapsed = now - start_time
                remaining = max(0.0, duration_sec - elapsed)

                if progress_callback:
                    progress_callback({
                        "elapsed_sec": round(elapsed, 1),
                        "remaining_sec": round(remaining, 1),
                        "temp_c": curr.get("temp_c", 0),
                        "power_w": curr.get("power_w", 0),
                        "clock_mhz": curr.get("clock_mhz", 0),
                        "gpu_util_pct": curr.get("gpu_util_pct", 100)
                    })

        elapsed_total = time.perf_counter() - start_time
        final_metrics = self.telemetry.get_live_metrics()["gpu"]

        self.cuda.cuMemFree_v2(d_out)
        self.cuda.cuModuleUnload(mod)

        # Telemetry Analysis
        init_temp = initial_metrics.get("temp_c", 0)
        peak_temp = max(temp_samples) if temp_samples else init_temp
        avg_temp = round(sum(temp_samples) / len(temp_samples), 1) if temp_samples else init_temp
        delta_temp = peak_temp - init_temp

        peak_power = max(power_samples) if power_samples else 0.0
        avg_power = round(sum(power_samples) / len(power_samples), 1) if power_samples else 0.0

        # Filter out idle clocks (< 600 MHz) when computing under-load throttling
        active_clocks = [c for c in clock_samples if c > 600]
        init_clock = active_clocks[0] if active_clocks else (clock_samples[0] if clock_samples else 0)
        max_clock = max(active_clocks) if active_clocks else (max(clock_samples) if clock_samples else 0)
        min_clock = min(active_clocks) if active_clocks else max_clock

        clock_drop_pct = max(0.0, round(((max_clock - min_clock) / max_clock) * 100.0, 1)) if max_clock > 0 else 0.0
        throttled = clock_drop_pct > 15.0 or peak_temp >= 86

        # Sustained compute TFLOPS
        total_flops = kernel_launches * threads * blocks * 120000 * 8.0
        sustained_tflops = round((total_flops / elapsed_total) / 1e12, 2) if elapsed_total > 0 else 0.0

        # Stability Score (100 - penalties)
        penalties = 0.0
        if peak_temp > 80:
            penalties += (peak_temp - 80) * 2.0
        if clock_drop_pct > 5:
            penalties += clock_drop_pct * 1.0

        stability_score = max(0.0, min(100.0, round(100.0 - penalties, 1)))

        return {
            "duration_sec": round(elapsed_total, 2),
            "target_duration_sec": duration_sec,
            "kernel_launches": kernel_launches,
            "sustained_tflops": sustained_tflops,
            "initial_temp_c": init_temp,
            "peak_temp_c": peak_temp,
            "avg_temp_c": avg_temp,
            "delta_temp_c": delta_temp,
            "peak_power_w": peak_power,
            "avg_power_w": avg_power,
            "initial_clock_mhz": init_clock,
            "max_clock_mhz": max_clock,
            "min_clock_mhz": min_clock,
            "gpu_clock_drop_pct": clock_drop_pct,
            "thermal_throttling_detected": throttled,
            "stability_score": stability_score
        }

    def run_all(self, verbose_callback=None):
        if not self.has_cuda:
            return {"available": False, "note": "No NVIDIA CUDA GPU detected on system."}

        results = {"available": True}
        if verbose_callback:
            verbose_callback("GPU PCIe & VRAM Memory Bandwidth...")
        results["memory_bandwidth"] = self.test_vram_bandwidth()

        if verbose_callback:
            verbose_callback("GPU Native CUDA FP32 Peak Compute (TFLOPS)...")
        results["cuda_compute"] = self.test_cuda_peak_compute()

        if verbose_callback:
            verbose_callback("GPU OpenCL 4K Vision Acceleration...")
        results["opencl_vision"] = self.test_opencl_vision_throughput()

        if verbose_callback:
            verbose_callback("GPU Thermal & Power Stress Test...")
        results["thermal_stress"] = self.run_stress_test(duration_sec=3)

        vram_bw = results["memory_bandwidth"].get("device_to_device_vram_gbps", 0)
        gflops = results["cuda_compute"].get("throughput_gflops", 0)
        htod = results["memory_bandwidth"].get("host_to_device_gbps", 0)

        results["composite_score"] = round(
            gflops * 1.5 * 0.60 +
            vram_bw * 30.0 * 0.25 +
            htod * 100.0 * 0.15
        )

        self._release_context()
        return results


# =====================================================================
# Combined Stress Engine (Max System Load)
# =====================================================================

def run_combined_stress_test(duration_sec, telemetry: HardwareTelemetry, progress_callback=None):
    """Runs CPU workers and GPU CUDA kernel simultaneously for the user-selected duration."""
    stop_event = threading.Event()

    cpu_bench = CPUBenchmark()
    gpu_bench = GPUBenchmark(telemetry)

    cpu_thread = threading.Thread(
        target=lambda: setattr(run_combined_stress_test, "cpu_res", cpu_bench.run_stress_test(duration_sec, stop_event=stop_event))
    )
    gpu_thread = threading.Thread(
        target=lambda: setattr(run_combined_stress_test, "gpu_res", gpu_bench.run_stress_test(duration_sec, stop_event=stop_event))
    )

    cpu_thread.start()
    gpu_thread.start()

    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) < duration_sec:
        time.sleep(1.0)
        elapsed = time.perf_counter() - start_time
        remaining = max(0.0, duration_sec - elapsed)
        metrics = telemetry.get_live_metrics()

        if progress_callback:
            progress_callback({
                "elapsed_sec": round(elapsed, 1),
                "remaining_sec": round(remaining, 1),
                "cpu_load_pct": metrics.get("cpu_util_pct", 0),
                "gpu_temp_c": metrics.get("gpu", {}).get("temp_c", 0),
                "gpu_power_w": metrics.get("gpu", {}).get("power_w", 0),
                "gpu_util_pct": metrics.get("gpu", {}).get("gpu_util_pct", 0)
            })

    stop_event.set()
    cpu_thread.join()
    gpu_thread.join()

    cpu_res = getattr(run_combined_stress_test, "cpu_res", {})
    gpu_res = getattr(run_combined_stress_test, "gpu_res", {})

    combined_stability = round((cpu_res.get("stability_score", 100) + gpu_res.get("stability_score", 100)) / 2.0, 1)

    return {
        "duration_sec": duration_sec,
        "cpu_stress": cpu_res,
        "gpu_stress": gpu_res,
        "combined_stability_score": combined_stability,
        "peak_gpu_temp_c": gpu_res.get("peak_temp_c", 0),
        "peak_gpu_power_w": gpu_res.get("peak_power_w", 0)
    }


# =====================================================================
# Cross-Comparison Engine
# =====================================================================

def compute_hardware_comparison(cpu_results, gpu_results):
    comparison = {
        "metrics": []
    }

    cpu_gflops = cpu_results.get("multi_core_gemm", {}).get("throughput_gflops", 0)
    gpu_gflops = gpu_results.get("cuda_compute", {}).get("throughput_gflops", 0)

    if cpu_gflops > 0 and gpu_gflops > 0:
        compute_speedup = round(gpu_gflops / cpu_gflops, 2)
        comparison["compute_speedup_x"] = compute_speedup
        comparison["metrics"].append({
            "category": "Raw Floating-Point Compute",
            "cpu_value": f"{cpu_gflops} GFLOPS",
            "gpu_value": f"{gpu_gflops} GFLOPS ({round(gpu_gflops/1000, 2)} TFLOPS)",
            "speedup": f"{compute_speedup}x Faster on GPU"
        })

    cpu_bw = cpu_results.get("memory_bandwidth", {}).get("avg_bandwidth_gbps", 0)
    gpu_bw = gpu_results.get("memory_bandwidth", {}).get("device_to_device_vram_gbps", 0)

    if cpu_bw > 0 and gpu_bw > 0:
        bw_speedup = round(gpu_bw / cpu_bw, 2)
        comparison["bandwidth_speedup_x"] = bw_speedup
        comparison["metrics"].append({
            "category": "Internal Memory Bandwidth",
            "cpu_value": f"{cpu_bw} GB/s (System RAM)",
            "gpu_value": f"{gpu_bw} GB/s (VRAM)",
            "speedup": f"{bw_speedup}x Higher on GPU"
        })

    return comparison


# =====================================================================
# HTML Report Generator
# =====================================================================

def generate_html_report(cpu_info, gpu_info, cpu_results, gpu_results, comparison, scores, stress_results=None, output_file="benchmark_report.html"):
    stability_html = ""
    if scores.get("stability"):
        stab = scores["stability"]
        stability_html = f"""
        <div class="card" style="margin-bottom: 25px; border-left: 5px solid {scores['badge_color']};">
            <div class="card-title">
                Thermal & Stress Stability Rating
                <span class="badge" style="background: rgba(255, 183, 3, 0.15); color: #ffb703;">{stab['stars']}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                <div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #ffb703;">{stab['score']}%</div>
                    <div style="color: var(--text-dim); margin-top: 4px;">{stab['rating_label']}</div>
                </div>
                <div style="text-align: right; font-size: 0.95rem; color: var(--text-dim);">
                    Peak GPU Temp: <strong style="color: #fff;">{stress_results.get('peak_temp_c', stress_results.get('gpu_stress', {}).get('peak_temp_c', 'N/A'))} &deg;C</strong><br>
                    Peak GPU Power: <strong style="color: #fff;">{stress_results.get('peak_power_w', stress_results.get('gpu_stress', {}).get('peak_power_w', 'N/A'))} W</strong>
                </div>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CPU & GPU Performance Benchmark & Stress Report</title>
    <style>
        :root {{
            --bg: #0b0f19;
            --card-bg: rgba(22, 30, 49, 0.75);
            --border: rgba(255, 255, 255, 0.08);
            --accent-cpu: #00d2ff;
            --accent-gpu: #76b900;
            --accent-gold: #ffb703;
            --text-main: #f1f5f9;
            --text-dim: #94a3b8;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background: var(--bg); color: var(--text-main); padding: 30px 20px; min-height: 100vh; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 35px; padding-bottom: 25px; border-bottom: 1px solid var(--border); }}
        h1 {{ font-size: 2.3rem; font-weight: 800; margin-bottom: 8px; }}
        .subtitle {{ color: var(--text-dim); font-size: 1.05rem; }}
        
        .score-banner {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .score-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(12px);
        }}
        .score-card.system {{ border-top: 4px solid {scores['badge_color']}; }}
        .score-card.cpu {{ border-top: 4px solid var(--accent-cpu); }}
        .score-card.gpu {{ border-top: 4px solid var(--accent-gpu); }}
        .score-card.ratio {{ border-top: 4px solid var(--accent-gold); }}
        .score-label {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); }}
        .score-value {{ font-size: 2.7rem; font-weight: 800; margin: 8px 0 4px; }}
        .score-desc {{ font-size: 0.9rem; color: var(--text-dim); }}

        .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 24px; backdrop-filter: blur(12px); }}
        .card-title {{ font-size: 1.25rem; font-weight: 700; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between; }}
        .badge {{ font-size: 0.75rem; padding: 5px 12px; border-radius: 20px; font-weight: 700; }}

        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-bottom: 30px; }}
        @media(max-width: 860px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.95rem; }}
        th {{ color: var(--text-dim); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }}
        td.val {{ font-weight: 600; text-align: right; }}

        footer {{ text-align: center; color: var(--text-dim); font-size: 0.85rem; margin-top: 50px; padding-top: 20px; border-top: 1px solid var(--border); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Hardware Performance & Stress Report</h1>
            <div class="subtitle">System Benchmarking & Sustained Thermal Endurance Analysis &bull; {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
        </header>

        <div class="score-banner">
            <div class="score-card system">
                <div class="score-label">System Performance Tier</div>
                <div class="score-value" style="color: {scores['badge_color']};">{scores['system_score']}</div>
                <div class="score-desc" style="font-weight: 700; color: #fff;">{scores['tier']}</div>
            </div>
            <div class="score-card cpu">
                <div class="score-label">CPU Benchmark Score</div>
                <div class="score-value" style="color: var(--accent-cpu);">{scores['cpu_score']}</div>
                <div class="score-desc">{cpu_info.get('name')} ({cpu_info.get('logical_cores')} Cores)</div>
            </div>
            <div class="score-card gpu">
                <div class="score-label">GPU Benchmark Score</div>
                <div class="score-value" style="color: var(--accent-gpu);">{scores['gpu_score']}</div>
                <div class="score-desc">{gpu_info.get('name')} ({gpu_info.get('total_vram_mb')} MB VRAM)</div>
            </div>
            <div class="score-card ratio">
                <div class="score-label">Compute Advantage</div>
                <div class="score-value" style="color: var(--accent-gold);">{comparison.get('compute_speedup_x', 'N/A')}x</div>
                <div class="score-desc">Parallel Compute Speedup vs CPU</div>
            </div>
        </div>

        {stability_html}

        <div class="card" style="margin-bottom: 30px;">
            <div class="card-title">
                Hardware Cross-Comparison
                <span class="badge" style="background: rgba(255, 183, 3, 0.15); color: var(--accent-gold);">Direct Speedup</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Workload Category</th>
                        <th>CPU Result</th>
                        <th>GPU Result</th>
                        <th style="text-align: right;">Speedup</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f"<tr><td><strong>{m['category']}</strong></td><td>{m['cpu_value']}</td><td>{m['gpu_value']}</td><td style='color: var(--accent-gold); font-weight:700; text-align:right;'>{m['speedup']}</td></tr>" for m in comparison.get('metrics', []))}
                </tbody>
            </table>
        </div>

        <div class="grid-2">
            <div class="card">
                <div class="card-title">
                    CPU Detailed Metrics
                    <span class="badge" style="background: rgba(0, 210, 255, 0.15); color: var(--accent-cpu);">Score: {scores['cpu_score']}</span>
                </div>
                <table>
                    <tbody>
                        <tr><td>Hardware Model</td><td class="val">{cpu_info.get('name')}</td></tr>
                        <tr><td>Cores / Threads</td><td class="val">{cpu_info.get('physical_cores')} Physical / {cpu_info.get('logical_cores')} Logical</td></tr>
                        <tr><td>Clock Frequency</td><td class="val">{cpu_info.get('current_freq_mhz')} MHz</td></tr>
                        <tr><td>Single-Core Arithmetic</td><td class="val">{cpu_results.get('single_core', {}).get('throughput_mflops', 0)} MFLOPS</td></tr>
                        <tr><td>Multi-Core GEMM (2048x2048)</td><td class="val">{cpu_results.get('multi_core_gemm', {}).get('throughput_gflops', 0)} GFLOPS</td></tr>
                        <tr><td>RAM Read / Write Bandwidth</td><td class="val">{cpu_results.get('memory_bandwidth', {}).get('read_bandwidth_gbps', 0)} / {cpu_results.get('memory_bandwidth', {}).get('write_bandwidth_gbps', 0)} GB/s</td></tr>
                        <tr><td>SHA-256 Hashing</td><td class="val">{cpu_results.get('hashing', {}).get('sha256_mbps', 0)} MB/s</td></tr>
                    </tbody>
                </table>
            </div>

            <div class="card">
                <div class="card-title">
                    GPU Detailed Metrics
                    <span class="badge" style="background: rgba(118, 185, 0, 0.15); color: var(--accent-gpu);">Score: {scores['gpu_score']}</span>
                </div>
                <table>
                    <tbody>
                        <tr><td>Hardware Model</td><td class="val">{gpu_info.get('name')}</td></tr>
                        <tr><td>VRAM Capacity</td><td class="val">{gpu_info.get('total_vram_mb')} MB</td></tr>
                        <tr><td>Native CUDA Compute</td><td class="val" style="color: var(--accent-gpu);">{gpu_results.get('cuda_compute', {}).get('throughput_tflops', 0)} TFLOPS ({gpu_results.get('cuda_compute', {}).get('throughput_gflops', 0)} GFLOPS)</td></tr>
                        <tr><td>Internal VRAM Bandwidth</td><td class="val">{gpu_results.get('memory_bandwidth', {}).get('device_to_device_vram_gbps', 0)} GB/s</td></tr>
                        <tr><td>PCIe Upload / Download</td><td class="val">{gpu_results.get('memory_bandwidth', {}).get('host_to_device_gbps', 0)} / {gpu_results.get('memory_bandwidth', {}).get('device_to_host_gbps', 0)} GB/s</td></tr>
                        <tr><td>4K Vision Acceleration</td><td class="val">{gpu_results.get('opencl_vision', {}).get('megapixels_per_sec', 0)} MP/s ({gpu_results.get('opencl_vision', {}).get('fps', 0)} FPS)</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <footer>
            Generated automatically by Antigravity Hardware Testing Suite &bull; Windows 11 &bull; Python {sys.version.split()[0]}
        </footer>
    </div>
</body>
</html>"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_file


# =====================================================================
# Interactive Modern Web Dashboard (FastAPI / SSE)
# =====================================================================

def create_fastapi_app(telemetry: HardwareTelemetry):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="CPU & GPU Performance Testing System")

    stress_state = {
        "is_running": False,
        "type": "none",
        "elapsed_sec": 0,
        "remaining_sec": 0,
        "duration_sec": 60,
        "live_data": {},
        "last_result": None,
        "stop_event": None
    }

    state = {
        "last_cpu_results": {},
        "last_gpu_results": {},
        "last_comparison": {},
        "last_scores": {}
    }

    def to_serializable(obj):
        """Recursively converts NumPy and custom types to native JSON-serializable types."""
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        return obj

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "traceback": traceback.format_exc()}
        )

    class StressRequest(BaseModel):
        duration_minutes: float = 1.0

    @app.get("/api/info")
    def api_info():
        return to_serializable({
            "cpu": telemetry.get_cpu_info(),
            "gpu": telemetry.get_gpu_info()
        })

    @app.get("/api/telemetry")
    def api_telemetry():
        return to_serializable(telemetry.get_live_metrics())

    @app.get("/api/stress/status")
    def api_stress_status():
        return to_serializable({
            "is_running": stress_state["is_running"],
            "type": stress_state["type"],
            "elapsed_sec": stress_state["elapsed_sec"],
            "remaining_sec": stress_state["remaining_sec"],
            "duration_sec": stress_state["duration_sec"],
            "live_data": stress_state["live_data"],
            "last_result": stress_state["last_result"],
            "last_scores": state["last_scores"]
        })

    @app.post("/api/stress/stop")
    def api_stress_stop():
        if stress_state["stop_event"]:
            stress_state["stop_event"].set()
        stress_state["is_running"] = False
        return {"status": "stopped"}

    @app.post("/api/stress/cpu")
    def api_stress_cpu(req: StressRequest):
        duration_sec = max(5, int(req.duration_minutes * 60))
        stress_state["is_running"] = True
        stress_state["type"] = "cpu"
        stress_state["duration_sec"] = duration_sec
        stress_state["stop_event"] = threading.Event()

        def run_thread():
            bench = CPUBenchmark()
            def cb(data):
                stress_state["elapsed_sec"] = data["elapsed_sec"]
                stress_state["remaining_sec"] = data["remaining_sec"]
                stress_state["live_data"] = data

            res = bench.run_stress_test(duration_sec, progress_callback=cb, stop_event=stress_state["stop_event"])
            stress_state["is_running"] = False
            stress_state["last_result"] = res

        threading.Thread(target=run_thread, daemon=True).start()
        return {"status": "started", "duration_sec": duration_sec}

    @app.post("/api/stress/gpu")
    def api_stress_gpu(req: StressRequest):
        duration_sec = max(5, int(req.duration_minutes * 60))
        stress_state["is_running"] = True
        stress_state["type"] = "gpu"
        stress_state["duration_sec"] = duration_sec
        stress_state["stop_event"] = threading.Event()

        def run_thread():
            bench = GPUBenchmark(telemetry)
            def cb(data):
                stress_state["elapsed_sec"] = data["elapsed_sec"]
                stress_state["remaining_sec"] = data["remaining_sec"]
                stress_state["live_data"] = data

            res = bench.run_stress_test(duration_sec, progress_callback=cb, stop_event=stress_state["stop_event"])
            stress_state["is_running"] = False
            stress_state["last_result"] = res

        threading.Thread(target=run_thread, daemon=True).start()
        return {"status": "started", "duration_sec": duration_sec}

    @app.post("/api/stress/both")
    def api_stress_both(req: StressRequest):
        duration_sec = max(5, int(req.duration_minutes * 60))
        stress_state["is_running"] = True
        stress_state["type"] = "both"
        stress_state["duration_sec"] = duration_sec
        stress_state["stop_event"] = threading.Event()

        def run_thread():
            def cb(data):
                stress_state["elapsed_sec"] = data["elapsed_sec"]
                stress_state["remaining_sec"] = data["remaining_sec"]
                stress_state["live_data"] = data

            res = run_combined_stress_test(duration_sec, telemetry, progress_callback=cb)
            stress_state["is_running"] = False
            stress_state["last_result"] = res

        threading.Thread(target=run_thread, daemon=True).start()
        return {"status": "started", "duration_sec": duration_sec}

    @app.post("/api/run-cpu")
    def api_run_cpu():
        cpu_bench = CPUBenchmark()
        cpu_res = cpu_bench.run_all()
        gpu_res = state.get("last_gpu_results", {})
        scores = calculate_hardware_scores(cpu_res, gpu_res)
        state["last_cpu_results"] = cpu_res
        state["last_scores"] = scores
        return to_serializable({
            "cpu": cpu_res,
            "scores": scores
        })

    @app.post("/api/run-gpu")
    def api_run_gpu():
        gpu_bench = GPUBenchmark(telemetry)
        gpu_res = gpu_bench.run_all()
        cpu_res = state.get("last_cpu_results", {})
        scores = calculate_hardware_scores(cpu_res, gpu_res)
        state["last_gpu_results"] = gpu_res
        state["last_scores"] = scores
        return to_serializable({
            "gpu": gpu_res,
            "scores": scores
        })

    @app.post("/api/run-all")
    def api_run_all():
        cpu_bench = CPUBenchmark()
        cpu_res = cpu_bench.run_all()
        gpu_bench = GPUBenchmark(telemetry)
        gpu_res = gpu_bench.run_all()
        comp = compute_hardware_comparison(cpu_res, gpu_res)
        scores = calculate_hardware_scores(cpu_res, gpu_res)

        state["last_cpu_results"] = cpu_res
        state["last_gpu_results"] = gpu_res
        state["last_comparison"] = comp
        state["last_scores"] = scores

        generate_html_report(
            telemetry.get_cpu_info(),
            telemetry.get_gpu_info(),
            cpu_res,
            gpu_res,
            comp,
            scores,
            output_file="benchmark_report.html"
        )

        return to_serializable({
            "cpu": cpu_res,
            "gpu": gpu_res,
            "comparison": comp,
            "scores": scores
        })

    @app.get("/", response_class=HTMLResponse)
    def index_page():
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CPU & GPU Performance Testing & Stress Suite</title>
    <style>
        :root {
            --bg: #090d16;
            --panel: rgba(18, 26, 43, 0.85);
            --border: rgba(255, 255, 255, 0.08);
            --accent-cpu: #00d2ff;
            --accent-gpu: #76b900;
            --accent-gold: #ffb703;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
        }
        * { margin:0; padding:0; box-sizing:border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text-main); min-height: 100vh; padding: 25px 20px; }
        .container { max-width: 1250px; margin: 0 auto; }
        
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; padding-bottom: 20px; border-bottom: 1px solid var(--border); flex-wrap: wrap; gap: 15px; }
        .header-title h1 { font-size: 1.8rem; font-weight: 800; }
        .header-title p { color: var(--text-dim); font-size: 0.95rem; margin-top: 4px; }
        
        .action-bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .timer-select-group { display: flex; align-items: center; gap: 8px; background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 10px; border: 1px solid var(--border); }
        .timer-select-group label { font-size: 0.85rem; color: var(--text-dim); font-weight: 600; text-transform: uppercase; }
        select, input[type="number"] { background: #121a2b; color: #fff; border: 1px solid var(--border); padding: 6px 10px; border-radius: 6px; font-weight: 600; outline: none; }
        
        button {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 9px 16px;
            border-radius: 9px;
            font-size: 0.92rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        button:hover { background: rgba(255, 255, 255, 0.12); transform: translateY(-1px); }
        button.btn-primary { background: linear-gradient(135deg, #0284c7, #0369a1); border: none; }
        button.btn-stress-cpu { background: linear-gradient(135deg, #0ea5e9, #0284c7); border: none; }
        button.btn-stress-gpu { background: linear-gradient(135deg, #65a30d, #4d7c0f); border: none; }
        button.btn-stress-max { background: linear-gradient(135deg, #ef4444, #b91c1c); border: none; }
        button.btn-stop { background: #dc2626; border: none; }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        .telemetry-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 15px; margin-bottom: 25px; }
        .tel-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; backdrop-filter: blur(10px); }
        .tel-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-dim); margin-bottom: 6px; }
        .tel-val { font-size: 1.8rem; font-weight: 800; }
        .tel-sub { font-size: 0.85rem; color: var(--text-dim); margin-top: 4px; }

        .progress-bar-bg { background: rgba(255, 255, 255, 0.08); height: 8px; border-radius: 4px; overflow: hidden; margin-top: 8px; }
        .progress-bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }

        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; }
        @media(max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }

        .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 22px; backdrop-filter: blur(10px); }
        .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }
        .panel-title { font-size: 1.2rem; font-weight: 700; }

        .spec-item { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 0.95rem; }
        .spec-label { color: var(--text-dim); }
        .spec-val { font-weight: 600; }

        .stress-banner {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 25px;
            display: none;
            justify-content: space-between;
            align-items: center;
        }
        .countdown-timer { font-size: 1.8rem; font-weight: 800; color: #f87171; font-variant-numeric: tabular-nums; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>Hardware Performance & Stress Testing</h1>
                <p>Timed Load Testing &bull; Native CUDA Compute &bull; Multi-Core CPU &bull; Scoring System</p>
            </div>
            <div class="action-bar">
                <div class="timer-select-group">
                    <label>Duration:</label>
                    <select id="timer-minutes">
                        <option value="0.5">0.5 Min (30s)</option>
                        <option value="1.0" selected>1.0 Minute</option>
                        <option value="2.0">2.0 Minutes</option>
                        <option value="5.0">5.0 Minutes</option>
                        <option value="10.0">10.0 Minutes</option>
                    </select>
                </div>
                <button class="btn-stress-cpu" onclick="startStress('/api/stress/cpu')">⚡ Stress CPU</button>
                <button class="btn-stress-gpu" onclick="startStress('/api/stress/gpu')">⚡ Stress GPU</button>
                <button class="btn-stress-max" onclick="startStress('/api/stress/both')">🔥 Max Load Both</button>
                <button onclick="runBenchmark('/api/run-cpu')">Test CPU</button>
                <button onclick="runBenchmark('/api/run-gpu')" style="background: rgba(118, 185, 0, 0.2); border-color: rgba(118, 185, 0, 0.4);">Test GPU</button>
                <button class="btn-primary" onclick="runBenchmark('/api/run-all')">&#9658; Full Benchmark</button>
            </div>
        </header>

        <!-- Live Stress Countdown Strip -->
        <div id="stress-box" class="stress-banner">
            <div>
                <div style="font-weight: 700; color: #f87171; font-size: 1.1rem;" id="stress-title">STRESS TESTING IN PROGRESS</div>
                <div style="color: var(--text-dim); margin-top: 4px;" id="stress-details">Active Load Generation...</div>
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div class="countdown-timer" id="stress-countdown">00:00</div>
                <button class="btn-stop" onclick="stopStress()">⏹ Stop Test</button>
            </div>
        </div>

        <!-- Telemetry Strip -->
        <div class="telemetry-strip">
            <div class="tel-card">
                <div class="tel-title">CPU Load</div>
                <div id="tel-cpu-util" class="tel-val" style="color: var(--accent-cpu);">0%</div>
                <div class="progress-bar-bg"><div id="bar-cpu" class="progress-bar-fill" style="width: 0%; background: var(--accent-cpu);"></div></div>
                <div id="tel-cpu-freq" class="tel-sub">0 MHz Frequency</div>
            </div>
            <div class="tel-card">
                <div class="tel-title">System Memory (RAM)</div>
                <div id="tel-ram-util" class="tel-val">0%</div>
                <div class="progress-bar-bg"><div id="bar-ram" class="progress-bar-fill" style="width: 0%; background: #38bdf8;"></div></div>
                <div id="tel-ram-sub" class="tel-sub">0 / 0 GB</div>
            </div>
            <div class="tel-card">
                <div class="tel-title">GPU Load</div>
                <div id="tel-gpu-util" class="tel-val" style="color: var(--accent-gpu);">0%</div>
                <div class="progress-bar-bg"><div id="bar-gpu" class="progress-bar-fill" style="width: 0%; background: var(--accent-gpu);"></div></div>
                <div id="tel-gpu-clock" class="tel-sub">0 MHz Clock</div>
            </div>
            <div class="tel-card">
                <div class="tel-title">GPU VRAM</div>
                <div id="tel-vram-util" class="tel-val">0 MB</div>
                <div class="progress-bar-bg"><div id="bar-vram" class="progress-bar-fill" style="width: 0%; background: var(--accent-gpu);"></div></div>
                <div id="tel-vram-sub" class="tel-sub">0% Used</div>
            </div>
            <div class="tel-card">
                <div class="tel-title">GPU Thermals & Power</div>
                <div id="tel-gpu-temp" class="tel-val" style="color: var(--accent-gold);">0 &deg;C</div>
                <div id="tel-gpu-power" class="tel-sub">0.0 W Power Draw</div>
            </div>
        </div>

        <!-- Scoring & Tier Banner -->
        <div class="panel" style="margin-bottom: 25px; border-left: 6px solid #ffb703;">
            <div class="panel-header">
                <div>
                    <div class="panel-title" style="color: var(--accent-gold);">System Performance Scoring & Stability</div>
                    <div id="score-tier" style="font-size: 1.1rem; font-weight: 700; color: #fff; margin-top: 4px;">Click "Quick Benchmark" or Run a Stress Test to evaluate system score</div>
                </div>
                <div id="score-system-pts" style="font-size: 2.4rem; font-weight: 800; color: var(--accent-gold);">-- PTS</div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px;">
                    <div style="font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase;">CPU Score</div>
                    <div id="score-cpu-pts" style="font-size: 1.5rem; font-weight: 700; color: var(--accent-cpu); margin-top: 2px;">--</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px;">
                    <div style="font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase;">GPU Score</div>
                    <div id="score-gpu-pts" style="font-size: 1.5rem; font-weight: 700; color: var(--accent-gpu); margin-top: 2px;">--</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px;">
                    <div style="font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase;">Stability / Thermal Rating</div>
                    <div id="score-stability" style="font-size: 1.5rem; font-weight: 700; color: #ffb703; margin-top: 2px;">--</div>
                </div>
            </div>
        </div>

        <div class="grid-2">
            <!-- CPU Section -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title" style="color: var(--accent-cpu);">CPU Specifications & Results</div>
                    <span id="badge-cpu-score" style="font-weight: 700; color: var(--accent-cpu);">--</span>
                </div>
                <div class="spec-item"><span class="spec-label">Processor</span><span id="spec-cpu-name" class="spec-val">Loading...</span></div>
                <div class="spec-item"><span class="spec-label">Cores / Threads</span><span id="spec-cpu-cores" class="spec-val">--</span></div>
                <div class="spec-item"><span class="spec-label">Single-Core Arithmetic</span><span id="res-cpu-single" class="spec-val">--</span></div>
                <div class="spec-item"><span class="spec-label">Multi-Core GEMM</span><span id="res-cpu-gemm" class="spec-val">--</span></div>
                <div class="spec-item"><span class="spec-label">RAM Bandwidth</span><span id="res-cpu-ram-bw" class="spec-val">--</span></div>
                <div class="spec-item"><span class="spec-label">SHA-256 Hashing</span><span id="res-cpu-sha" class="spec-val">--</span></div>
            </div>

            <!-- GPU Section -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title" style="color: var(--accent-gpu);">GPU Specifications & Results</div>
                    <span id="badge-gpu-score" style="font-weight: 700; color: var(--accent-gpu);">--</span>
                </div>
                <div class="spec-item"><span class="spec-label">Graphics Card</span><span id="spec-gpu-name" class="spec-val">Loading...</span></div>
                <div class="spec-item"><span class="spec-label">VRAM Capacity</span><span id="spec-gpu-vram" class="spec-val">--</span></div>
                <div class="spec-item"><span class="spec-label">Native CUDA Compute</span><span id="res-gpu-cuda" class="spec-val" style="color: var(--accent-gpu);">--</span></div>
                <div class="spec-item"><span class="spec-label">VRAM Bandwidth</span><span id="res-gpu-vram-bw" class="spec-val">--</span></div>
                <div class="spec-item"><span class="spec-label">PCIe Upload (HtoD)</span><span id="res-gpu-htod" class="spec-val">--</span></div>
                <div class="spec-item"><span class="spec-label">4K Vision Throughput</span><span id="res-gpu-vision" class="spec-val">--</span></div>
            </div>
        </div>
    </div>

    <script>
        async function fetchInfo() {
            try {
                const res = await fetch('/api/info');
                const data = await res.json();
                document.getElementById('spec-cpu-name').innerText = data.cpu.name;
                document.getElementById('spec-cpu-cores').innerText = `${data.cpu.physical_cores} Physical / ${data.cpu.logical_cores} Logical`;
                document.getElementById('spec-gpu-name').innerText = data.gpu.name;
                document.getElementById('spec-gpu-vram').innerText = `${data.gpu.total_vram_mb} MB`;
            } catch(e) {}
        }

        async function updateTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                const d = await res.json();
                
                document.getElementById('tel-cpu-util').innerText = `${d.cpu_util_pct}%`;
                document.getElementById('bar-cpu').style.width = `${d.cpu_util_pct}%`;
                document.getElementById('tel-cpu-freq').innerText = `${d.cpu_freq_mhz} MHz Frequency`;

                document.getElementById('tel-ram-util').innerText = `${d.ram_pct}%`;
                document.getElementById('bar-ram').style.width = `${d.ram_pct}%`;
                document.getElementById('tel-ram-sub').innerText = `${d.ram_used_gb} / ${d.ram_total_gb} GB`;

                if (d.gpu && d.gpu.available) {
                    document.getElementById('tel-gpu-util').innerText = `${d.gpu.gpu_util_pct}%`;
                    document.getElementById('bar-gpu').style.width = `${d.gpu.gpu_util_pct}%`;
                    document.getElementById('tel-gpu-clock').innerText = `${d.gpu.clock_mhz} MHz Clock`;

                    document.getElementById('tel-vram-util').innerText = `${d.gpu.vram_used_mb} MB`;
                    document.getElementById('bar-vram').style.width = `${d.gpu.vram_pct}%`;
                    document.getElementById('tel-vram-sub').innerText = `${d.gpu.vram_pct}% of ${d.gpu.vram_total_mb} MB`;

                    document.getElementById('tel-gpu-temp').innerHTML = `${d.gpu.temp_c} &deg;C`;
                    document.getElementById('tel-gpu-power').innerText = `${d.gpu.power_w} W Power Draw`;
                }
            } catch(e) {}
        }

        async function checkStressStatus() {
            try {
                const res = await fetch('/api/stress/status');
                const s = await res.json();
                const box = document.getElementById('stress-box');

                if (s.is_running) {
                    box.style.display = 'flex';
                    const m = Math.floor(s.remaining_sec / 60);
                    const sec = Math.floor(s.remaining_sec % 60);
                    document.getElementById('stress-countdown').innerText = `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
                    document.getElementById('stress-title').innerText = `STRESS TESTING: ${s.type.toUpperCase()} IN PROGRESS`;
                    document.getElementById('stress-details').innerText = `Elapsed: ${s.elapsed_sec}s / Target: ${s.duration_sec}s`;
                } else {
                    box.style.display = 'none';
                    if (s.last_result && s.last_result.stability_score !== undefined) {
                        document.getElementById('score-stability').innerText = `${s.last_result.stability_score}% (${s.last_result.thermal_throttling_detected ? 'Throttled' : 'Rock Solid'})`;
                    }
                }
            } catch(e) {}
        }

        async function startStress(endpoint) {
            const mins = parseFloat(document.getElementById('timer-minutes').value);
            try {
                const res = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ duration_minutes: mins })
                });
                if (!res.ok) {
                    const txt = await res.text();
                    throw new Error(`Server status ${res.status}: ${txt}`);
                }
            } catch(e) {
                alert('Failed to start stress test: ' + e.message);
            }
        }

        async function stopStress() {
            try {
                await fetch('/api/stress/stop', { method: 'POST' });
            } catch(e) {}
        }

        async function runBenchmark(endpoint = '/api/run-all') {
            const btns = document.querySelectorAll('button');
            btns.forEach(b => b.disabled = true);
            try {
                const res = await fetch(endpoint, { method: 'POST' });
                if (!res.ok) {
                    let errMsg = `Server returned status ${res.status}`;
                    try {
                        const errJson = await res.json();
                        errMsg = errJson.error || errJson.detail || errMsg;
                    } catch(err) {
                        errMsg = await res.text();
                    }
                    throw new Error(errMsg);
                }
                const d = await res.json();
                
                if (d.scores) {
                    if (d.scores.system_score !== undefined) document.getElementById('score-system-pts').innerText = `${d.scores.system_score} PTS`;
                    if (d.scores.tier) document.getElementById('score-tier').innerText = d.scores.tier;
                    if (d.scores.cpu_score !== undefined) document.getElementById('score-cpu-pts').innerText = `${d.scores.cpu_score} PTS`;
                    if (d.scores.gpu_score !== undefined) document.getElementById('score-gpu-pts').innerText = `${d.scores.gpu_score} PTS`;
                }

                if (d.cpu) {
                    document.getElementById('res-cpu-single').innerText = `${d.cpu.single_core.throughput_mflops} MFLOPS`;
                    document.getElementById('res-cpu-gemm').innerText = `${d.cpu.multi_core_gemm.throughput_gflops} GFLOPS`;
                    document.getElementById('res-cpu-ram-bw').innerText = `${d.cpu.memory_bandwidth.avg_bandwidth_gbps} GB/s`;
                    document.getElementById('res-cpu-sha').innerText = `${d.cpu.hashing.sha256_mbps} MB/s`;
                }

                if (d.gpu && d.gpu.available) {
                    document.getElementById('res-gpu-cuda').innerText = `${d.gpu.cuda_compute.throughput_tflops} TFLOPS`;
                    document.getElementById('res-gpu-vram-bw').innerText = `${d.gpu.memory_bandwidth.device_to_device_vram_gbps} GB/s`;
                    document.getElementById('res-gpu-htod').innerText = `${d.gpu.memory_bandwidth.host_to_device_gbps} GB/s`;
                    document.getElementById('res-gpu-vision').innerText = `${d.gpu.opencl_vision.megapixels_per_sec} MP/s`;
                }
            } catch(e) {
                alert('Benchmark error: ' + e.message);
            } finally {
                btns.forEach(b => b.disabled = false);
            }
        }

        fetchInfo();
        setInterval(updateTelemetry, 1000);
        setInterval(checkStressStatus, 1000);
    </script>
</body>
</html>"""

    return app


# =====================================================================
# CLI Pretty Printer & Formatter
# =====================================================================

class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def print_banner():
    print(f"""
{Colors.CYAN}{Colors.BOLD}======================================================================
         HARDWARE PERFORMANCE & STRESS TESTING SYSTEM (CPU & GPU)
======================================================================{Colors.RESET}""")

def run_cli_stress(mode, duration_sec, telemetry: HardwareTelemetry):
    print_banner()
    duration_min = round(duration_sec / 60.0, 2)
    print(f"{Colors.YELLOW}{Colors.BOLD}[!] STARTING SUSTAINED STRESS TEST{Colors.RESET}")
    print(f"  Mode: {Colors.BOLD}{mode.upper()}{Colors.RESET} | Timer Duration: {Colors.BOLD}{duration_sec}s ({duration_min} minutes){Colors.RESET}")
    print("=" * 70)

    last_print_time = [0]
    def progress_callback(data):
        now = time.time()
        if now - last_print_time[0] >= 1.0:
            last_print_time[0] = now
            elapsed = data.get("elapsed_sec", 0)
            rem = data.get("remaining_sec", 0)
            temp = data.get("temp_c", data.get("gpu_temp_c", "N/A"))
            pwr = data.get("power_w", data.get("gpu_power_w", "N/A"))
            clk = data.get("clock_mhz", "N/A")
            cpu_l = data.get("cpu_load_pct", "N/A")
            sys.stdout.write(f"\r  [{elapsed:.0f}s / {duration_sec}s] Rem: {rem:.0f}s | CPU Load: {cpu_l}% | GPU Temp: {temp} C | Power: {pwr} W | Clock: {clk} MHz   ")
            sys.stdout.flush()

    if mode == "cpu":
        cpu_bench = CPUBenchmark()
        res = cpu_bench.run_stress_test(duration_sec, progress_callback=progress_callback)
        print("\n\n" + "=" * 70)
        print(f"{Colors.CYAN}{Colors.BOLD}CPU STRESS TEST COMPLETE:{Colors.RESET}")
        print(f"  - Threads Saturated:           {res['logical_threads_used']}")
        print(f"  - Matrix Computations:         {res['total_matrix_operations']:,} GEMM Ops")
        print(f"  - Sustained Compute:           {res['sustained_gflops']} GFLOPS")
        print(f"  - Average CPU Load:            {res['avg_cpu_load_pct']}%")
        print(f"  - Frequency (Base/Avg/Min):    {res['baseline_freq_mhz']} / {res['avg_freq_mhz']} / {res['min_freq_mhz']} MHz")
        print(f"  - Clock Throttling Detected:   {res['thermal_throttling_detected']} ({res['cpu_clock_drop_pct']}% drop)")
        print(f"  {Colors.BOLD}>> CPU Stability Rating:       {res['stability_score']}%{Colors.RESET}")

    elif mode == "gpu":
        gpu_bench = GPUBenchmark(telemetry)
        res = gpu_bench.run_stress_test(duration_sec, progress_callback=progress_callback)
        print("\n\n" + "=" * 70)
        print(f"{Colors.GREEN}{Colors.BOLD}GPU STRESS TEST COMPLETE:{Colors.RESET}")
        print(f"  - Kernel Launches:             {res['kernel_launches']}")
        print(f"  - Sustained Compute:           {res['sustained_tflops']} TFLOPS")
        print(f"  - Temperature (Start/Peak):    {res['initial_temp_c']} C -> {res['peak_temp_c']} C (+{res['delta_temp_c']} C)")
        print(f"  - Power (Avg/Peak):            {res['avg_power_w']} W -> {res['peak_power_w']} W")
        print(f"  - Graphics Clock (Max/Min):    {res['max_clock_mhz']} / {res['min_clock_mhz']} MHz ({res['gpu_clock_drop_pct']}% drop)")
        print(f"  - Thermal Throttling Detected: {res['thermal_throttling_detected']}")
        print(f"  {Colors.BOLD}>> GPU Stability Rating:       {res['stability_score']}%{Colors.RESET}")

    elif mode == "both":
        res = run_combined_stress_test(duration_sec, telemetry, progress_callback=progress_callback)
        print("\n\n" + "=" * 70)
        print(f"{Colors.RED}{Colors.BOLD}COMBINED MAXIMUM LOAD STRESS COMPLETE:{Colors.RESET}")
        print(f"  - Peak GPU Temp:               {res['peak_gpu_temp_c']} C")
        print(f"  - Peak GPU Power Draw:         {res['peak_gpu_power_w']} W")
        print(f"  - CPU Stability:               {res['cpu_stress']['stability_score']}%")
        print(f"  - GPU Stability:               {res['gpu_stress']['stability_score']}%")
        print(f"  {Colors.YELLOW}{Colors.BOLD}>> Overall System Stability:   {res['combined_stability_score']}%{Colors.RESET}")

def run_cli_benchmark(args, telemetry: HardwareTelemetry):
    print_banner()

    cpu_info = telemetry.get_cpu_info()
    gpu_info = telemetry.get_gpu_info()

    print(f"{Colors.BOLD}HARDWARE DETECTED:{Colors.RESET}")
    print(f"  {Colors.CYAN}CPU:{Colors.RESET} {cpu_info['name']}")
    print(f"       {cpu_info['physical_cores']} Physical Cores, {cpu_info['logical_cores']} Threads @ {cpu_info['current_freq_mhz']} MHz | RAM: {cpu_info['total_ram_gb']} GB")
    print(f"  {Colors.GREEN}GPU:{Colors.RESET} {gpu_info.get('name', 'None')}")
    print(f"       VRAM: {gpu_info.get('total_vram_mb', 0)} MB | Driver: {gpu_info.get('driver_version', 'N/A')} | Backend: {gpu_info.get('backend', 'N/A')}")
    print("=" * 70)

    cpu_results = {}
    gpu_results = {}

    run_cpu = not args.gpu_only
    run_gpu = not args.cpu_only

    if run_cpu:
        print(f"\n{Colors.CYAN}{Colors.BOLD}>>> RUNNING CPU BENCHMARK SUITE...{Colors.RESET}")
        cpu_bench = CPUBenchmark()
        cpu_results = cpu_bench.run_all(verbose_callback=lambda msg: print(f"  [*] {msg}"))

        print(f"\n  {Colors.BOLD}CPU Results:{Colors.RESET}")
        print(f"    - Single-Core Arithmetic:     {cpu_results['single_core']['throughput_mflops']} MFLOPS")
        print(f"    - Multi-Core GEMM (2048x2048):{cpu_results['multi_core_gemm']['throughput_gflops']} GFLOPS ({cpu_results['multi_core_gemm']['avg_time_ms']} ms)")
        print(f"    - RAM Read Bandwidth:         {cpu_results['memory_bandwidth']['read_bandwidth_gbps']} GB/s")
        print(f"    - RAM Write Bandwidth:        {cpu_results['memory_bandwidth']['write_bandwidth_gbps']} GB/s")
        print(f"    - SHA-256 Hashing:            {cpu_results['hashing']['sha256_mbps']} MB/s")

    if run_gpu:
        print(f"\n{Colors.GREEN}{Colors.BOLD}>>> RUNNING GPU BENCHMARK SUITE...{Colors.RESET}")
        gpu_bench = GPUBenchmark(telemetry)
        gpu_results = gpu_bench.run_all(verbose_callback=lambda msg: print(f"  [*] {msg}"))

        if gpu_results.get("available"):
            print(f"\n  {Colors.BOLD}GPU Results:{Colors.RESET}")
            print(f"    - Native CUDA FP32 Peak:      {gpu_results['cuda_compute']['throughput_tflops']} TFLOPS ({gpu_results['cuda_compute']['throughput_gflops']} GFLOPS)")
            print(f"    - PCIe Host to Device:        {gpu_results['memory_bandwidth']['host_to_device_gbps']} GB/s")
            print(f"    - PCIe Device to Host:        {gpu_results['memory_bandwidth']['device_to_host_gbps']} GB/s")
            print(f"    - VRAM Internal Bandwidth:    {gpu_results['memory_bandwidth']['device_to_device_vram_gbps']} GB/s")
            if "opencl_vision" in gpu_results and gpu_results["opencl_vision"].get("fps"):
                print(f"    - 4K Vision Acceleration:     {gpu_results['opencl_vision']['megapixels_per_sec']} MP/s ({gpu_results['opencl_vision']['fps']} FPS)")

    # Scoring calculation
    scores = calculate_hardware_scores(cpu_results, gpu_results)
    comparison = compute_hardware_comparison(cpu_results, gpu_results)

    print(f"\n{Colors.MAGENTA}{Colors.BOLD}======================================================================{Colors.RESET}")
    print(f"{Colors.MAGENTA}{Colors.BOLD}                   SYSTEM HARDWARE SCORECARD                          {Colors.RESET}")
    print(f"{Colors.MAGENTA}{Colors.BOLD}======================================================================{Colors.RESET}")
    print(f"  CPU Score:     {Colors.CYAN}{scores['cpu_score']:,} Points{Colors.RESET}")
    print(f"  GPU Score:     {Colors.GREEN}{scores['gpu_score']:,} Points{Colors.RESET}")
    print(f"  System Score:  {Colors.YELLOW}{Colors.BOLD}{scores['system_score']:,} Points{Colors.RESET}")
    print(f"  Hardware Tier: {Colors.BOLD}{scores['tier']}{Colors.RESET}")
    print("=" * 70)

    if run_cpu and run_gpu and gpu_results.get("available"):
        for m in comparison.get("metrics", []):
            print(f"  * {m['category']:<28} CPU: {m['cpu_value']:<15} GPU: {m['gpu_value']:<25} -> {Colors.BOLD}{m['speedup']}{Colors.RESET}")
        print(f"\n  {Colors.BOLD}Compute Advantage:{Colors.RESET} GPU is {Colors.YELLOW}{Colors.BOLD}{comparison.get('compute_speedup_x', 'N/A')}x faster{Colors.RESET} for compute workloads.")
        print(f"  {Colors.BOLD}Memory Advantage:{Colors.RESET}  VRAM has {Colors.YELLOW}{Colors.BOLD}{comparison.get('bandwidth_speedup_x', 'N/A')}x higher bandwidth{Colors.RESET} than System RAM.")

    html_file = generate_html_report(cpu_info, gpu_info, cpu_results, gpu_results, comparison, scores, output_file=args.html or "benchmark_report.html")
    print(f"\n[+] Interactive HTML Report saved to: {html_file}")

    if args.json:
        full_data = {
            "timestamp": time.time(),
            "cpu_info": cpu_info,
            "gpu_info": gpu_info,
            "cpu_results": cpu_results,
            "gpu_results": gpu_results,
            "comparison": comparison,
            "scores": scores
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2)
        print(f"[+] JSON benchmark data saved to: {args.json}")


# =====================================================================
# Main Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="CPU & GPU Performance Testing & Stress Suite")
    parser.add_argument("--cpu-only", action="store_true", help="Run only CPU benchmarks")
    parser.add_argument("--gpu-only", action="store_true", help="Run only GPU benchmarks")

    # Stress testing and timer options
    parser.add_argument("--stress-cpu", action="store_true", help="Run sustained CPU stress test")
    parser.add_argument("--stress-gpu", action="store_true", help="Run sustained GPU stress test")
    parser.add_argument("--stress-both", action="store_true", help="Run combined maximum CPU + GPU stress test")
    parser.add_argument("--minutes", "-m", type=float, default=None, help="Timer duration in minutes for stress test (e.g. 1.0, 2.5, 5.0)")
    parser.add_argument("--seconds", "-s", type=int, default=None, help="Timer duration in seconds for stress test")

    # Output and web
    parser.add_argument("--html", type=str, default=None, help="Path to export standalone HTML report")
    parser.add_argument("--json", type=str, default=None, help="Path to export JSON results")
    parser.add_argument("--web", action="store_true", help="Launch interactive Web Dashboard")
    parser.add_argument("--port", type=int, default=8585, help="Port for Web Dashboard (default 8585)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for Web Dashboard (default 127.0.0.1)")

    args = parser.parse_args()
    telemetry = HardwareTelemetry()

    try:
        # Determine duration
        duration_sec = 60
        if args.seconds:
            duration_sec = args.seconds
        elif args.minutes:
            duration_sec = int(args.minutes * 60)

        if args.stress_cpu:
            run_cli_stress("cpu", duration_sec, telemetry)
        elif args.stress_gpu:
            run_cli_stress("gpu", duration_sec, telemetry)
        elif args.stress_both:
            run_cli_stress("both", duration_sec, telemetry)
        elif args.web:
            import uvicorn
            print_banner()
            print(f"{Colors.GREEN}[*] Launching Interactive Web Dashboard at: http://{args.host}:{args.port}{Colors.RESET}")
            app = create_fastapi_app(telemetry)
            uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        else:
            run_cli_benchmark(args, telemetry)
    finally:
        telemetry.close()

if __name__ == "__main__":
    main()
