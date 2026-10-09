# ⚡ CPU & GPU Hardware Performance & Stress Testing Suite

[![Live Demo](https://img.shields.io/badge/Live_Report-GitHub_Pages-2ea44f?style=for-the-badge&logo=github)](https://anurag2005930.github.io/cpu-gpu-benchmark/)
[![GitHub](https://img.shields.io/badge/Repository-cpu--gpu--benchmark-blue?style=for-the-badge&logo=github)](https://github.com/anurag2005930/cpu-gpu-benchmark)

**Live Web Report:** [https://anurag2005930.github.io/cpu-gpu-benchmark/](https://anurag2005930.github.io/cpu-gpu-benchmark/)

A high-performance hardware testing, benchmarking, and sustained thermal stress testing tool for CPUs and GPUs on Windows. Built with direct **Native CUDA Driver API** integration and **NVIDIA NVML** hardware telemetry for sub-millisecond readings with zero external SDK overhead.

---

## 🚀 Key Features

- **Direct Native CUDA Driver API (`nvcuda.dll`)**: Runs raw compute kernels directly via CUDA PTX without requiring heavy GB-sized SDKs or PyTorch.
- **NVML Telemetry (`nvml.dll`)**: Sub-millisecond GPU temperature (°C), power draw (Watts), graphics clock (MHz), and VRAM utilization tracking.
- **Configurable Timer Load Testing**: Sustained stress testing with a custom timer (seconds / minutes) for stability, cooling, and throttling analysis.
- **Comprehensive CPU Benchmark Suite**:
  - Single-Core floating-point arithmetic & Mandelbrot generation.
  - Multi-Core GEMM (matrix multiplication) GFLOPS using multi-threading.
  - System RAM sequential read & write bandwidth (GB/s).
  - SHA-256 and MD5 cryptographic hashing throughput (MB/s).
- **Comprehensive GPU Benchmark Suite**:
  - Native CUDA FP32 peak compute (TFLOPS / GFLOPS).
  - PCIe Host-to-Device (upload) & Device-to-Host (download) speeds.
  - Internal GDDR VRAM copy bandwidth (GB/s).
  - 4K resolution OpenCL image filtering acceleration (FPS & MP/s).
- **Advanced Scoring & Tier Evaluation**:
  - Balanced point-based scoring for CPU, GPU, and Total System.
  - Hardware Tier Classification (**Tier S+**, **Tier S**, **Tier A**, **Tier B**, **Tier C**).
  - Thermal Headroom & Clock Stability Rating (0 – 100%).
- **Interactive Modern Web Dashboard**:
  - Live animated gauges, telemetry dials, and real-time stress testing graphs via FastAPI & SSE.
- **Standalone HTML Report**:
  - Automatically exports a self-contained, responsive report (`benchmark_report.html`).

---

## 📦 Installation

Ensure you have Python 3.10+ installed on your system.

```bash
git clone https://github.com/anurag2005930/cpu-gpu-benchmark.git
cd cpu-gpu-benchmark
pip install -r requirements.txt
```

---

## 🛠️ Usage

### 1. Interactive Menu (Windows)
Double-click `run_tester.bat` or run:
```bat
run_tester.bat
```

### 2. Command Line Benchmarking
- **Full Benchmark (CPU + GPU + Hardware Tier)**:
  ```bash
  python cpu_gpu_tester.py
  ```
- **CPU Benchmark Only**:
  ```bash
  python cpu_gpu_tester.py --cpu-only
  ```
- **GPU Benchmark Only**:
  ```bash
  python cpu_gpu_tester.py --gpu-only
  ```

### 3. Sustained Stress Testing with Timer
- **GPU Stress Test (e.g. 2 minutes)**:
  ```bash
  python cpu_gpu_tester.py --stress-gpu --minutes 2
  ```
- **CPU Stress Test (e.g. 5 minutes)**:
  ```bash
  python cpu_gpu_tester.py --stress-cpu --minutes 5
  ```
- **Max Power Load (Both CPU + GPU simultaneously)**:
  ```bash
  python cpu_gpu_tester.py --stress-both --minutes 1
  ```

### 4. Interactive Web Dashboard
```bash
python cpu_gpu_tester.py --web --port 8585
```
Open [http://127.0.0.1:8585](http://127.0.0.1:8585) in your web browser.

---

## 📊 Sample Hardware Scorecard

```
======================================================================
                   SYSTEM HARDWARE SCORECARD                          
======================================================================
  CPU Score:     6,260 Points
  GPU Score:     12,787 Points
  System Score:  10,176 Points
  Hardware Tier: Tier S (High-End Gaming & Pro Compute)
======================================================================
  * Raw Floating-Point Compute   CPU: 233.32 GFLOPS   GPU: 6808.79 GFLOPS (6.81 TFLOPS) -> 29.18x Faster on GPU
  * Internal Memory Bandwidth    CPU: 5.42 GB/s (RAM) GPU: 90.12 GB/s (VRAM)            -> 16.63x Higher on GPU
```

---

## 📄 License
MIT License. Feel free to use and modify for your hardware testing needs.
