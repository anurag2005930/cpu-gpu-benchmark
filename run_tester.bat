@echo off
setlocal EnableDelayedExpansion
title CPU and GPU Performance & Stress Testing System

:MENU
cls
echo ======================================================================
echo       CPU and GPU PERFORMANCE, STRESS TESTING & SCORING SYSTEM
echo ======================================================================
echo.
echo   1. Quick Full Benchmark (CPU + GPU + Scores + Hardware Tier)
echo   2. Run CPU Benchmark Only
echo   3. Run GPU Benchmark Only
echo   4. Sustained CPU Stress Test (with Custom Timer)
echo   5. Sustained GPU Stress Test (with Custom Timer)
echo   6. Sustained MAX LOAD (CPU + GPU Combined Stress Test)
echo   7. Launch Interactive Web Dashboard (Live UI & Gauges)
echo   8. Exit
echo.
set /p choice="Choose an option (1-8): "

if "%choice%"=="1" (
    cls
    python cpu_gpu_tester.py
    echo.
    echo Benchmark complete! Report saved to benchmark_report.html
    pause
    goto MENU
)
if "%choice%"=="2" (
    cls
    python cpu_gpu_tester.py --cpu-only
    echo.
    pause
    goto MENU
)
if "%choice%"=="3" (
    cls
    python cpu_gpu_tester.py --gpu-only
    echo.
    pause
    goto MENU
)
if "%choice%"=="4" (
    cls
    echo Enter stress test duration in minutes (e.g. 0.5, 1, 2, 5):
    set /p mins="Duration in minutes [default 1]: "
    if "!mins!"=="" set mins=1
    cls
    python cpu_gpu_tester.py --stress-cpu --minutes !mins!
    echo.
    pause
    goto MENU
)
if "%choice%"=="5" (
    cls
    echo Enter stress test duration in minutes (e.g. 0.5, 1, 2, 5):
    set /p mins="Duration in minutes [default 1]: "
    if "!mins!"=="" set mins=1
    cls
    python cpu_gpu_tester.py --stress-gpu --minutes !mins!
    echo.
    pause
    goto MENU
)
if "%choice%"=="6" (
    cls
    echo Enter combined max load duration in minutes (e.g. 0.5, 1, 2, 5):
    set /p mins="Duration in minutes [default 1]: "
    if "!mins!"=="" set mins=1
    cls
    python cpu_gpu_tester.py --stress-both --minutes !mins!
    echo.
    pause
    goto MENU
)
if "%choice%"=="7" (
    cls
    echo Starting Web Dashboard on http://127.0.0.1:8585...
    start http://127.0.0.1:8585
    python cpu_gpu_tester.py --web --port 8585
    pause
    goto MENU
)
if "%choice%"=="8" (
    exit /b
)

goto MENU
