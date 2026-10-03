"""Time the implementations in implementations/ for each beamformer and a range of sensors.

    python benchmarks/run.py                           # writes results/benchmark.json
    python benchmarks/run.py --cases numpy_matmul torch_matmul --methods mvdr --sensors 100 1000
    python benchmarks/run.py --list

The problem: a 100 x 100 grid, 100 time windows of --window-lengths seconds at 10 Hz, 0.1-1 Hz,
a source recorded with noise. The time includes everything from the spectra to the beampowers:
the cross-spectral density matrix, its inverse or eigenvectors, and the grid search.
Each case runs in its own process; within a case, the number of sensors increases until
one run takes longer than --time-limit seconds. Compilation (numba, jax, ...) is done in
a warm-up run and not timed.
"""

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "implementations"))

# name: (module, keyword arguments, needs a GPU, label)
CASES = {
    "numpy_einsum": ("numpy_einsum", {}, False, "numpy, einsum"),
    "numpy_matmul": ("numpy_matmul", {}, False, "numpy"),
    "torch_matmul": ("torch_matmul", {}, False, "pytorch"),
    "jax_matmul_cpu": ("jax_matmul", {"device": "cpu"}, False, "jax"),
    "numba_loops": ("numba_loops", {}, False, "numba"),
    "torch_matmul_cuda": ("torch_matmul", {"device": "cuda"}, True, "pytorch"),
    "cupy_matmul": ("cupy_matmul", {}, True, "cupy"),
    "jax_matmul_gpu": ("jax_matmul", {"device": "gpu"}, True, "jax"),
}
METHODS = ["crosscorrelation", "mvdr", "music"]  # bartlett costs the same as crosscorrelation
N_WINDOWS = 100


def hardware_info():
    info = {"machine": platform.node(), "cpu_threads": os.cpu_count()}
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    info["cpu"] = next((ln.split(":", 1)[1].strip() for ln in lscpu.splitlines() if "Model name" in ln), "")
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
        ).stdout.strip()
        info["gpu"] = gpu.splitlines()[0] if gpu else None
    except FileNotFoundError:
        info["gpu"] = None
    return info


def run_case(case, window_length, method, sensors, precision, time_limit):
    """Runs in a subprocess; prints one JSON line per number of sensors."""
    from problem import make_problem

    module, kwargs, _, _ = CASES[case]
    beamform = importlib.import_module(module).beamform
    options = dict(precision=precision, method=method, **kwargs)
    beamform(**make_problem(n_sensors=5, grid_spacing=50, n_windows=3, noise_level=1.0), **options)  # warm-up
    for n_sensors in sensors:
        problem = make_problem(
            n_sensors=n_sensors, window_length=window_length, n_windows=N_WINDOWS, noise_level=1.0
        )
        runtimes = []
        for _ in range(3):
            start = time.perf_counter()
            beamform(**problem, **options)
            runtimes.append(time.perf_counter() - start)
            if runtimes[-1] > time_limit / 10:
                break  # slow: once is enough
        record = dict(
            case=case,
            method=method,
            window_length=window_length,
            n_windows=N_WINDOWS,
            n_sensors=n_sensors,
            n_gridpoints=len(problem["gridpoints"]),
            n_freqs=len(problem["omega"]),
            precision=precision,
            runtime=min(runtimes),
        )
        print(json.dumps(record), flush=True)
        if min(runtimes) > time_limit:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--sensors", nargs="*", type=int, default=[10, 30, 100, 300, 1000, 3000])
    parser.add_argument("--window-lengths", nargs="*", type=float, default=[600])
    parser.add_argument("--precision", default="single")
    parser.add_argument("--time-limit", type=float, default=60.0, help="seconds per run")
    parser.add_argument("--output", default=str(ROOT / "results" / "benchmark.json"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--_worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.list:
        for name, (_, _, gpu, label) in CASES.items():
            print(f"{name:22s} {'GPU' if gpu else 'CPU'}  {label}")
        return
    if args._worker:
        case, window_length, method = args._worker
        run_case(case, float(window_length), method, args.sensors, args.precision, args.time_limit)
        return

    output = Path(args.output)
    results = json.loads(output.read_text()) if output.exists() else dict(hardware=hardware_info(), runs=[])
    for case in args.cases:
        for method in args.methods:
            for window_length in args.window_lengths:
                # replace earlier results of the same case, method, window length and precision
                key = (case, method, window_length, args.precision)
                results["runs"] = [
                    r
                    for r in results["runs"]
                    if (r["case"], r["method"], r["window_length"], r["precision"]) != key
                ]
                cmd = [sys.executable, __file__, "--_worker", case, str(window_length), method]
                cmd += ["--precision", args.precision, "--time-limit", str(args.time_limit)]
                cmd += ["--sensors", *map(str, args.sensors)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                for line in proc.stdout.splitlines():
                    if line.startswith("{"):
                        record = json.loads(line)
                        results["runs"].append(record)
                        print(f"{case:22s} {method:17s} {window_length:5.0f} s windows "
                              f"{record['n_sensors']:5d} sensors: {record['runtime']:9.4f} s", flush=True)  # fmt: skip
                if proc.returncode != 0:
                    print(f"{case} {method}: {proc.stderr.strip().splitlines()[-1:]}", flush=True)
                output.parent.mkdir(exist_ok=True)
                output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
