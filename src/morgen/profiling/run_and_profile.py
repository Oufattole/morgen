from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The pynvml package is deprecated",
            category=FutureWarning,
        )
        import pynvml
except ImportError:  # pragma: no cover - optional dependency at runtime
    pynvml = None


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _format_timestamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class Sample:
    elapsed_seconds: float
    cpu_rss_bytes: int
    gpu_memory_bytes: int | None
    gpu_indices: list[int]
    process_count: int


class NVMLTracker:
    def __init__(self) -> None:
        self._available = False
        self._error: str | None = None
        self._device_count = 0

        if pynvml is None:
            self._error = "pynvml is not installed"
            return

        try:
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._available = True
        except Exception as exc:  # pragma: no cover - depends on host GPU setup
            self._error = str(exc)

    @property
    def error(self) -> str | None:
        return self._error

    def shutdown(self) -> None:
        if not self._available:
            return
        try:
            pynvml.nvmlShutdown()
        except Exception:  # pragma: no cover - defensive cleanup
            pass

    def get_gpu_memory_bytes(self, tracked_pids: set[int]) -> tuple[int | None, list[int]]:
        if not self._available:
            return None, []

        total_bytes = 0
        gpu_indices: list[int] = []

        for device_index in range(self._device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            processes = self._get_device_processes(handle)
            device_bytes = sum(proc.usedGpuMemory for proc in processes if proc.pid in tracked_pids)
            if device_bytes > 0:
                gpu_indices.append(device_index)
                total_bytes += int(device_bytes)

        return total_bytes, gpu_indices

    @staticmethod
    def _get_device_processes(handle: Any) -> list[Any]:
        getter_names = (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        )

        for getter_name in getter_names:
            getter = getattr(pynvml, getter_name, None)
            if getter is None:
                continue
            try:
                return list(getter(handle))
            except pynvml.NVMLError_NotSupported:
                return []
            except pynvml.NVMLError:
                continue

        return []


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single Morgen stage command and record wall time, CPU RSS, and GPU memory.",
    )
    parser.add_argument("--stage", required=True, help="Logical stage name, e.g. pretrain, fit_em, inference.")
    parser.add_argument("--dataset", default=None, help="Optional dataset label to store in the metrics summary.")
    parser.add_argument(
        "--metrics-dir",
        required=True,
        type=Path,
        help="Directory where profiling outputs should be written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name prefix for the generated metrics files.",
    )
    parser.add_argument(
        "--sample-every",
        type=float,
        default=1.0,
        help="Polling interval in seconds for memory sampling.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute. Pass it after `--`, for example `-- morgen_pretrain ...`.",
    )
    args = parser.parse_args()

    if args.sample_every <= 0:
        parser.error("--sample-every must be positive.")

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    if not args.command:
        parser.error("A command to execute is required. Pass it after `--`.")

    return args


def _collect_process_tree(root_pid: int) -> tuple[set[int], int]:
    try:
        root = psutil.Process(root_pid)
    except psutil.Error:
        return set(), 0

    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        pass

    unique_processes: dict[int, psutil.Process] = {}
    for proc in processes:
        unique_processes[proc.pid] = proc

    rss_bytes = 0
    live_pids: set[int] = set()
    for proc in unique_processes.values():
        try:
            rss_bytes += proc.memory_info().rss
            live_pids.add(proc.pid)
        except psutil.Error:
            continue

    return live_pids, rss_bytes


def _take_sample(root_pid: int, start_time: float, nvml_tracker: NVMLTracker) -> Sample:
    tracked_pids, cpu_rss_bytes = _collect_process_tree(root_pid)
    gpu_memory_bytes, gpu_indices = nvml_tracker.get_gpu_memory_bytes(tracked_pids)
    return Sample(
        elapsed_seconds=time.perf_counter() - start_time,
        cpu_rss_bytes=cpu_rss_bytes,
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_indices=gpu_indices,
        process_count=len(tracked_pids),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = _parse_args()
    metrics_dir = args.metrics_dir.expanduser().resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)

    run_stem = args.run_name or f"{args.stage}_{_format_timestamp()}"
    summary_path = metrics_dir / f"{run_stem}_summary.json"
    samples_path = metrics_dir / f"{run_stem}_samples.jsonl"

    env_snapshot = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYENV_VERSION": os.environ.get("PYENV_VERSION"),
    }

    start_wall_clock = _utc_now_iso()
    start_perf_counter = time.perf_counter()

    process = subprocess.Popen(args.command)
    nvml_tracker = NVMLTracker()

    peak_cpu_rss_bytes = 0
    peak_gpu_memory_bytes: int | None = None
    gpu_indices_seen: set[int] = set()
    sample_count = 0

    try:
        with samples_path.open("w", encoding="utf-8") as sample_file:
            while True:
                sample = _take_sample(process.pid, start_perf_counter, nvml_tracker)
                sample_count += 1
                peak_cpu_rss_bytes = max(peak_cpu_rss_bytes, sample.cpu_rss_bytes)

                if sample.gpu_memory_bytes is not None:
                    if peak_gpu_memory_bytes is None:
                        peak_gpu_memory_bytes = sample.gpu_memory_bytes
                    else:
                        peak_gpu_memory_bytes = max(peak_gpu_memory_bytes, sample.gpu_memory_bytes)

                gpu_indices_seen.update(sample.gpu_indices)

                sample_file.write(
                    json.dumps(
                        {
                            "elapsed_seconds": round(sample.elapsed_seconds, 6),
                            "cpu_rss_bytes": sample.cpu_rss_bytes,
                            "gpu_memory_bytes": sample.gpu_memory_bytes,
                            "gpu_indices": sample.gpu_indices,
                            "process_count": sample.process_count,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                sample_file.flush()

                if process.poll() is not None:
                    break

                time.sleep(args.sample_every)
    finally:
        nvml_tracker.shutdown()

    wall_time_seconds = time.perf_counter() - start_perf_counter
    exit_code = process.wait()

    summary = {
        "command": args.command,
        "dataset": args.dataset,
        "end_time": _utc_now_iso(),
        "env": env_snapshot,
        "exit_code": exit_code,
        "gpu_indices_seen": sorted(gpu_indices_seen),
        "hostname": socket.gethostname(),
        "metrics_dir": str(metrics_dir),
        "nvml_error": nvml_tracker.error,
        "peak_cpu_rss_bytes": peak_cpu_rss_bytes,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "pid": process.pid,
        "sample_count": sample_count,
        "sampling_interval_seconds": args.sample_every,
        "samples_file": str(samples_path),
        "stage": args.stage,
        "start_time": start_wall_clock,
        "summary_file": str(summary_path),
        "wall_time_seconds": round(wall_time_seconds, 6),
    }
    _write_json(summary_path, summary)

    print(f"[morgen_profile_run] Wrote summary to {summary_path}", file=sys.stderr)
    print(f"[morgen_profile_run] Wrote samples to {samples_path}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
