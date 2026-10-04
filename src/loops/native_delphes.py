"""Standalone and subprocess-parallel orchestration for native Delphes C++.

Unlike ``adaptive_delphes``, this module does not create a Python process
pool or read events through PyROOT.  It exports a validated branch-card
manifest once, then launches the standalone C++ executable for each entry
range.  ROOT/Delphes and ``hadd`` must be available to build/run the binary.
"""
from __future__ import annotations

import argparse
try:
    from importlib.resources import files as resource_files
except ImportError:  # pragma: no cover - Python 3.8
    from importlib_resources import files as resource_files
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..core.branches_reader import BranchesHandler
from .loop_utilis import check_loop_args


def _native_binary() -> Path:
    configured = os.environ.get("HEPDATASET_NATIVE_EXECUTABLE")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parent / "native_cpp" / "native_adaptive_delphes"


def _run_checked(command: list[str], *, env: dict[str, str], capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=True,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    rootsys = env.get("ROOTSYS")
    if rootsys:
        bindir = str(Path(rootsys) / "bin")
        libdir = str(Path(rootsys) / "lib")
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        wheel_private_libs = Path(rootsys).resolve().parent / "root.libs"
        if wheel_private_libs.is_dir():
            env["LD_LIBRARY_PATH"] = str(wheel_private_libs) + os.pathsep + env["LD_LIBRARY_PATH"]
    delphes_dir = env.get("DELPHES_DIR")
    if delphes_dir:
        env["LD_LIBRARY_PATH"] = str(Path(delphes_dir).resolve()) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        exroot_include = str(Path(delphes_dir).resolve() / "external")
        env["ROOT_INCLUDE_PATH"] = exroot_include + os.pathsep + env.get("ROOT_INCLUDE_PATH", "")
    return env


def _ensure_native_binary() -> tuple[Path, dict[str, str]]:
    binary = _native_binary()
    env = _runtime_env()
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary, env
    make = shutil.which("make")
    if not make:
        raise RuntimeError("The native C++ executable is not built and `make` is unavailable. See README native-loop build instructions.")
    if not env.get("DELPHES_DIR"):
        raise RuntimeError("Set DELPHES_DIR to a Delphes source/build directory before building the native loop.")
    _run_checked([make, "-C", str(Path(__file__).resolve().parent / "native_cpp"), "native_adaptive_delphes"], env=env)
    if not binary.is_file():
        raise RuntimeError(f"The native build completed but did not create {binary}")
    return binary, env


def _count_entries(binary: Path, env: dict[str, str], input_root_file: str) -> int:
    result = _run_checked([str(binary), "--count-entries", str(input_root_file)], env=env, capture=True)
    line = next((x for x in result.stdout.splitlines() if x.startswith("TOTAL_ENTRIES=")), None)
    if line is None:
        raise RuntimeError("Native executable did not return its ROOT entry count")
    return int(line.split("=", 1)[1])


def _channel_keys(split_by_flavour: bool = False) -> list[str]:
    if split_by_flavour:
        channels = {
            "0L": ["Nothing", "J", "JJ", "JJJ"],
            "1L": ["e", "mu", "eJ", "muJ", "eJJ", "muJJ"],
            "2OSL": ["ee", "emu", "mumu", "eeJ", "emuJ", "mumuJ"],
            "2SSL": ["ee", "emu", "mumu", "eeJ", "emuJ", "mumuJ"],
            "3L": ["eee", "eemu", "emumu", "mumumu"],
        }
    else:
        channels = {
            "0L": ["Nothing", "J", "JJ", "JJJ"],
            "1L": ["lep", "lepJ", "lepJJ"],
            "2OSL": ["leplep", "leplepJ"],
            "2SSL": ["leplep", "leplepJ"],
            "3L": ["lepleplep"],
        }
    return [f"{region}_{category}" for region, categories in channels.items() for category in categories]


def _native_command(binary: Path, manifest_path: str, args: dict[str, Any], *,
                    output_dir: str, output_file_name: str, start: int, end: int,
                    event_weight: float, overwrite: bool) -> list[str]:
    command = [
        str(binary), "--input-root-file", str(args["inputRootFile"]),
        "--branch-manifest", str(manifest_path), "--output-dir", output_dir,
        "--output-file-name", output_file_name, "--tree-name", str(args["treeName"]),
        "--start-entry", str(start), "--end-entry", str(end),
        "--event-weight", str(event_weight), "--luminosity", str(args["luminosity"] or 400.0),
    ]
    if args["show_progress"]:
        command.append("--show-progress")
    if args.get("split_by_flavour", False):
        command.append("--split-by-flavour")
    if overwrite:
        command.append("--overwrite")
    return command


def _hadd(env: dict[str, str], target: Path, source_files: list[Path], max_workers: int | None) -> None:
    if not source_files:
        raise RuntimeError(f"No ROOT chunk files were produced for {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    hadd = shutil.which("hadd", path=env.get("PATH"))
    if not hadd and env.get("ROOTSYS"):
        candidate = Path(env["ROOTSYS"]) / "bin" / "hadd"
        if candidate.is_file():
            hadd = str(candidate)
    if not hadd:
        raise RuntimeError("Cannot merge native C++ chunks: ROOT `hadd` was not found in PATH or ROOTSYS/bin")
    command = [hadd, "-f"]
    if max_workers and max_workers > 1:
        command += ["-j", str(max_workers)]
    _run_checked(command + [str(target)] + [str(path) for path in source_files], env=env)


def run_native_in_parallel(
    max_workers: int | None = None,
    n_chunks: int | None = None,
    **loop_kwargs: Any,
) -> dict[str, Any]:
    """Run native C++ event loop chunks as direct subprocesses and hadd outputs."""
    args = dict(loop_kwargs)
    args.setdefault("treeName", "Delphes")
    args.setdefault("eventWeight", None)
    args.setdefault("cross_section", None)
    args.setdefault("luminosity", None)
    args.setdefault("start_entry", 0)
    args.setdefault("end_entry", None)
    args.setdefault("output_dir", ".")
    args.setdefault("output_file_name", "events.root")
    args.setdefault("show_progress", False)
    args.setdefault("debug_loop", False)
    args.setdefault("overwrite", False)
    args.setdefault("branches_config_path", None)
    args.setdefault("split_by_flavour", False)
    if not args.get("inputRootFile"):
        raise ValueError("inputRootFile is required")
    if args["output_dir"] is None:
        raise ValueError("The native C++ loop writes ROOT files; output_dir cannot be None")

    config_path = args["branches_config_path"]
    if not config_path:
        config_path = str(resource_files("hepdataset").joinpath("defaults/branches_config_minimal.yml"))
    branch_reader = BranchesHandler(str(config_path))
    if not branch_reader.is_valid():
        branch_reader.print_validation()
        raise ValueError(f"Invalid branches configuration: {config_path}")

    binary, env = _ensure_native_binary()
    total_entries = _count_entries(binary, env, str(args["inputRootFile"]))
    checked = check_loop_args(args, total_entries)
    start, end = checked["start_entry"], checked["end_entry"]
    event_weight = checked["eventWeight"]
    n_processed = end - start
    out_dir = Path(checked["output_dir"]).resolve()
    out_name = checked["output_file_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_processed == 0:
        return {"trees": None, "root_paths": {}, "ObjectSelector": None, "EventSelector": None}

    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        max_workers = min(cpu_count, 4, n_processed)
    if max_workers < 1 or max_workers > cpu_count:
        raise ValueError(f"max_workers must be between 1 and {cpu_count}")
    if n_chunks is None:
        n_chunks = min(max_workers, n_processed)
    if n_chunks < 1 or n_chunks > n_processed:
        raise ValueError(f"n_chunks must be between 1 and {n_processed}")

    manifest_tmp = tempfile.TemporaryDirectory(prefix="hepdataset-native-config-")
    manifest_path = str(Path(manifest_tmp.name) / "branches.tsv")
    branch_reader.write_native_manifest(manifest_path)

    # A single C++ process writes directly to the requested destination.  For
    # multiple shards, each process gets a private directory and ROOT outputs
    # are merged only after every child exits successfully.
    if n_chunks == 1:
        command = _native_command(binary, manifest_path, checked,
                                  output_dir=str(out_dir), output_file_name=out_name,
                                  start=start, end=end, event_weight=event_weight,
                                  overwrite=True)
        try:
            _run_checked(command, env=env)
        finally:
            manifest_tmp.cleanup()
        paths = {key: str(out_dir / key.rpartition("_")[0] / key.rpartition("_")[2] / out_name)
                 for key in _channel_keys(args["split_by_flavour"])}
        return {"trees": None, "root_paths": paths, "ObjectSelector": None, "EventSelector": None}

    bounds = []
    base, remainder = divmod(n_processed, n_chunks)
    cursor = start
    for idx in range(n_chunks):
        size = base + (1 if idx < remainder else 0)
        bounds.append((cursor, cursor + size))
        cursor += size

    chunk_root = out_dir / ".native_chunks" / f"{Path(out_name).stem}_{os.getpid()}"
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunk_dirs = [chunk_root / f"chunk_{i:04d}" for i in range(len(bounds))]
    processes: list[tuple[subprocess.Popen, Path]] = []
    try:
        for idx, ((chunk_start, chunk_end), chunk_dir) in enumerate(zip(bounds, chunk_dirs)):
            cmd = _native_command(binary, manifest_path, checked,
                                  output_dir=str(chunk_dir), output_file_name=out_name,
                                  start=chunk_start, end=chunk_end,
                                  event_weight=event_weight, overwrite=True)
            if not args["show_progress"]:
                cmd = [x for x in cmd if x != "--show-progress"]
            logfile = chunk_root / f"chunk_{idx:04d}.log"
            stream = logfile.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((proc, logfile))
            # Enforce a concurrency limit without a Python ProcessPoolExecutor.
            if len(processes) >= max_workers:
                proc0, log0 = processes.pop(0)
                code = proc0.wait()
                if code:
                    for child, _ in processes:
                        child.terminate()
                    for child, _ in processes:
                        child.wait()
                    raise subprocess.CalledProcessError(code, cmd, output=log0.read_text(errors="replace"))
        failures = []
        for proc, logfile in processes:
            code = proc.wait()
            if code:
                failures.append((code, logfile))
        if failures:
            code, logfile = failures[0]
            raise subprocess.CalledProcessError(code, "native_adaptive_delphes", output=logfile.read_text(errors="replace"))

        root_paths: dict[str, str] = {}
        keys = _channel_keys(args["split_by_flavour"])
        for key in keys:
            region, _, category = key.rpartition("_")
            target = out_dir / region / category / out_name
            sources = [directory / region / category / out_name for directory in chunk_dirs]
            _hadd(env, target, sources, max_workers)
            root_paths[key] = str(target)
        for subdir in ("ObjectSelection", "EventSelection"):
            target = out_dir / subdir / out_name
            sources = [directory / subdir / out_name for directory in chunk_dirs]
            _hadd(env, target, sources, max_workers)
        return {"trees": None, "root_paths": root_paths, "ObjectSelector": None, "EventSelector": None}
    finally:
        manifest_tmp.cleanup()
        shutil.rmtree(chunk_root, ignore_errors=True)


def loop_tree(**loop_args: Any) -> dict[str, Any]:
    """Run one native C++ range while preserving the usual Python loop API."""
    loop_args.pop("max_workers", None)
    loop_args.pop("n_chunks", None)
    return run_native_in_parallel(n_chunks=1, max_workers=1, **loop_args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputRootFile", help="Input Delphes ROOT file or TChain wildcard")
    parser.add_argument("--branches-config-path", default=None, help="Validated branch YAML card (exported to a temporary native manifest)")
    parser.add_argument("--tree-name", default="Delphes")
    parser.add_argument("--event-weight", type=float, default=None)
    parser.add_argument("--cross-section", type=float, default=None)
    parser.add_argument("--luminosity", type=float, default=None)
    parser.add_argument("--start-entry", type=int, default=0)
    parser.add_argument("--end-entry", type=int, default=None)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--output-file-name", default="events.root")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parallel", action="store_true", help="Run native C++ ranges as subprocess shards")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--n-chunks", type=int, default=None)
    parser.add_argument("--split-by-flavour", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    args = vars(_parser().parse_args())
    if args.pop("parallel"):
        run_native_in_parallel(**args)
    else:
        loop_tree(**args)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a standalone command
    sys.exit(main())
