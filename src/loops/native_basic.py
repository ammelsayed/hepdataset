"""Run native fixed-schema Delphes loops from the Python dataset coordinator."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import uproot


def _runtime_env():
    env = os.environ.copy()
    root_config = env.get("ROOT_CONFIG") or shutil.which("root-config")
    if not root_config and env.get("ROOTSYS"):
        root_config = str(Path(env["ROOTSYS"]) / "bin" / "root-config")
    delphes_dir = env.get("DELPHES_DIR", env.get("DELPHES_HOME"))
    if root_config:
        env["PATH"] = str(Path(root_config).resolve().parent) + os.pathsep + env.get("PATH", "")
    if delphes_dir:
        env["LD_LIBRARY_PATH"] = delphes_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def merge_native_root_files(sources, target, max_workers=None, env=None):
    sources = [Path(path) for path in sources if Path(path).is_file()]
    if not sources:
        raise FileNotFoundError("No C++ output ROOT files were produced")
    if len(sources) == 1:
        shutil.copy2(sources[0], target)
    else:
        subprocess.run(
            ["hadd", "-f", "-j", str(max_workers or os.cpu_count() or 1), str(target),
             *(str(path) for path in sources)],
            env=env or _runtime_env(), check=True,
        )


def run_native_basic(loop_name, input_root_file, tree_name, event_weight,
                     output_dir, output_file_name, max_workers=None,
                     n_chunks=None, show_progress=False):
    """Split an input file into ranges, run C++ workers, and merge their ROOT outputs."""
    source_dir = Path(__file__).resolve().parent / "native_cpp"
    binary = source_dir / "build" / loop_name
    root_config = os.environ.get("ROOT_CONFIG") or shutil.which("root-config")
    if not root_config and os.environ.get("ROOTSYS"):
        root_config = str(Path(os.environ["ROOTSYS"]) / "bin" / "root-config")
    delphes_dir = os.environ.get("DELPHES_DIR", os.environ.get("DELPHES_HOME"))
    if not binary.is_file():
        if not root_config or not delphes_dir:
            raise RuntimeError("Build the native loop first, or set ROOT_CONFIG and DELPHES_DIR")
        subprocess.run(
            ["make", "-C", str(source_dir), loop_name,
             f"ROOT_CONFIG={root_config}", f"DELPHES_DIR={delphes_dir}"],
            check=True,
        )

    env = _runtime_env()
    with uproot.open(input_root_file) as root_file:
        entries = root_file["Delphes"].num_entries
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    chunks = n_chunks or max_workers or os.cpu_count() or 1
    if chunks < 1 or chunks > entries:
        raise ValueError(f"n_chunks must be between 1 and {entries}")
    ranges = [(entries * i // chunks, entries * (i + 1) // chunks) for i in range(chunks)]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".native-basic-", dir=output_dir))

    try:
        commands = []
        for index, (start, end) in enumerate(ranges):
            chunk_dir = work_dir / f"chunk_{index}"
            chunk_dir.mkdir()
            command = [
                str(binary), "--input-root-file", str(input_root_file),
                "--output-dir", str(chunk_dir), "--output-file-name", output_file_name,
                "--tree-name", tree_name, "--start-entry", str(start), "--end-entry", str(end),
                "--event-weight", str(event_weight),
            ]
            if show_progress:
                command.append("--show-progress")
            commands.append(command)

        worker_limit = max(1, max_workers or os.cpu_count() or len(commands))
        for offset in range(0, len(commands), worker_limit):
            batch = commands[offset:offset + worker_limit]
            processes = [subprocess.Popen(command, env=env) for command in batch]
            failures = [process.wait() for process in processes]
            failures = [code for code in failures if code]
            if failures:
                raise subprocess.CalledProcessError(failures[0], batch[0])

        if loop_name == "basic3_delphes":
            channel_files = work_dir.glob(f"chunk_*/*/*/{output_file_name}")
            channels = sorted({f"{path.parent.parent.name}_{path.parent.name}" for path in channel_files})
            root_paths = {}
            for channel in channels:
                region, category = channel.split("_", 1)
                sources = [work_dir / f"chunk_{i}" / region / category / output_file_name for i in range(chunks)]
                target = output_dir / region / category / output_file_name
                target.parent.mkdir(parents=True, exist_ok=True)
                merge_native_root_files(sources, target, max_workers, env)
                root_paths[channel] = str(target)
            for metadata in ("ObjectSelection", "EventSelection"):
                sources = [work_dir / f"chunk_{i}" / metadata / f"{tree_name}.root" for i in range(chunks)]
                target = output_dir / metadata / output_file_name
                target.parent.mkdir(parents=True, exist_ok=True)
                merge_native_root_files(sources, target, max_workers, env)
        else:
            sources = [work_dir / f"chunk_{i}" / output_file_name for i in range(chunks)]
            target = output_dir / output_file_name
            merge_native_root_files(sources, target, max_workers, env)
            root_paths = {None: str(target)}
        return {"root_paths": root_paths, "ObjectSelector": None, "EventSelector": None}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
