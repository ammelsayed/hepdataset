#!/usr/bin/env python3

"""Run one of the Delphes loop modules over a ROOT file in parallel."""

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import importlib.util
import inspect
from pathlib import Path

PARALLELIZATION_DIR = Path(__file__).resolve().parent
SRC_DIR = PARALLELIZATION_DIR.parent
LOOPS_DIR = SRC_DIR / "loops"

# Loop modules use local imports such as ``from delphes import ...``.
if str(LOOPS_DIR) not in sys.path:
    sys.path.insert(0, str(LOOPS_DIR))

from delphes import count_entries, load_delphes  # noqa: E402

_parallelization_init = importlib.util.spec_from_file_location(
    "hepdataset_parallelization", PARALLELIZATION_DIR / "__init__.py"
)
_parallelization_module = importlib.util.module_from_spec(_parallelization_init)
_parallelization_init.loader.exec_module(_parallelization_module)
format_time = _parallelization_module.format_time
parallel_runs = _parallelization_module.parallel_runs

# where to spill per-chunk/merge temp .root files
tmpDir = str(PARALLELIZATION_DIR)


def hadd_files(target_path, source_paths):
    """Merge ROOT files using the hadd command-line tool.

    Parameters
    ----------
    target_path : str
        Output file path (created / overwritten by hadd).
    source_paths : list[str] or str
        One or more input ROOT file paths / globs.

    Returns
    -------
    bool  – True on success, False on failure.
    """
    if isinstance(source_paths, str):
        source_paths = [source_paths]
    cmd = ["hadd", "-f", target_path] + list(source_paths)
    # print(f"  hadd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  hadd FAILED (rc={result.returncode}):\n{result.stderr}")
        return False
    return True


def hadd_chunks(chunk_results, output_dir, treeName):
    """Merge all files returned by workers into ``output_dir``.

    Parameters
    ----------
    chunk_results : list
        Worker return values. A loop may return a path, a mapping of channel
        names to paths, or a list of paths.
    output_dir : str
        Directory for merged ROOT files.
    treeName : str
        Name of the tree inside the files (e.g. "tt_sample1").

    Returns
    -------
    dict[str, str]
        Mapping of merged output paths keyed by filename.
    """
    print("Merging chunks with hadd ..")
    start = time.perf_counter()

    grouped_files = {}
    for result in chunk_results:
        if isinstance(result, dict):
            paths = result.values()
        elif isinstance(result, (list, tuple, set)):
            paths = result
        else:
            paths = [result]
        for path in paths:
            if path:
                path_obj = Path(path)
                filename = path_obj.name
                if filename.startswith(f"{treeName}_") and filename.endswith(".root"):
                    channel = filename[len(treeName) + 1:-len(".root")]
                    relative_name = os.path.join(channel, filename)
                else:
                    relative_name = filename
                grouped_files.setdefault(relative_name, []).append(path)

    merged = {}
    for relative_name, files in grouped_files.items():
        out_path = os.path.join(output_dir, relative_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ok = hadd_files(out_path, files)
        if ok:
            merged[relative_name] = out_path
        else:
            print(f"  WARNING: hadd failed for {relative_name}")

    print(f"Merged {len(chunk_results)} chunks x {len(merged)} output file(s) "
          f"[{format_time(time.perf_counter() - start)}]")
    return merged

def split_range(total, n):
    if total <= 0:
        return []
    n = max(1, min(n, total))
    base, rem = divmod(total, n)
    out, start = [], 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        out.append((start, start + size))
        start += size
    return out

def loop_tree_parallel(
    inputRootFile,
    treeName,
    eventWeight = 1.0,
    start_entry = 0,
    end_entry = None,
    temp_dir_path = None,
    show_progress = False,
    debug_loop = False,
    n_chunks=None, 
    max_workers=None,
    loop_tree=None,
    loop_kwargs=None,
):

    total = count_entries(inputRootFile)
    start_entry = max(0, start_entry)
    end_entry = total if end_entry is None else min(end_entry, total)
    if end_entry < start_entry:
        raise ValueError("end_entry must be greater than or equal to start_entry")
    n_chunks = n_chunks or (max_workers or os.cpu_count() or 1)
    if n_chunks < 1:
        raise ValueError("n_chunks must be at least 1")
    if loop_tree is None:
        raise ValueError("loop_tree must be provided")
    loop_kwargs = dict(loop_kwargs or {})
    accepted_kwargs = set(inspect.signature(loop_tree).parameters)
    loop_kwargs = {key: value for key, value in loop_kwargs.items() if key in accepted_kwargs}
    output_dir = temp_dir_path or os.path.join(tmpDir, "TempReaderOutput")

    # Create a temp directory for this sample's chunk outputs.
    # Each chunk gets its own subdirectory to avoid write collisions in case of many analysic channels root files
    sample_temp_dir = tempfile.mkdtemp(prefix=f"{treeName}_chunk", dir=tmpDir)

    try:
        chunk_dirs = []
        chunks = []
        for i, (offset_start, offset_end) in enumerate(
            split_range(end_entry - start_entry, n_chunks)
        ):
            s = start_entry + offset_start
            e = start_entry + offset_end
            chunk_dir = os.path.join(sample_temp_dir, f"chunk_{i}")
            os.makedirs(chunk_dir, exist_ok=True)
            chunk_dirs.append(chunk_dir)
            chunk_kwargs = {
                "inputRootFile": inputRootFile,
                "treeName": treeName,
                "eventWeight": eventWeight,
                "start_entry": s,
                "end_entry": e,
                "temp_dir_path": chunk_dir,
                "show_progress": show_progress,
                "debug_loop": debug_loop,
                **loop_kwargs,
            }
            chunks.append((loop_tree, chunk_kwargs))
        results = parallel_runs(
            _run_loop_chunk, chunks, max_workers=max_workers,
            info="Loop jobs: ", mpContext="fork"
        )
        for r in results:
            if isinstance(r, Exception):
                raise r
        # results is a list of lists of file paths (one list per chunk)
        return hadd_chunks(results, output_dir, treeName)
    finally:
        # Clean up the temp directory (chunk files were already removed by hadd_chunks;
        # this removes any empty sr_key subdirs and the temp dir itself).
        shutil.rmtree(sample_temp_dir, ignore_errors=True)


def _run_loop_chunk(loop_tree, kwargs):
    return loop_tree(**kwargs)


def load_loop(loop_file):
    """Import a loop module named like ``basic1_delphes`` or with ``.py``."""
    module_name = Path(loop_file).stem
    if not module_name.isidentifier():
        raise ValueError(f"Invalid loop module name: {loop_file}")
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise ModuleNotFoundError(
            f"Could not import loop '{module_name}' from {LOOPS_DIR}"
        ) from error


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a Delphes loop over one ROOT file using multiple workers."
    )
    parser.add_argument("input_root_file", help="Path to the input Delphes ROOT file.")
    parser.add_argument(
        "--loop-file", default="adaptive_delphes",
        help="Loop module filename in src/loops (default: adaptive_delphes).",
    )
    parser.add_argument("--tree-name", default="Delphes", help="Input/output tree name.")
    parser.add_argument("--output-dir", default=".", help="Directory for merged ROOT files.")
    parser.add_argument("--event-weight", type=float, default=1.0)
    parser.add_argument("--start-entry", type=int, default=0)
    parser.add_argument("--end-entry", type=int, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--debug", action="store_true", dest="debug_loop")
    parser.add_argument("--n-chunks", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_chunks is not None and args.n_chunks < 1:
        raise SystemExit("--n-chunks must be at least 1")
    if args.max_workers is not None and args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1")

    loop_module = load_loop(args.loop_file)
    if not hasattr(loop_module, "loop_tree"):
        raise SystemExit(f"Loop module '{args.loop_file}' does not define loop_tree")

    load_delphes()
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = loop_tree_parallel(
        inputRootFile=args.input_root_file,
        treeName=args.tree_name,
        eventWeight=args.event_weight,
        start_entry=args.start_entry,
        end_entry=args.end_entry,
        temp_dir_path=args.output_dir,
        show_progress=args.show_progress,
        debug_loop=args.debug_loop,
        n_chunks=args.n_chunks,
        max_workers=args.max_workers,
        loop_tree=loop_module.loop_tree,
    )
    for output_path in outputs.values():
        print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
