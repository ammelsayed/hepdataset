"""Run Delphes loop methods in parallel and merge their outputs."""

import os
import shutil
import subprocess
import tempfile

import ROOT

from . import parallel_runs
from delphes import count_entries


def add_parallel_arguments(parser):
    parser.add_argument(
        "--parallel", action="store_true",
        help=(
            "Enable parallel event processing. Use --max-workers to limit "
            "simultaneous processes and --n-chunks to choose the number of "
            "event ranges."
        ),
    )
    parser.add_argument(
        "--max-workers", type=int, default=None, metavar="N",
        help=(
            "Maximum number of worker processes running at the same time. "
            "If omitted, use all available CPU cores."
        ),
    )
    parser.add_argument(
        "--n-chunks", type=int, default=None, metavar="N",
        help=(
            "Total number of event-range jobs to create. Each job processes "
            "one contiguous part of the selected input range. This controls "
            "the number of jobs, not the number running simultaneously."
        ),
    )
    parser.add_argument(
        "--merge-method", type=int, choices=(1, 2), default=1, metavar="N",
        help=(
            "1: keep each worker TTree in memory, merge all trees in the "
            "parent process, then write events.root; if output_dir is None, "
            "return the merged trees without writing files. 2: make each "
            "worker write an events_chunk file, then merge those files with "
            "hadd into events.root."
        ),
    )
    parser.add_argument(
        "--temp-dir", default=None, metavar="PATH",
        help=(
            "Directory where method 2 writes worker chunk files before hadd. "
            "If omitted, a temporary directory is created automatically and "
            "removed after merging."
        ),
    )


def split_range(total, n_chunks):
    # Divide the requested event interval into contiguous [start, end) ranges.
    n_chunks = max(1, min(n_chunks, total))
    base, remainder = divmod(total, n_chunks)
    chunks = []
    start = 0
    for index in range(n_chunks):
        size = base + (index < remainder)
        chunks.append((start, start + size))
        start += size
    return chunks


def merge_trees(results):
    """Merge the same-key trees returned by each worker."""
    # A flat loop returns one TTree; a channelized loop returns {channel: TTree}.
    keys = set()
    for result in results:
        keys.update(result if isinstance(result, dict) else [None])

    merged_trees = {}
    for key in keys:
        tree_list = ROOT.TList()
        for result in results:
            trees = result if isinstance(result, dict) else {None: result}
            if key in trees:
                tree_list.Add(trees[key])
        merged = ROOT.TTree.MergeTrees(tree_list)
        if merged:
            merged.SetDirectory(0)
            merged_trees[key] = merged
    return merged_trees


def hadd_files(target_path, source_paths):
    # Merge all chunk ROOT files for one output tree into the final ROOT file.
    result = subprocess.run(
        ["hadd", "-f", target_path, *source_paths], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"hadd failed for {target_path}: {result.stderr}")


def run_in_parallel(
    loop_tree_method,
    max_workers = None,
    n_chunk = None,
    merge_method = 1,
    temp_dir = None,
    **loop_kwargs,
):
    if merge_method not in (1, 2):
        raise ValueError("merge_method must be 1 or 2")

    inputRootFile = loop_kwargs["inputRootFile"]
    output_dir = loop_kwargs["output_dir"]
    treeName = loop_kwargs["treeName"]
    if output_dir is None:
        # Without an output directory, files cannot be used for merging.
        merge_method = 1
    numberOfEntries = count_entries(inputRootFile)
    start_entry = loop_kwargs.pop("start_entry", 0)
    end_entry = loop_kwargs.pop("end_entry", None)

    if start_entry < 0 or start_entry > numberOfEntries:
        raise ValueError(f"start_entry must be between 0 and {numberOfEntries}")
    if end_entry is None or end_entry > numberOfEntries:
        end_entry = numberOfEntries
    if end_entry < start_entry:
        raise ValueError("end_entry must be greater than or equal to start_entry")

    numberOfProcessedEntries = end_entry - start_entry
    if numberOfProcessedEntries == 0:
        return {}

    # Check aganist max_workers input
    n_cpu = os.cpu_count()
    max_workers = max_workers or n_cpu
    if (max_workers < 0) or (max_workers > n_cpu):
        raise ValueError(f"max_workers must be between 0 and {n_cpu}")

    # Check againist n_chunk input
    n_chunk = n_chunk or max_workers
    if (n_chunk < 0) or (n_chunk > numberOfProcessedEntries):
        raise ValueError(f"n_chunk must be between 0 and {numberOfProcessedEntries}")

    if loop_kwargs.get("eventWeight") is None:
        cross_section = loop_kwargs.get("cross_section")
        luminosity = loop_kwargs.get("luminosity")
        if (cross_section is None) != (luminosity is None):
            raise ValueError("cross section and luminosity must be provided together")
        if cross_section is not None:
            loop_kwargs["eventWeight"] = (
                cross_section * luminosity / numberOfProcessedEntries
            )
    loop_kwargs.pop("cross_section", None)
    loop_kwargs.pop("luminosity", None)

    # Workers should never print their own progress bars or summaries.
    loop_kwargs["show_progress"] = False

    temporary_dirs = []
    if output_dir is not None and merge_method == 2:
        # Method 2 owns the temporary directory and removes it after hadd.
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix=f"{treeName}_parallel_")
        else:
            os.makedirs(temp_dir, exist_ok=True)
        temporary_dirs.append(temp_dir)

    try:
        # Build one keyword-argument dictionary for every event-range job.
        per_job_kwargs = []
        chunk_ranges = split_range(numberOfProcessedEntries, n_chunk)
        for offset_start, offset_end in chunk_ranges:
            job_kwargs = dict(loop_kwargs)
            job_kwargs["start_entry"] = start_entry + offset_start
            job_kwargs["end_entry"] = start_entry + offset_end
            
            # No output directory means workers return trees for in-memory merging.
            if output_dir == None:
                job_kwargs["output_dir"] = None
            
            # With an output directory, method 1 still returns trees to the parent.
            # Method 2 writes each worker's chunk directly to a ROOT file.
            else:
                if merge_method == 1:
                    job_kwargs["output_dir"] = None
                
                elif merge_method == 2:
                    # All workers write their chunk files below this directory.
                    job_kwargs["output_dir"] = temp_dir
                    job_kwargs["output_file_name"] = f"events_chunk_{job_kwargs['start_entry']}_{job_kwargs['end_entry']}.root"
                    
            per_job_kwargs.append(job_kwargs)

        # Execute every prepared job with the requested worker limit.
        results = parallel_runs(
            loop_tree_method, per_job_kwargs, max_workers=max_workers, mpContext="fork"
        )
        for result in results:
            if isinstance(result, Exception):
                raise result

        # Method 1 merges worker TTrees in the parent process.
        if merge_method == 1:
            merged_trees = merge_trees(results)
            if output_dir is not None:
                # Write each merged tree using the same channel layout as loop_tree.
                os.makedirs(output_dir, exist_ok=True)
                paths = {}
                for key, tree in merged_trees.items():
                    if key is None:
                        path = os.path.join(output_dir, "events.root")
                    else:
                        ac, ac_r = key.split("_", 1)
                        path = os.path.join(output_dir, ac, ac_r, "events.root")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    f_out = ROOT.TFile.Open(path, "RECREATE")
                    tree.SetDirectory(f_out)
                    tree.Write()
                    f_out.Close()
                    paths[key] = path
                return paths
            else:
                return merged_trees

        elif merge_method == 2:
            # Method 2 merges the chunk files directly with hadd.
            os.makedirs(output_dir, exist_ok=True)
            paths = {}
            
            # Every worker returns the same tree keys; flat loops use key None.
            first_result = results[0]
            keys = first_result.keys() if isinstance(first_result, dict) else [None]

            for key in keys:
                if key is None:
                    # Flat loop: temp_dir/events_chunk_start_end.root.
                    target_path = os.path.join(output_dir, "events.root")
                    source_paths = []
                    for offset_start, offset_end in chunk_ranges:
                        file_name = f"events_chunk_{start_entry + offset_start}_{start_entry + offset_end}.root"
                        source_paths.append(os.path.join(temp_dir, file_name))
                else:
                    # Channelized loop: temp_dir/channel/region/events_chunk_*.root.
                    ac, ac_r = key.split("_", 1)
                    target_path = os.path.join(output_dir, ac, ac_r, "events.root")
                    source_paths = []
                    for offset_start, offset_end in chunk_ranges:
                        file_name = f"events_chunk_{start_entry + offset_start}_{start_entry + offset_end}.root"
                        source_paths.append(os.path.join(temp_dir, ac, ac_r, file_name))

                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                hadd_files(target_path, source_paths)
                paths[key] = target_path

            return paths
    finally:
        for directory in temporary_dirs:
            shutil.rmtree(directory, ignore_errors=True)
