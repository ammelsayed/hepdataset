"""Run Delphes loop methods in parallel and merge their outputs."""

import os
import time
import subprocess
import shutil
import tempfile
from pathlib import Path
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import ROOT

n_cpu = os.cpu_count()

def merge_summaries(summaries):
    merged_cutflow = {}
    for s in summaries:
        for obj, stages in s["objSel_cutflow"].items():
            dst = merged_cutflow.setdefault(obj, {})
            for stage, count in stages.items():
                dst[stage] = dst.get(stage, 0) + count

    merged_ac = {}
    for s in summaries:
        for k, v in s["ac_counts"].items():
            merged_ac[k] = merged_ac.get(k, 0) + v

    merged_h = merge_object_selection_histograms([s["objSel_h"] for s in summaries])

    return {
        "objSel_cutflow": merged_cutflow,
        "ac_counts": merged_ac,
        "objSel_h": merged_h,
    }


def print_merged_summary(summary, treeName, eventWeight, luminosity):
    PrintObjectSelectionSummary(
        summary["objSel_cutflow"], lum = luminosity, event_weight = eventWeight
    )
    PrintAnalysisChannelYields(
        summary["ac_counts"], treeName, event_weight = eventWeight, lum = luminosity
    )


def build_chain(inputRootFile):
    chain = ROOT.TChain("Delphes")
    if isinstance(inputRootFile, list):
        for file in inputRootFile:
            chain.Add(file)
    elif isinstance(inputRootFile, str):
        chain.Add(inputRootFile)
    else:
        raise TypeError(f"inputRootFile must be str or list, got {type(inputRootFile).__name__}")
    return chain

def count_entries(inputRootFile):
    return build_chain(inputRootFile).GetEntries()


def isRootFilePath(path):
    return isinstance(path, (str, Path)) and str(path).endswith(".root") and Path(path).is_file()


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {int(s)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m"

def parallel_runs(func, args_list, max_workers = None, info = "", mpContext = None, show_job_progress=False):
    
    total = len(args_list)
    pending = list(enumerate(args_list))
    futures = {}
    results = [None] * total
    completed = 0
    start = time.perf_counter()

    # get the number of cores
    # do not take numbers higher than actual numbers of cores
    if max_workers is None:
        max_workers = os.cpu_count()
    else:
        max_workers = min(max_workers, os.cpu_count())
        

    # mp_context lets a caller force e.g. multiprocessing.get_context("spawn")
    # instead of the platform default (fork on Linux). This matters whenever
    # func relies on a library that initializes a multithreaded runtime at
    # import time (JAX/XLA being the relevant case here): forking after that
    # runtime is up is unsafe and can deadlock, whereas spawn re-imports the
    # module cleanly in each worker. Default None keeps existing callers
    # (e.g. the subprocess-based __main__ demo below) on the cheaper default.
    with ProcessPoolExecutor(max_workers=max_workers, mp_context = (get_context(mpContext) if mpContext else None)) as executor:
        # Launch the first batch
        while len(futures) < max_workers and pending:
            idx, args = pending.pop(0)
            futures[executor.submit(func, **args) if isinstance(args, dict) else executor.submit(func, *args)] = idx

        # Print initial status (time = 0.0s)
        current_time = time.strftime("%Hh%Mm%Ss") 
        last_time_str = format_time(0.0)
        print(f"{info}Idle: {len(pending)},  Running: {len(futures)},  Completed: {completed} [ current time: {current_time} ]")

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            for future in done:
                idx = futures.pop(future)
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = e
                completed += 1
                if pending:
                    new_idx, new_args = pending.pop(0)
                    futures[executor.submit(func, **new_args) if isinstance(new_args, dict) else executor.submit(func, *new_args)] = new_idx

            # Only print if the formatted time has changed (group by time slot)
            elapsed = time.perf_counter() - start
            time_str = format_time(elapsed)
            if time_str != last_time_str:
                print(f"{info}Idle: {len(pending)},  Running: {len(futures)},  Completed: {completed} [ {time_str} ]")
                last_time_str = time_str

    # Final print (ensures we show the very last state)
    final_time = format_time(time.perf_counter() - start)
    if final_time != last_time_str or completed == total:
        print(f"{info}Idle: {len(pending)},  Running: {len(futures)},  Completed: {completed} [ {final_time} ]")
    return results


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


def split_range(total, n):
    n = max(1, min(n, total))
    base, rem = divmod(total, n)
    out, start = [], 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        out.append((start, start + size))
        start += size
    return out

def merge_trees(results, treeName):
    """results = list of {sr_key: TTree} dicts from parallel workers."""

    print("Merging split ROOT.TTrees dict ..")
    start = time.perf_counter()

    ac_keys = list(results[0].keys())  # learn structure from first worker
    merged_trees = {}
    for ac_key in ac_keys:
        tl = ROOT.TList()
        for trees in results:
            tl.Add(trees[ac_key])
        merged = ROOT.TTree.MergeTrees(tl)
        merged.SetName(treeName)
        merged.SetTitle(ac_key)
        merged.SetDirectory(0)   # keep detached; read() will cd() + Write()
        merged_trees[ac_key] = merged
    
    print(f"Merged {len(results)} splits x {len(merged_trees)} AC(s) [{format_time(time.perf_counter() - start)}]")
    return merged_trees



def hadd_files(target_path, source_paths, max_workers=None):
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
    j = max_workers or n_cpu
    cmd = ["hadd", "-f", "-j", str(j), target_path] + list(source_paths)
    # print(f"  hadd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  hadd FAILED (rc={result.returncode}):\n{result.stderr}")
        return False
    return True

def hadd_splits(results, treeName, output_dir, output_file_name="events.root", max_workers=None):
    """results = list of {ac_key: file_path} dicts from parallel workers."""

    print(f"Merging split .root files with hadd (-j {max_workers or n_cpu}) ..")
    start = time.perf_counter()

    ac_keys = list(results[0].keys())  # learn structure from first worker

    # Group chunk files by ac_key
    ac_files = {ac_key: [] for ac_key in ac_keys}
    for chunk_paths in results:
        for ac_key, path in chunk_paths.items():
            if ac_key in ac_files and path is not None:
                ac_files[ac_key].append(path)

    merged = {}
    for ac_key, files in ac_files.items():
        if not files:
            continue

        # Chunk files already live in the destination directory, e.g.
        # <output_dir>/<ac>/<ac_r>/<treeName>_split_<i>_events_<s>_<e>.root.
        out_path = os.path.join(os.path.dirname(files[0]), output_file_name)
        ok = hadd_files(out_path, files, max_workers=max_workers)
        if ok:
            merged[ac_key] = out_path
            # hadd never deletes its inputs; clean them up ourselves.
            for f in files:
                try:
                    os.remove(f)
                except OSError as exc:
                    print(f"  WARNING: could not remove chunk {f}: {exc}")
        else:
            print(f"  WARNING: hadd failed for {ac_key}")

    print(f"Merged {len(results)} chunks x {len(merged)} SR(s) in {format_time(time.perf_counter() - start)}")
    return merged



def run_in_parallel(
    loop_tree_method,
    max_workers = None,
    n_chunks = None,
    merge_method = 1,
    return_summary = False,
    **loop_kwargs,
    ):  


    if merge_method not in (1, 2):
        raise ValueError("merge_method must be 1 or 2")

    inputRootFile = loop_kwargs["inputRootFile"]
    output_dir = loop_kwargs["output_dir"]
    treeName = loop_kwargs["treeName"]
    output_file_name = loop_kwargs["output_file_name"]
    overwrite = loop_kwargs.get("overwrite", False)

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

    luminosity = loop_kwargs.get("luminosity")
    parent_show_progress = loop_kwargs.get("show_progress", False)

    if loop_kwargs.get("eventWeight") is None:
        cross_section = loop_kwargs.get("cross_section")
        luminosity = loop_kwargs.get("luminosity")
        if (cross_section is None) != (luminosity is None):
            raise ValueError("cross section and luminosity must be provided together")
        if cross_section is not None:
            loop_kwargs["eventWeight"] = (
                cross_section * luminosity / numberOfProcessedEntries
            )
        else:
            loop_kwargs["eventWeight"] = 1.0

    loop_kwargs.pop("cross_section", None)
    loop_kwargs.pop("luminosity", None)

    # Check aganist max_workers input
    n_cpu = os.cpu_count()
    max_workers = max_workers or n_cpu
    if (max_workers < 1) or (max_workers > n_cpu):
        raise ValueError(f"max_workers must be between 1 and {n_cpu}")

    # Check againist n_chunks input
    n_chunks = n_chunks or max_workers
    if (n_chunks < 1) or (n_chunks > numberOfProcessedEntries):
        raise ValueError(f"n_chunks must be between 1 and {numberOfProcessedEntries}")

    print(f"Reading ROOT file: {inputRootFile}")
    print(f"Total events in file: {numberOfEntries}")
    print(f"Processing events: {start_entry} to {end_entry - 1} ({numberOfProcessedEntries} events)")
    print(f"Event weight: {loop_kwargs['eventWeight']}")

    # Describe the split plan
    chunk_ranges = split_range(numberOfProcessedEntries, n_chunks)
    chunk_sizes = [e - s for s, e in chunk_ranges]
    print(
        f"Splitting {numberOfProcessedEntries} events into {len(chunk_ranges)} chunks "
        f"({int(numberOfProcessedEntries / len(chunk_ranges))} per chunk) "
        f"across {max_workers} worker(s)"
    )
    print(f"Merge method: {merge_method} "
          f"({'in-memory TTree merge' if merge_method == 1 else 'per-chunk .root files + hadd'})")

    # Workers should never print their own progress bars or summaries.
    loop_kwargs["show_progress"] = False
    loop_kwargs["debug_loop"] = False

    # If the destination directory does not exist yet, create it here so that
    # the parallel workers do not race on os.makedirs inside loop_tree.
    if output_dir is not None:
        if os.path.exists(output_dir):
            if not os.path.isdir(output_dir):
                raise ValueError(f"output_dir is not a directory: {output_dir}")
            if not overwrite:
                raise FileExistsError(f"Output directory already exists, cannot write there: {output_dir}")
        else:
            print(f"Creating output directory : {output_dir}")
            os.makedirs(output_dir)

    splits_args = []
    for i, (s, e) in enumerate(chunk_ranges):

        split_argrs = dict(loop_kwargs)
        split_argrs["start_entry"] = start_entry + s
        split_argrs["end_entry"] = start_entry + e
        
        split_argrs["collect_summary"] = True

        if merge_method == 2:
            # Workers write their chunk to the destination layout with a
            # unique chunk file name; the parent then merges with hadd.
            split_argrs["output_dir"] = output_dir
            split_argrs["output_file_name"] = f"tmp_{treeName}_split_{i}_events_{start_entry + s}_{start_entry + e - 1}.root"
            split_argrs["overwrite"] = True
        else:
            # merge_method == 1: workers return their TTrees in memory;
            # the parent process merges them and (if requested) writes.
            split_argrs["output_dir"] = None
            split_argrs["output_file_name"] = "events.root"
            split_argrs["overwrite"] = False

        splits_args.append(split_argrs)


    # Run process pool executor
    results = parallel_runs(loop_tree_method, splits_args, max_workers=max_workers, info="", mpContext="fork")
    for r in results:
        if isinstance(r, Exception):
            raise r
    
    # Unwrap (result, summary) tuples returned by the workers.
    merged_summary = None
    summaries = []
    if results and all(isinstance(r, tuple) and len(r) == 2 for r in results):
        unwrapped = []
        for r, s in results:
            unwrapped.append(r)
            summaries.append(s)
        results = unwrapped

    # the results could be a list of ROOT.TTree objects that needs to be merged.
    # Or could be a list of file paths (one list per chunk)
    # depending on the value of output_dir, if its None, and the results a dict of ROOT.TTrees, 
    # then we merge the ROOT.TTrees and return the dict of merghed ROOT.TTrees,
    # but if its dict of root paths, then we merge the .root files using hadd.
    # Hence if from the begining the output_dir is None, we do not need to path split_output_dir to the split, workers
    # because each split worker will output its dict of TTrees, and we just merge them
    if isinstance(results, list) and all(isinstance(r, dict) for r in results):

        # Print the merged cutflow / channel counts once in the parent.
        if summaries and not return_summary:
            merged_summary = merge_summaries(summaries)
            print_merged_summary(
                merged_summary,
                treeName=treeName,
                eventWeight=loop_kwargs.get("eventWeight"),
                luminosity=luminosity,
            )
            if parent_show_progress and output_dir is not None:
                out_path = os.path.join(output_dir, "ObjectSelection", treeName)
                os.makedirs(out_path, exist_ok=True)
                DrawObjectSelectionHistograms(merged_summary["objSel_h"], output_dir=out_path)

        if all(isinstance(v, ROOT.TTree) for r in results for v in r.values()):
            merged_trees = merge_trees(results, treeName)
            if output_dir is None:
                result = merged_trees
            else:
                written = {}
                for ac_key, tree in merged_trees.items():
                    ac, _, ac_r = ac_key.rpartition("_")
                    out_path = os.path.join(output_dir, ac, ac_r, output_file_name)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    f_out = ROOT.TFile.Open(out_path, "RECREATE")
                    tree.SetDirectory(f_out)
                    tree.Write()
                    f_out.Close()
                    written[ac_key] = out_path
                result = written

        elif all(isRootFilePath(v) for r in results for v in r.values()):
            result = hadd_splits(results, treeName, output_dir, output_file_name, max_workers=max_workers)

        else:
            result = None

    else:
        result = None

    if return_summary:
        return result, summaries
    return result