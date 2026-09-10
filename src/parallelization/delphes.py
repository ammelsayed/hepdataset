import os
import ROOT
import subprocess

# where to spill per-chunk/merge temp .root files
tmpDir = os.path.dirname(os.path.abspath(__file__))
tempReaderDir = os.path.join(tmpDir, "TempReaderOutput")


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


def split_range(total, n):
    n = max(1, min(n, total))
    base, rem = divmod(total, n)
    out, start = [], 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        out.append((start, start + size))
        start += size
    return out

def hadd_chunks(chunk_results, signal_regions_keys, treeName):
    """Merge per-chunk temp files for each SR using hadd.

    Parameters
    ----------
    chunk_results : list[list[str]]
        Each element is the list of temp file paths returned by a chunk worker.
    signal_regions_keys : list[str]
        All known SR keys.
    treeName : str
        Name of the tree inside the files (e.g. "tt_sample1").

    Returns
    -------
    dict[str, str]
        Mapping  sr_key -> merged_sample_file_path  for SRs that got entries.
        The merged files live under TempReaderOutput/{sr_key}/{treeName}.root
    """
    print("Merging chunks with hadd ..")
    start = time.perf_counter()

    # Group chunk files by sr_key
    sr_files = {sr_key: [] for sr_key in signal_regions_keys}
    for chunk_paths in chunk_results:
        for path in chunk_paths:
            # path pattern: .../{sr_key}/{treeName}.root
            sr_key = os.path.basename(os.path.dirname(path))
            if sr_key in sr_files:
                sr_files[sr_key].append(path)

    merged = {}
    for sr_key, files in sr_files.items():
        if not files:
            continue
        out_path = os.path.join(tempReaderDir, sr_key, f"{treeName}.root")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ok = hadd_files(out_path, files)
        if ok:
            merged[sr_key] = out_path
        else:
            print(f"  WARNING: hadd failed for {sr_key}")

    print(f"Merged {len(chunk_results)} chunks x {len(merged)} SR(s) "
          f"[{format_time(time.perf_counter() - start)}]")
    return merged


def loop_tree_advanced(inputRootFile, treeName, sampleWeight, signal_regions_keys, run_parallel=True, n_chunks=None, max_workers=None, max_entries=None):
    if not run_parallel:
        # Non-parallel path: loop_tree returns {sr_key: TTree} in memory.
        # Write each tree to TempReaderOutput so the rest of the pipeline
        # (which expects file paths) works uniformly.
        trees = loop_tree(inputRootFile, treeName, sampleWeight, signal_regions_keys, max_entries=max_entries)
        written = {}
        for sr_key, tree in trees.items():
            if tree.GetEntries() == 0:
                continue
            out_path = os.path.join(tempReaderDir, sr_key, f"{treeName}.root")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            f_out = ROOT.TFile.Open(out_path, "RECREATE")
            tree.SetDirectory(f_out)
            tree.Write()
            f_out.Close()
            written[sr_key] = out_path
        return written

    total = count_entries(inputRootFile)
    n_chunks = n_chunks or (max_workers or os.cpu_count())

    # Create a temp directory for this sample's chunk outputs.
    # Each chunk gets its own subdirectory to avoid write collisions.
    sample_temp_dir = tempfile.mkdtemp(prefix=f"make_dataset_chunks_{treeName}_", dir=tmpDir)

    try:
        chunk_dirs = []
        chunks = []
        for i, (s, e) in enumerate(split_range(total, n_chunks)):
            chunk_dir = os.path.join(sample_temp_dir, f"chunk_{i}")
            os.makedirs(chunk_dir, exist_ok=True)
            chunk_dirs.append(chunk_dir)
            chunks.append((inputRootFile, treeName, sampleWeight, signal_regions_keys,
                           s, e, False, False, None, chunk_dir))
        results = parallel_runs(loop_tree, chunks, max_workers=max_workers, info="", mpContext="fork")
        for r in results:
            if isinstance(r, Exception):
                raise r
        # results is a list of lists of file paths (one list per chunk)
        return hadd_chunks(results, signal_regions_keys, treeName)
    finally:
        # Clean up the temp directory (chunk files were already removed by hadd_chunks;
        # this removes any empty sr_key subdirs and the temp dir itself).
        shutil.rmtree(sample_temp_dir, ignore_errors=True)
