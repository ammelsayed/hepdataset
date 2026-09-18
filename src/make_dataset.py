#!/usr/bin/env python3
"""Process every ROOT file in a samples YAML card and merge per process."""

import os
import sys
import importlib
try:
    from importlib.resources import files as resource_files
except ImportError:  # pragma: no cover - only used on Python 3.8
    from importlib_resources import files as resource_files
from pathlib import Path

def resolve_category_name(category):
    cat = str(category).lower()
    return "bkg" if cat == "background" else "sig" if cat == "signal" else "Unknown"

def _load_loop(name):
    """Import a hepdataset.loops module and return its loop_tree."""
    module = importlib.import_module(f".loops.{Path(name).stem}", package=__package__)
    return module.loop_tree

def make_dataset(
    samples_file,
    branches_config_file = None, 
    output_dir = "HEPDataset", 
    loop_method = "adaptive_delphes",
    working_luminosity = 400.0,
    run_parallel = True, 
    merge_proc_samples = False,
    max_workers = None, 
    n_chunks = None,
    show_progress=False, 
    ):

    # Import the required libraries
    from .samples_reader          import SamplesReader
    from .core.delphes_utilis     import load_delphes
    from .loops.parallel_loop     import run_in_parallel, hadd_files
    from .core.object_selection   import ObjectSelector
    from .core.event_selection    import EventSelector

    # Read samples
    print("Reading the samples file ...")
    Data = SamplesReader(str(samples_file)).read()

    # Read the branches configuration file.
    if branches_config_file is None:
        branches_config_file = str(
            resource_files("hepdataset").joinpath("defaults/branches_config.yml")
        )
        print(f"Using packaged default branches configuration file at: {branches_config_file}")
    else:
        print("Checking the given branches configuration file ...")

    loop_tree = _load_loop(loop_method)
    load_delphes()
    output_dir = Path(output_dir).resolve()

    for category, processes in Data.items():
        prefix = resolve_category_name(category)

        for proc_name, proc_meta in processes.items():
            proc_RootFiles      = proc_meta["files"]
            proc_NbRootFiles    = len(proc_RootFiles)
            proc_totalNbEvents  = proc_meta["nb_events"]
            proc_CrossSection   = proc_meta["cross_section"] * 1000

            if not proc_RootFiles or not proc_totalNbEvents:
                continue

            proc_eventWeight = proc_CrossSection * working_luminosity / proc_totalNbEvents
            proc_treeName    = f"{prefix}_{proc_name}"

            print("-" * 80)
            print(f"Processing : {proc_name} ({prefix})")
            print(f" Cross section                  : {proc_CrossSection} fb")
            print(f" Total number of .root files:   : {proc_NbRootFiles}")
            print(f" Total number of events         : {proc_totalNbEvents}")
            print(f" Average weight per event       : {proc_eventWeight} (at {working_luminosity} fb^-1)")
            print("-" * 80)

            proc_paths = {}
            all_objSel, all_evtSel = [], []
            for i, sampleRootFile in enumerate(proc_RootFiles):
                if i > 0: print("-"*80)
                print(f"Sample {i+1}/{proc_NbRootFiles}")   
                print("-"*80)

                sample_treeName = f"{proc_treeName}_sample{i}"
                sample_fileName = f"{proc_treeName}_sample{i}.root"

                result = run_in_parallel(
                    loop_tree,
                    max_workers=max_workers,
                    n_chunks=n_chunks,
                    merge_method=2,
                    inputRootFile=str(sampleRootFile),
                    treeName=proc_treeName,               # <-- same tree name for every sample
                    eventWeight=proc_eventWeight,
                    output_dir=str(output_dir),
                    output_file_name=sample_fileName,
                    overwrite=True,
                    show_progress=show_progress,
                    branches_config_path = branches_config_file
                )

                all_objSel.append(result["ObjectSelector"])
                all_evtSel.append(result["EventSelector"])

                for key, path in result["root_paths"].items():
                    proc_paths.setdefault(key, []).append(path)

            # Merge per-sample files with hadd 
            if merge_proc_samples:
                for key, paths in proc_paths.items():
                    if key is None:
                        ac, ac_r = "", ""
                    else:
                        ac, _, ac_r = key.rpartition("_")
                    target = output_dir / ac / ac_r / f"{proc_treeName}.root"
                    target.parent.mkdir(parents=True, exist_ok=True)

                    if hadd_files(str(target), paths, max_workers=max_workers):
                        for p in paths:
                            try:
                                os.remove(p)
                            except OSError:
                                pass

            # Merge the selectors across samples and print 
            if all_objSel and all_evtSel:
                merged_objSel = ObjectSelector.Merge(all_objSel)
                merged_evtSel = EventSelector.Merge(all_evtSel)
                merged_objSel.PrintObjectSelectionSummary(lum=working_luminosity, event_weight=proc_eventWeight)
                merged_evtSel.PrintEventSelectionSummary(proc_treeName, event_weight=proc_eventWeight, lum=working_luminosity)
                obj_dir = output_dir / "ObjectSelection" / proc_treeName
                obj_dir.mkdir(parents=True, exist_ok=True)
                merged_objSel.DrawObjectSelectionHistograms(output_dir=str(obj_dir))
            
            print(f"Finished working on {proc_name}.\n")


def run_in_background(log_file):
    import subprocess
    log_path = Path(log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    filtered = [a for a in sys.argv[1:] if a not in ("-b", "--background")]
    # The -u flag means "unbuffered", so every print shows up in the log the moment it happens.
    if __package__:
        # We were imported as part of a package (the normal `hepdataset` CLI path),
        # so re-launch as a module: `python -u -m hepdataset.make_dataset ...`
        # This preserves __package__ in the child, so relative imports keep working.
        module_name = f"{__package__}.{Path(__file__).stem}"
        argv = [sys.executable, "-u", "-m", module_name] + filtered
    else:
        # Fallback: we were genuinely run as a bare script (e.g. `python make_dataset.py`),
        # so re-exec the file as-is.
        argv = [sys.executable, "-u", os.path.abspath(__file__)] + filtered

    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  
            close_fds=True, 
        )
    print(f"Started PID {proc.pid}; logs -> {log_path}")
    print(f"If you wish to kill this process, use: kill -TERM -- -{proc.pid}")


def main():
    import argparse
    from datetime import datetime
    from pprint import pprint

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("samples_file", help="Samples YAML file.")
    p.add_argument("--branches-config-file", default=None)
    p.add_argument("--output-dir", default = "HEPDataset", help="Output directory.")
    p.add_argument("--working-luminosity", type=float, default=400.0)
    p.add_argument("--loop-method", default="adaptive_delphes")
    p.add_argument("--merge-proc-samples", action="store_true")
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--n-chunks", type=int, default=None)
    p.add_argument("--show-progress", action="store_true")
    p.add_argument("-b", "--background", action="store_true",  help="Run in background")
    p.add_argument("--log-file", default="hepdataset.log", help="Log file used in background mode (default: hepdataset.log).")
    args = p.parse_args()

    if args.background:
        # Parent: spawn the detached child and exit. Nothing else is printed here —
        # the header belongs to the child, whose stdout is the log file.
        run_in_background(args.log_file)
        return

    # Foreground run OR the detached child (whose argv no longer contains -b).
    # Only this branch's stdout is the log file (in background mode) or the terminal.
    started = datetime.now()
    print(f"=== Started at {started.isoformat(timespec='seconds')} ===", flush=True)
    print(f"Args: {vars(args)}", flush=True)

    kwargs = vars(args).copy()
    kwargs.pop("background", None)
    kwargs.pop("log_file", None)

    try:
        make_dataset(**kwargs)
    except BaseException:
        finished = datetime.now()
        print(f"=== CRASHED at {finished.isoformat(timespec='seconds')} "
              f"(after {finished - started}) ===", flush=True)
        raise
    else:
        finished = datetime.now()
        print(f"=== Finished at {finished.isoformat(timespec='seconds')} "
              f"(total {finished - started}) ===", flush=True)

if __name__ == "__main__":
    main()
