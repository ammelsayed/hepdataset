#!/usr/bin/env python3
"""Process every ROOT file in a samples YAML card and merge per process."""

import argparse
import importlib
import shutil
import sys
from pathlib import Path


def resolve_category_name(category):
    cat = str(category).lower()
    return "bkg" if cat == "background" else "sig" if cat == "signal" else "Unknown"

def _load_loop(name):
    """Import a loop module from src/loops and return its loop_tree."""
    loops_dir = Path(__file__).resolve().parent / "loops"
    if str(loops_dir) not in sys.path:
        sys.path.insert(0, str(loops_dir))
    return importlib.import_module(Path(name).stem).loop_tree


def _append_tree(output_path, tree_name, sample_paths):
    """Merge sample_paths into one TTree named tree_name inside output_path."""
    import ROOT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = ROOT.TFile.Open(str(output_path), "UPDATE")
    open_files = []
    tlist = ROOT.TList()
    for p in sample_paths:
        f = ROOT.TFile.Open(str(p))
        if not f or f.IsZombie():
            continue
        open_files.append(f)
        for k in f.GetListOfKeys():
            obj = f.Get(k.GetName())
            if obj and obj.InheritsFrom("ROOT.TTree"):
                tlist.Add(obj)
                break
    if tlist.GetEntries() > 0:
        merged = ROOT.TTree.MergeTrees(tlist)
        merged.SetName(tree_name)
        target.cd()
        merged.Write()
    for f in open_files:
        f.Close()
    target.Close()


def make_dataset(
    samples, 
    output_dir, 
    luminosity=400.0, 
    loop_file="adaptive_delphes",
    merge=False,
    max_workers=None, 
    n_chunks=None,
    show_progress=False, 
    ):

    # Import the required libraries
    import sys
    loops_dir = str(Path(__file__).resolve().parent / "loops")
    if loops_dir not in sys.path:
        sys.path.insert(0, loops_dir)
    from samples_reader import SamplesReader
    from loops.delphes import load_delphes
    from loops.parallel_loop import run_in_parallel, merge_summaries
    from loops.object_selection import PrintObjectSelectionSummary, PrintAnalysisChannelYields, DrawObjectSelectionHistograms

    loop_tree = _load_loop(loop_file)
    load_delphes()
    output_dir = Path(output_dir).resolve()


    for category, processes in SamplesReader(str(samples)).read().items():

        prefix = resolve_category_name(category)

        for proc_name, proc_meta in processes.items():

            proc_RootFiles = proc_meta["files"]
            proc_NbRootFiles = len(proc_RootFiles)
            proc_totalNbEvents  = proc_meta["nb_events"]
            proc_CrossSection  = proc_meta["cross_section"] * 1000

            if not proc_RootFiles or not proc_totalNbEvents :
                continue

            proc_eventWeight  = proc_CrossSection * luminosity / proc_totalNbEvents 
            proc_treeName  = f"{prefix}_{proc_name}"

            print("-"*80)
            print(f"Processing : {proc_name} ({prefix})")
            print(f" Cross section                  : {proc_CrossSection} fb")
            print(f" Total number of .root files:   : {proc_NbRootFiles}")
            print(f" Total number of events         : {proc_totalNbEvents }")
            print(f" Average weight per event       : {proc_eventWeight} (at {luminosity} fb^-1)")
            print("-"*80)

            summaries, proc_paths = [], {}
            for i, sampleRootFile in enumerate(proc_RootFiles):
                if i > 0: print("-"*80)
                print(f"Sample {i+1}/{proc_NbRootFiles}")   
                print("-"*80)

                sample_treeName = f"{proc_treeName}_sample{i}"

                result, sample_summaries  = run_in_parallel(
                    loop_tree,

                    max_workers=max_workers,
                    n_chunks=n_chunks,
                    merge_method = 2,

                    inputRootFile=str(sampleRootFile),
                    treeName = sample_treeName,
                    eventWeight = proc_eventWeight,
                    output_dir=str(output_dir),
                    output_file_name= f"{sample_treeName}.root",
                    overwrite=True,
                    show_progress=False,
                    return_summary=True,
                )

                if sample_summaries:
                    summaries.extend(sample_summaries) 
                if not isinstance(result, dict):
                    result = {None: result}
                for key, path in result.items():
                    proc_paths.setdefault(key, []).append(path)

            if merge:
                for key, paths in proc_paths.items():
                    if key is None:
                        target = output_dir / "events.root"
                    else:
                        ac, _, ac_r = key.rpartition("_")
                        target = output_dir / ac / ac_r / "events.root"
                    _append_tree(target, proc_treeName, paths)

            if summaries:
                merged = merge_summaries(summaries)
                PrintObjectSelectionSummary(merged["objSel_cutflow"], lum=luminosity, event_weight=proc_eventWeight)
                PrintAnalysisChannelYields(merged["ac_counts"], proc_treeName, event_weight=proc_eventWeight, lum=luminosity)
                obj_dir = output_dir / "ObjectSelection" / proc_treeName
                obj_dir.mkdir(parents=True, exist_ok=True)
                DrawObjectSelectionHistograms(merged["objSel_h"], output_dir=str(obj_dir))

    if merge:
        shutil.rmtree(samples_dir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("samples", help="Samples YAML file.")
    p.add_argument("output_dir", help="Output directory.")
    p.add_argument("--luminosity", type=float, default=400.0)
    p.add_argument("--loop-file", default="adaptive_delphes")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--n-chunks", type=int, default=None)
    p.add_argument("--show-progress", action="store_true")
    args = p.parse_args()
    make_dataset(**vars(args))


if __name__ == "__main__":
    main()