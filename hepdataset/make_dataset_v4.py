#!/usr/bin/env python3
"""
make_dataset_v3.py

Build ML-ready tabular datasets from Delphes ROOT files.

Memory-efficient variant of make_dataset_v2.py: in the parallel path each chunk
worker spills its per-SR trees to individual temp .root files (one per signal
region) and returns the file paths.  Chunks are then merged via hadd, and the
per-sample files are finally hadd'd into the output ReaderOutput ROOT files.
This avoids holding large TTrees in memory or writing detached trees into
long-lived TFiles (which caused corruption in earlier attempts).

Responsibilities:
    - Read background ROOT paths from bkg_directories.yml
    - Select objects according to SR definitions, read the required kinematics
    - Flatten event-level information into a tabular format (ROOT Tree Bracnhs)
    - Write output datasets in Parquet or CSV format using Pandas.

Author : A.M.M. Elsayed (University of Science and Technology of China)
Email  : ammelsayed@mail.ustc.edu.cn / ahmedphysica@outlook.com
"""

import os
import sys
import time
import uuid
import shutil
import tempfile
import subprocess
import numpy as np
import pandas as pd
import itertools
import tabulate
from .kinematics import EventShapes, Centrality, MtW
from .branch_names import (

    get_float_branch_names, get_int_branch_names, get_obj_repr, get_obj_count, get_obj_kinematics,
    get_obj_instances, get_nbody_combinations, get_nbody_kinematics, print_summary
)
from yaml import safe_load as yml_safe_load
from tqdm import tqdm
from mt2 import mt2
import ROOT
from .parallelization import parallel_runs, format_time
from .py_loop.adaptive import loop_tree as _adaptive_loop_tree
DELPHES_PATH = os.environ.get("DELPHES_HOME", "/home/ammelsayed/softwares/MG5_aMC_v3_5_15/Delphes")
ROOT.gInterpreter.AddIncludePath(DELPHES_PATH)
ROOT.gInterpreter.AddIncludePath(f"{DELPHES_PATH}/classes")
ROOT.gInterpreter.AddIncludePath(f"{DELPHES_PATH}/external")
ROOT.gSystem.Load("libDelphes")
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
ROOT.gInterpreter.Declare('#include "classes/SortableObject.h"')
ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("ATLAS")
print("Using ROOT version:", ROOT.__version__)
print("Using Delphes libraries found at:", DELPHES_PATH)
script_nb_version = 3
print(f"Making datasets with script version : {script_nb_version}")

## =============================
## Run options
## =============================

debug = 1
debug_loop = 0
LooseSR = True
multiObjects_Nmax = 3
multiObjects_include_same_represenations = True
multiObjects_include_trival_kinematics = False
multiObjects_include_mt2 = True # only calculated for 2-body objects (leptons and fatjets (see main lop))
multiObjects_combo_objects_set = ("Lepton", "FatJet", "MET")

print_summary()

# test options (set to false for production)
testCode = False
testSamplePaths = ["/data/ammelsayed/stuff/root_files/tt012j_MLM_run_01_01.root", "/data/ammelsayed/stuff/root_files/tt012j_MLM_run_01_02.root"]

# output directories
baseDir = f"ReaderOutput_v{script_nb_version}" if not testCode else f"ReaderOutput_Test_v{script_nb_version}"
outputRootFileName = "mc_events.root"

# where to spill per-chunk/merge temp .root files
tmpDir = os.path.dirname(os.path.abspath(__file__))
tempReaderDir = os.path.join(tmpDir, "TempReaderOutput")


## Main reader loop and quality selections
## ==========================================



def loop_tree(inputRootFile, treeName, sampleWeight, signal_regions_keys, start_entry=0, end_entry=None, progress=True, debug_loop = False, max_entries=None, temp_dir_path=None):
    """Thin wrapper around :func:`hepdataset.py_loop.adaptive.loop_tree`.

    The heavy event loop code now lives in ``py_loop/adaptive.py`` to keep this
    file focused on the high-level pipeline.  We pass the v4 script's helpers
    (isGoodMuon, classify_signal_region_key, ...) and configuration globals
    (multiObjects_Nmax, LooseSR, ...) as arguments so the loop implementation
    remains version-agnostic.
    """
    return _adaptive_loop_tree(
        inputRootFile,
        treeName,
        sampleWeight,
        signal_regions_keys,
        start_entry=start_entry,
        end_entry=end_entry,
        progress=progress,
        debug_loop=debug_loop,
        max_entries=max_entries,
        temp_dir_path=temp_dir_path,
        # -- helpers --
        build_chain=build_chain,
        isGoodMuon=isGoodMuon,
        isGoodElectron=isGoodElectron,
        classify_signal_region_key=classify_signal_region_key,
        lepton_type_fn=lepton_type,
        # -- configuration knobs (defined at module top in make_dataset_v4.py) --
        multiObjects_Nmax=multiObjects_Nmax,
        multiObjects_include_same_represenations=multiObjects_include_same_represenations,
        multiObjects_include_trival_kinematics=multiObjects_include_trival_kinematics,
        multiObjects_include_mt2=multiObjects_include_mt2,
        multiObjects_combo_objects_set=multiObjects_combo_objects_set,
        isLooseSR=LooseSR,
    )



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


def read(processes, signal_regions, signal_regions_keys):

    print("\n")
    
    # output root files paths (final destination)
    paths = {}
    for channelName, regionNamesList in signal_regions.items():
        for regionName in regionNamesList:
            key = f"{channelName}_{regionName}"
            paths[key] = os.path.join(baseDir, channelName, regionName, outputRootFileName)

    # Ensure TempReaderOutput exists
    os.makedirs(tempReaderDir, exist_ok=True)

    # Collect per-sample merged files for each SR, then hadd into final output
    # sample_files[sr_key] = list of per-sample ROOT file paths in TempReaderOutput
    sample_files = {sr_key: [] for sr_key in signal_regions_keys}
    
    for proc, metadata in processes.items():
        inputRootFilesList = metadata['dir']
        TotalCrossSection = metadata['cross_section_[pb]']

        if not inputRootFilesList:
            print(f"Skipping {proc}: no input files.")
            continue

        print(f"Processing {proc}: {len(inputRootFilesList)} sample root files.")
        numEvents = count_entries(inputRootFilesList)
        if numEvents == 0:
            print(f"Skipping {proc}: zero events.")
            continue
        perEventWeight = TotalCrossSection * 1000 * 400 / numEvents
        print(f"    > Total number of events : {numEvents}")
        print(f"    > Per-event weight : {perEventWeight}")

        for idx, inputRootFile in enumerate(inputRootFilesList, start = 1):
            treeName = f"{proc}_sample{idx}"
            merged_sample = loop_tree_advanced(inputRootFile, treeName, sampleWeight=perEventWeight, signal_regions_keys=signal_regions_keys)
            for sr_key, sample_path in merged_sample.items():
                sample_files[sr_key].append(sample_path)
                print(f"  {treeName}: merged {sample_path} -> will go into {paths[sr_key]}")
            print("\n")

    # Final merge: hadd all per-sample files for each SR into the final output
    print("Final merge into ReaderOutput ..")
    for sr_key in signal_regions_keys:
        if not sample_files[sr_key]:
            continue
        final_path = paths[sr_key]
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        ok = hadd_files(final_path, sample_files[sr_key])
        if ok:
            n_samples = len(sample_files[sr_key])
            print(f"  Final: {n_samples} sample files -> {final_path}")
        else:
            print(f"  WARNING: final hadd failed for {sr_key}")

    # Clean up TempReaderOutput
    if os.path.isdir(tempReaderDir):
        shutil.rmtree(tempReaderDir, ignore_errors=True)
        print(f"Cleaned up {tempReaderDir}")


def make_datasets():

    signal_regions, signal_regions_keys = get_signal_regions(isLoose = LooseSR)
    print("Signal regions:")
    for key, values in signal_regions.items():
        print(f"{key:<5} : {values}")

    read(get_processes(), signal_regions, signal_regions_keys)

def test():
    proc = {
    "test1_s": {
        "dir": clean(testSamplePaths), "cross_section_[pb]": 1,
    }}
    read(proc, {"test" : ["test1"]}, ["test_test1"])


if __name__ == '__main__':

    if testCode:
        test()
    else:
        make_datasets()


