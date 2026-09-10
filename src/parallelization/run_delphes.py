
"""
Run multiple Delphes jobs in parallel on a set of HEPMC files. 
The script will look for HEPMC files in the specified directory 
and its subdirectories (up to a certain depth), run Delphes on 
each file, and save the output ROOT files in the specified output 
directory. It also handles logging and optional deletion of the 
original HEPMC files after successful processing.
"""

import os
import argparse
import subprocess
from pathlib import Path

# Works for all python versions
def find_files(directory, ends_with=".root", max_depth=2):
    directory = os.path.abspath(directory)
    found = set()
    for root, dirs, files in os.walk(directory):
        depth = root[len(directory):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []  # don't descend further
        for f in files:
            if f.endswith(ends_with):
                found.add(os.path.join(root, f))  
    return sorted(found)

def check_hepmc(path):
    p = Path(path)
    return p.is_file() and p.stat().st_size > 0

def get_custom_name(path):
    p = Path(path)
    return f"{p.parents[2].name}_{p.parent.name}"

def run_delphes(hepmc_path, delphes_card, output_dir, exe, overwrite=True, auto_remove_hepmc = True):
    out = Path(output_dir)
    name = get_custom_name(hepmc_path)
    root_out = out / f"{name}.root"   
    log_out = out / f"{name}.log"
    if root_out.exists() and not overwrite:
        return 0

    cmd = [exe, str(delphes_card), str(root_out), str(hepmc_path)]
    with open(log_out, "w") as log:

        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    
        # Delete HEPMC file only if Delphes succeeded
        if result.returncode == 0 and auto_remove_hepmc:
            try:
                os.remove(hepmc_path)
            except OSError as e:
                log.write(f"Warning: Could not delete {hepmc_path}: {e}\n")

        return result.returncode

if __name__ == "__main__":

    from __init__ import parallel_runs
    from ROOT import gInterpreter, gSystem
    from ROOT import __version__ as rootVersion

    parser = argparse.ArgumentParser(
        description="Run DelphesHepMC2 on HEPMC files found under an input directory."
    )
    parser.add_argument(
        "--delphes-path",
        default=os.environ.get("DELPHES_HOME"),
        help="Path to the Delphes installation (defaults to DELPHES_HOME)."
    )
    parser.add_argument(
        "--delphes-card", 
        required=True, 
        help="Path to the Delphes card."
    )
    parser.add_argument(
        "--input-dir", 
        required=True, 
        help="Directory containing HEPMC files."
    )
    parser.add_argument(
        "--output-dir",
         required=True, 
         help="Directory for Delphes ROOT files and logs."
    )
    parser.add_argument(
        "--nb-cores-delphes",
        type=int,
        default=4,
        help="Number of workers used to run Delphes jobs (default: 4).",
    )
    parser.add_argument(
        "--ends-with",
        default=".hepmc",
        help="File suffix to process (default: .hepmc).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum directory depth to search (default: 3).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing ROOT outputs (default: true).",
    )
    parser.add_argument(
        "--auto-remove-hepmc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove HEPMC files after successful processing (default: true).",
    )
    args = parser.parse_args()

    if not args.delphes_path:
        parser.error("--delphes-path is required when DELPHES_HOME is not set")
    if args.nb_cores_delphes < 1:
        parser.error("--nb-cores-delphes must be at least 1")
    if args.max_depth < 0:
        parser.error("--max-depth must be non-negative")

    DELPHES_PATH = args.delphes_path
    gInterpreter.AddIncludePath(DELPHES_PATH)
    gInterpreter.AddIncludePath(f"{DELPHES_PATH}/classes")
    gInterpreter.AddIncludePath(f"{DELPHES_PATH}/external")
    gSystem.Load("libDelphes")
    gInterpreter.Declare('#include "classes/DelphesClasses.h"')
    gInterpreter.Declare('#include "classes/SortableObject.h"')
    gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
    print("Using ROOT version:", rootVersion)
    print("Using Delphes libraries found at:", DELPHES_PATH)

    exe = os.path.join(DELPHES_PATH, "DelphesHepMC2")
    card = args.delphes_card
    outdir = args.output_dir

    os.makedirs(outdir, exist_ok=True)
    files = [
        f
        for f in find_files(args.input_dir, ends_with=args.ends_with, max_depth=args.max_depth)
        if check_hepmc(f)
    ]
    args_list = [
        (f, card, outdir, exe, args.overwrite, args.auto_remove_hepmc)
        for f in files
    ]
    parallel_runs(
        run_delphes,
        args_list,
        max_workers=args.nb_cores_delphes,
        info="Delphes jobs",
    )