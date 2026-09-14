#!/usr/bin/env python3
import argparse
import os
import re
import glob
from collections import defaultdict
import uproot

def merge_samples(path):
    # Dictionary to group files by process name (e.g., "bkg_tt_2L", "sig_M1000")
    groups = defaultdict(list)
    
    # Find all .root files in the given path
    for f in glob.glob(os.path.join(path, "*.root")):
        # Match filenames like "bkg_tt_2L_sample0.root" or "sig_M1000_sample0.root"
        # m.group(1) captures "bkg_" or "sig_"
        # m.group(2) captures the process name like "tt_2L" or "M1000"
        m = re.match(r'^(bkg_|sig_)(.+?)_sample\d+\.root$', os.path.basename(f))
        if m:
            prefix = m.group(1) + m.group(2)
            groups[prefix].append(f)
            
    for prefix, flist in groups.items():
        # Open all sample files for this specific process
        handles = [uproot.open(f) for f in flist]
        # Extract the tree from each file (assuming one main tree per file)
        trees = [fh[list(fh.keys())[0]] for fh in handles]
        
        # Create a new file named e.g. "bkg_tt_2L.root"
        with uproot.recreate(os.path.join(path, prefix + ".root")) as out:
            # Assigning a list of trees to out[prefix] concatenates them automatically
            # The resulting merged tree will be named e.g. "bkg_tt_2L"
            out[prefix] = trees
            
        # Delete the original sample files to clean up the directory
        for f in flist:
            os.remove(f)

def merge_all(path):
    # First, run the sample merging step so we only have one file per process
    merge_samples(path)
    
    # Gather all merged process files
    all_files = sorted(glob.glob(os.path.join(path, "bkg_*.root")) + 
                       glob.glob(os.path.join(path, "sig_*.root")))
                   
    # Create the final combined file "events.root"
    with uproot.recreate(os.path.join(path, "events.root")) as out:
        for f in all_files:
            # The tree name will be the file name without the ".root" extension
            name = os.path.basename(f)[:-5]
            with uproot.open(f) as fh:
                # Copy the tree into events.root, keeping its distinct name
                out[name] = fh[list(fh.keys())[0]]
                
    # Delete the individual process files after events.root is safely written
    for f in all_files:
        os.remove(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--merge-samples", action="store_true")
    p.add_argument("--merge", action="store_true")
    a = p.parse_args()
    
    # Convert to absolute path so the script works correctly from anywhere
    path = os.path.abspath(a.path)
    
    if a.merge:
        merge_all(path)
    elif a.merge_samples:
        merge_samples(path)

if __name__ == "__main__":
    main()