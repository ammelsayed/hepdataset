#!/usr/bin/env python3
import argparse, os, re
import ROOT

def get_tree_name(path):
    f = ROOT.TFile.Open(path)
    for k in f.GetListOfKeys():
        if k.GetClassName() == "TTree":
            name = k.GetName()
            f.Close()
            return name
    f.Close()
    return None

def get_tree_entries(path):
    f = ROOT.TFile.Open(path)
    for k in f.GetListOfKeys():
        if k.GetClassName() == "TTree":
            n = f.Get(k.GetName()).GetEntries()
            f.Close()
            return n
    f.Close()
    return 0

def merge_samples(path):
    groups = {}
    for fn in os.listdir(path):
        m = re.match(r"^(bkg_|sig_)(.+)_sample\d+\.root$", fn)
        if m:
            process = m.group(1) + m.group(2)
            groups.setdefault(process, []).append(fn)

    for process, files in groups.items():
        files.sort()
        full_paths = [os.path.join(path, f) for f in files]
        out_path   = os.path.join(path, process + ".root")

        # Expected total = sum of entries of every source tree
        expected = sum(get_tree_entries(f) for f in full_paths)

        # Build a chain that knows the tree name inside EACH file
        chain = ROOT.TChain()
        for f in full_paths:
            chain.AddFile(f, -1, get_tree_name(f))

        out = ROOT.TFile(out_path, "RECREATE")
        ROOT.SetOwnership(out, False)
        out.cd()

        tree = chain.CloneTree(-1, "fast")
        tree.SetName(process)
        written = tree.GetEntries()

        out.Write()
        out.Close()

        # Safety: only delete sources if we actually wrote everything
        if written != expected:
            print(f"!! {process}: wrote {written}, expected {expected} — keeping sources, aborting")
            os.remove(out_path)   # remove the bad output
            continue

        for f in full_paths:
            os.remove(f)

def merge_all(path):
    """
    Merge samples first, then combine every per-process file into events.root,
    then delete the per-process files.
    """
    merge_samples(path)

    per_process = [
        fn for fn in sorted(os.listdir(path))
        if (fn.startswith("bkg_") or fn.startswith("sig_"))
        and fn.endswith(".root")
        and fn != "events.root"
    ]

    out_path = os.path.join(path, "events.root")
    out = ROOT.TFile(out_path, "RECREATE")
    ROOT.SetOwnership(out, False)     # keep the output file alive too

    for fn in per_process:
        fi = ROOT.TFile.Open(os.path.join(path, fn))

        for k in fi.GetListOfKeys():
            if k.GetClassName() != "TTree":
                continue

            src  = fi.Get(k.GetName())
            name = os.path.splitext(fn)[0]   # tree name = file base name

            out.cd()                          # clone target = events.root
            tree = src.CloneTree(-1, "fast")
            ROOT.SetOwnership(tree, False)    # don't let PyROOT delete it
            tree.SetName(name)
            tree.Write()                      # write it now, with the new name

        fi.Close()

    out.Close()

    # Delete the per-process files now that events.root has them
    for fn in per_process:
        os.remove(os.path.join(path, fn))

def main():
    import argparse
    p = argparse.ArgumentParser(description="Merge per-sample ROOT files.")
    p.add_argument("path", help="Directory containing the ROOT files")
    p.add_argument("--merge-samples", action="store_true",  help="Merge samples of the same process and delete originals")
    p.add_argument("--merge", action="store_true",  help="Merge samples, combine into events.root, delete per-process files")
    args = p.parse_args()

    if args.merge:
        merge_all(args.path)
    elif args.merge_samples:
        merge_samples(args.path)


if __name__ == "__main__":
    main()