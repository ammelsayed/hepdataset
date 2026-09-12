#!/usr/bin/env python3

"""
Supports reading of one Delphes root file at a time, and writing a flat tree with selected events to a new root file.
Requires an event to contain exactly one lepton and at least one fatjet to be selected.
More branches are supported compared to basic1_delphes.py, including pairwise kinematic variables and global event variables.
"""

import os
import ROOT
import math
import argparse
import numpy as np
from tqdm import tqdm
from delphes import load_delphes, build_chain
from kinematics import DeltaR, DeltaPhi, DeltaEta
from object_selection import select_objects
from itertools import combinations
from parallelization.parallel_loop import add_parallel_arguments, run_in_parallel

def loop_tree(
    inputRootFile,
    treeName = "Delphes",
    eventWeight = None,
    cross_section = None,
    luminosity = None,
    start_entry = 0,
    end_entry = None,
    show_progress = True,
    debug_loop = False,
    output_dir = None,
    output_file_name = "events.root",
):

    # Read the input file
    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)

    # Branches to read
    Electron_branch  = TreeReader.UseBranch("Electron")
    Muon_branch      = TreeReader.UseBranch("Muon")
    FatJet_branch    = TreeReader.UseBranch("FatJet")
    Jet_branch       = TreeReader.UseBranch("Jet")
    MissingET_branch = TreeReader.UseBranch("MissingET")
    ScalarHT_branch  = TreeReader.UseBranch("ScalarHT")
    Weight_branch    = TreeReader.UseBranch("Weight")

    # Setup branches to write
    nb_lep_max = 3
    nb_fj_max = 2
    branch_names  = [f"{k}_Lepton{i}" for k in ["PT", "Eta", "Phi"] for i in range(nb_lep_max)]
    branch_names += [f"{k}_FatJet{i}" for k in ["PT", "Eta", "Phi", "Mass"] for i in range(nb_fj_max)]
    branch_names += [f"{k}_{'_'.join(comb)}" for k in ["DeltaR", "DeltaPhi", "DeltaEta", "M"] for comb in combinations([f"Lepton{i}" for i in range(nb_lep_max)] + [f"FatJet{i}" for i in range(nb_fj_max)], 2)]
    branch_names += ["MET", "HT", "LT", "ST", "Meff"]
    branch_names += ["weight", "gen_weight"]

    # book the tree and create the branch buffers
    b = {} 
    tree = ROOT.TTree(treeName, treeName)
    tree.SetDirectory(0)
    for name in branch_names:
        b[name] = np.zeros(1, dtype=np.float64)
        tree.Branch(name, b[name], f"{name}/D")

    # Prepare count dict to count number of events going to each analysis channel:
    # Also prepare some dictonaries to loging object selection cutflow
    counts = {k: 0 for k in ac_keys}
    objSel_cutflow = {
        "lepton" : {"initial" : 0},
        "fatjet" : {"initial" : 0}
    }

    # Event loop
    numberOfEntries = TreeReader.GetEntries()
    if end_entry is None or end_entry > numberOfEntries:
        end_entry = numberOfEntries

    rng = range(start_entry, end_entry)
    for entry in (tqdm(rng) if show_progress else rng):
        
        TreeReader.ReadEntry(entry)

        # Reset branches to np.nan (weight defaults to 0.0 for skipped events)
        for name in b:
            b[name][0] = np.nan
        b["weight"][0] = 0.0

        # Object selection
        selected_objects = select_objects(Muon_branch, Electron_branch, FatJet_branch, Jet_branch, objSel_cutflow)
        goodFatJets = selected_objects["goodFatJets"]
        goodLeptons = selected_objects["goodLeptons"]
        goodJets = selected_objects["goodJets"]
        goodBJets = selected_objects["goodBJets"]
        goodTauJets = selected_objects["goodTauJets"]

        # A dictonary to hold the selected objects' four-momenta for easier access
        P4s = {}

        # Fill lepton data
        if len(goodLeptons) > 0:
            for idx, lep in enumerate(goodLeptons[:nb_lep_max]):
                P4s[f"Lepton{idx}"] = lep.P4()
                b[f"PT_Lepton{idx}"][0]  = lep.PT
                b[f"Eta_Lepton{idx}"][0] = lep.Eta
                b[f"Phi_Lepton{idx}"][0] = lep.Phi
                
        
        # Fill fatjet data
        if len(goodFatJets) > 0:
            for idx, fj in enumerate(goodFatJets[:nb_fj_max]):
                P4s[f"FatJet{idx}"] = fj.P4()
                b[f"PT_FatJet{idx}"][0]   = fj.PT
                b[f"Eta_FatJet{idx}"][0]  = fj.Eta
                b[f"Phi_FatJet{idx}"][0]  = fj.Phi
                b[f"Mass_FatJet{idx}"][0] = fj.SoftDroppedP4[0].M()
        
        # Fill pairwise kinematic variables
        for obj1, obj2 in combinations([f"Lepton{i}" for i in range(nb_lep_max)] + [f"FatJet{i}" for i in range(nb_fj_max)], 2):
            if obj1 in P4s and obj2 in P4s:
                sufx = f"{obj1}_{obj2}"
                p4_1, p4_2 = P4s[obj1], P4s[obj2]
                b[f"DeltaR_{sufx}"][0] = p4_1.DeltaR(p4_2)
                b[f"DeltaPhi_{sufx}"][0] = p4_1.DeltaPhi(p4_2)
                b[f"DeltaEta_{sufx}"][0] = p4_1.Eta() - p4_2.Eta()
                b[f"M_{sufx}"][0]        = (p4_1 + p4_2).M()
        
        # Fill global event variables
        met = MissingET_branch.At(0).MET
        ht  = ScalarHT_branch.At(0).HT
        lt  = sum(lep.PT for lep in goodLeptons)
        st  =  ht + lt
        meff = st + met
        b["MET"][0]  = met
        b["HT"][0]   = ht
        b["LT"][0]   = lt
        b["ST"][0]   = st
        b["Meff"][0] = meff

        # Fill event weights
        w_gen = Weight_branch.At(0).Weight
        b["gen_weight"][0] = w_gen
        b["weight"][0] = eventWeight

        tree.Fill()

    # if output_dir is given, then write the tree into root file.
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, output_file_name)
        f_out = ROOT.TFile.Open(path, "RECREATE")
        tree.SetDirectory(f_out)
        tree.Write()
        f_out.Close()
        return path
    else:
        return tree

if __name__ == "__main__":

    from pprint import pprint
    load_delphes()  
    
    parser = argparse.ArgumentParser(description="Process a Delphes ROOT file and write a flat tree with selected events.")
    parser.add_argument("input_root_file",  type=str, help="Path to the input Delphes ROOT file.")
    parser.add_argument("--tree-name", type=str, default="Delphes", metavar="", help="Name of the output TTree (default: Delphes).")
    parser.add_argument("--event-weight", type=float, default=None, metavar="", help="Weight to apply to each event. If omitted, calculate it from cross section and luminosity.")
    parser.add_argument("--cross-section", type=float, default=None, metavar="", help="Cross section for the process in fb. Used with --luminosity when --event-weight is omitted.")
    parser.add_argument("--luminosity", type=float, default=None, metavar="", help="Target integrated luminosity in fb^-1. Used with --cross_section when --event-weight is omitted.")
    parser.add_argument("--start-entry", type=int, default=0, metavar="", help="Entry to start processing from (default: 0).")
    parser.add_argument("--end-entry", type=int, default=None, metavar="", help="Entry to stop processing at (default: None, meaning process all entries).")
    parser.add_argument("--show-progress", action="store_true", help="Show a progress bar during processing.")
    parser.add_argument("--output-file-name", default="events.root", metavar="", help="Name of the output ROOT file.")
    parser.add_argument("--debug", action="store_true", help="Show debug information during processing.")
    parser.add_argument("--output-dir", type=str, default=".", metavar="", help="Directory where the output ROOT file will be written (default: current directory).")
    add_parallel_arguments(parser)

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    loop_kwargs = {
        "inputRootFile" : args.input_root_file,
        "treeName" : args.tree_name,
        "eventWeight" : args.event_weight,
        "cross_section" : args.cross_section,
        "luminosity" : args.luminosity,
        "start_entry" : args.start_entry,
        "end_entry" : args.end_entry,
        "show_progress" : args.show_progress,
        "debug_loop" : args.debug,
        "output_dir" : args.output_dir,
        "output_file_name" : args.output_file_name,
    }

    if args.parallel:
        out_path = run_in_parallel(
            loop_tree_method=loop_tree,
            max_workers=args.max_workers,
            n_chunk=args.n_chunks,
            merge_method=args.merge_method,
            temp_dir=args.temp_dir,
            **loop_kwargs
        )
    else:
        out_path = loop_tree(**loop_kwargs)

    print(f"\nOutput written to:")
    pprint(out_path)
