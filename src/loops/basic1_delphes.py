#!/usr/bin/env python3

"""
Supports reading of one Delphes root file at a time, and writing a flat tree with selected events to a new root file.
Requires an event to contain exactly one lepton and at least one fatjet to be selected.
Supports limited amount of branches.
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
    FatJet_branch   = TreeReader.UseBranch("FatJet")
    Electron_branch = TreeReader.UseBranch("Electron")
    Muon_branch     = TreeReader.UseBranch("Muon")
    Event_branch    = TreeReader.UseBranch("Event")

    # Setup the tree
    tree = ROOT.TTree(treeName, treeName)
    tree.SetDirectory(0)

    # Setup branches to write
    branch_names = [
        "PT_Muon0",     "Eta_Muon0",     "Phi_Muon0",
        "PT_Electron0", "Eta_Electron0", "Phi_Electron0",
        "PT_FatJet0",   "Eta_FatJet0",   "Phi_FatJet0",   "M_FatJet0",
        "PT_FatJet1",   "Eta_FatJet1",   "Phi_FatJet1",   "M_FatJet1",
        "DeltaR_Lepton_FatJet0",   "DeltaPhi_Lepton_FatJet0",   "DeltaEta_Lepton_FatJet0",
        "DeltaR_Lepton_FatJet1",   "DeltaPhi_Lepton_FatJet1",   "DeltaEta_Lepton_FatJet1",
        "M_Lepton_FatJet0", "M_Lepton_FatJet1",
        "weight",
    ]

    b = {} 
    for name in branch_names:
        b[name] = np.zeros(1, dtype=np.float64)
        tree.Branch(name, b[name], f"{name}/D")

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
        selected_objects = select_objects(FatJet_branch, Electron_branch, Muon_branch)
        goodFatJets = selected_objects["goodFatJets"]
        goodLeptons = selected_objects["goodLeptons"]

        # Event selection (analysis channels configuration)
        # Exactly one lepton and at least one fatjet
        if not (len(goodLeptons) == 1 and len(goodFatJets) >= 1):
            continue

        # Sort fatjets by PT (descending)
        goodFatJets.sort(key=lambda fj: fj.PT, reverse=True)

        # Fill lepton branches
        lepton = goodLeptons[0]
        if lepton.ClassName() == "Muon":
            b["PT_Muon0"][0]  = lepton.PT
            b["Eta_Muon0"][0] = lepton.Eta
            b["Phi_Muon0"][0] = lepton.Phi
        else:
            b["PT_Electron0"][0]  = lepton.PT
            b["Eta_Electron0"][0] = lepton.Eta
            b["Phi_Electron0"][0] = lepton.Phi

        # Fill fatjet + lepton–fatjet pair branches
        for idx, fj in enumerate(goodFatJets[:2]):
            tag = f"FatJet{idx}"
            b[f"PT_{tag}"][0]   = fj.PT
            b[f"Eta_{tag}"][0]  = fj.Eta
            b[f"Phi_{tag}"][0]  = fj.Phi
            b[f"M_{tag}"][0]    = fj.SoftDroppedP4[0].M()
            b[f"DeltaR_Lepton_{tag}"][0]   = DeltaR(lepton, fj)
            b[f"DeltaPhi_Lepton_{tag}"][0] = DeltaPhi(lepton, fj)
            b[f"DeltaEta_Lepton_{tag}"][0] = DeltaEta(lepton, fj)
            b[f"M_Lepton_{tag}"][0]        = (fj.P4() + lepton.P4()).M()

        # Weight
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