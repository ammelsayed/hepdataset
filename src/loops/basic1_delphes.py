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
    output_dir = None,
    output_file_name = "events.root",
    allow_overwrite = False,
):

    # Read the input file
    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)

    # Basic checks
    numberOfEntries = TreeReader.GetEntries()
    if start_entry < 0 or start_entry > numberOfEntries:
        raise ValueError(f"start_entry must be between 0 and {numberOfEntries}")
    if end_entry is None or end_entry > numberOfEntries:
        end_entry = numberOfEntries
    if end_entry < start_entry:
        raise ValueError("end_entry must be greater than or equal to start_entry")

    numberOfProcessedEntries = end_entry - start_entry
    if eventWeight is None:
        if (cross_section is None) != (luminosity is None):
            raise ValueError("cross section and luminosity must be provided together")
        if cross_section is not None and luminosity is not None:
            if numberOfEntries == 0:
                raise ValueError("Cannot calculate an event weight for an empty ROOT file")
            eventWeight = cross_section * luminosity / numberOfProcessedEntries
        else:
            eventWeight = 1.0

    if show_progress:
        print(f"Reading ROOT file: {inputRootFile}")
        print(f"Total events in file: {numberOfEntries}")
        print(f"Processing events: {start_entry} to {end_entry - 1} ({numberOfProcessedEntries} events)")
        print(f"Event weight: {eventWeight}")
    
    # Check output_dir is None, we use "./HEPDatasetOutput" as defaul
    if output_dir is None:
        output_dir = "./HEPDatasetOutput"

    # Check if the output_dir already exists or not
    # Allow overwrite if allow_overwrite == True
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise ValueError(f"output_dir is not a directory: {output_dir}")
        
        print(f"The output directory exists; overwriting its contents: {output_dir}")
        if not allow_overwrite:
            raise FileExistsError(f"Output directory already exists, cannot write there: {output_dir}")
    else:
        print(f"Creating output directory : {output_dir}")
        os.makedirs(output_dir)
    
    # Check output_file_name
    if not output_file_name or not output_file_name.endswith(".root"):
        raise ValueError("output_file_name must end with .root!")

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
    from parallelization.parallel_loop import add_parallel_arguments, run_in_parallel
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
    parser.add_argument("--debug", action="store_true", help="Show debug information during processing.")
    parser.add_argument("--output-dir", type=str, default=".", metavar="", help="Directory where the output ROOT file will be written (default: current directory).")
    parser.add_argument("--output-file-name", default="events.root", metavar="", help="Name of the output ROOT file.")
    parser.add_argument("--allow-overwrite", action="store_true", metavar="", help="Allow overwriting the output directory if it already exists.")
    
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
        "allow_overwrite" : args.allow_overwrite,
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