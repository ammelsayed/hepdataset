#!/usr/bin/env python3

"""
Supports reading of one Delphes root file at a time, and writing a flat tree with selected events to a new root file.
Requires an event to contain exactly one lepton and at least one fatjet to be selected.
Supports limited amount of branches.
"""

import os
import ROOT
import math
import numpy as np
from tqdm import tqdm
from delphes import load_delphes, build_chain
from kinematics import DeltaR, DeltaPhi, DeltaEta
from object_selection import select_objects
from loop_utilis import apply_loop_defaults, check_start_end_entries, get_event_weight, prepare_output_dir
from itertools import combinations
from tabulate import tabulate

def loop_tree(**loop_args):
    loop_args = apply_loop_defaults(loop_args)

    inputRootFile = loop_args["inputRootFile"]
    treeName      = loop_args["treeName"]
    show_progress = loop_args["show_progress"]
    debug_loop    = loop_args["debug_loop"]
    output_dir    = loop_args["output_dir"]
    output_file_name = loop_args["output_file_name"]

    # Read the input file
    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)

    # basic checks
    numberOfEntries = TreeReader.GetEntries()
    start_entry, end_entry = check_start_end_entries(numberOfEntries, loop_args["start_entry"], loop_args["end_entry"])
    numberOfProcessedEntries = end_entry - start_entry
    eventWeight = get_event_weight(
        loop_args["eventWeight"],
        loop_args["cross_section"],
        loop_args["luminosity"],
        numberOfProcessedEntries,
        numberOfEntries,
    )
    output_dir = prepare_output_dir(output_dir, loop_args["overwrite"])
    if output_dir is not None and not output_file_name.endswith(".root"):
        raise ValueError("output_file_name must end with .root!")
    if show_progress:
        print(f"Reading ROOT file: {inputRootFile}")
        print(f"Total events in file: {numberOfEntries}")
        print(f"Processing events: {start_entry} to {end_entry - 1} ({numberOfProcessedEntries} events)")
        print(f"Event weight: {eventWeight}")
    
    # Branches to read
    Electron_branch  = TreeReader.UseBranch("Electron")
    Muon_branch      = TreeReader.UseBranch("Muon")
    FatJet_branch    = TreeReader.UseBranch("FatJet")
    Jet_branch       = TreeReader.UseBranch("Jet")
    MissingET_branch = TreeReader.UseBranch("MissingET")
    ScalarHT_branch  = TreeReader.UseBranch("ScalarHT")
    Weight_branch    = TreeReader.UseBranch("Weight")

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
    from loop_utilis import run_loop_cli
    load_delphes()
    run_loop_cli(loop_tree)