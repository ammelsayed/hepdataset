#!/usr/bin/env python3

"""
Supports reading of one Delphes root file at a time.
Events are selected based on the number of leptons and fatjets, and written to separate flat trees for each analysis channel.
Each tree is then written into a different root file.
More branches are supported compared to basic1_delphes.py, including pairwise kinematic variables and global event variables.
"""

import os
import ROOT
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from delphes import load_delphes, build_chain
from kinematics import DeltaR, DeltaPhi, DeltaEta
from object_selection import select_objects
from object_selection import PrintObjectSelectionSummary
from object_selection import BookObjectSelectionHistograms, DrawObjectSelectionHistograms
from analysis_channels import get_analysis_channel_keys, classify_analysis_channel
from itertools import combinations
from tabulate import tabulate

def loop_tree(
    inputRootFile,
    treeName,
    eventWeight = None,
    cross_section = None,
    luminosity = None,
    start_entry = 0,
    end_entry = None,
    temp_dir_path = None,
    show_progress = True,
    debug_loop = False,
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

    # get the analysis channels keys and definitions
    ac_keys, ac_dict = get_analysis_channel_keys(splitByFlavour=False)

    # book the trees and create the branch buffers
    trees, buffers = {}, {}
    for ac_key in ac_keys:
        tree_name = f"{treeName}_{ac_key}"
        tree = ROOT.TTree(tree_name, tree_name)
        tree.SetDirectory(0)

        # Create a fresh buffer dictionary for this tree
        b = {}
        for name in branch_names:
            b[name] = np.zeros(1, dtype=np.float64)
            tree.Branch(name, b[name], f"{name}/D")
        buffers[ac_key] = b      

        trees[ac_key] = tree

    # Prepare count dict to count number of events going to each analysis channel:
    # Also prepare some dictonaries to loging object selection cutflow
    ac_counts = dict.fromkeys(["initial", *ac_keys, "dropped"], 0)

    objSel_h = BookObjectSelectionHistograms()
    objSel_cutflow = {
        "lepton" : {"initial" : 0},
        "fatjet" : {"initial" : 0}
    }

    # Event loop
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

    rng = range(start_entry, end_entry)
    for entry in (tqdm(rng) if show_progress else rng):
        
        TreeReader.ReadEntry(entry)

        # Object selection
        selected_objects = select_objects(Muon_branch, Electron_branch, FatJet_branch, Jet_branch, objSel_cutflow, objSel_h, event_weight = eventWeight)
        goodFatJets = selected_objects["goodFatJets"]
        goodLeptons = selected_objects["goodLeptons"]
        goodJets = selected_objects["goodJets"]
        goodBJets = selected_objects["goodBJets"]
        goodTauJets = selected_objects["goodTauJets"]

        # Identify the analysis channel key
        ac_key = classify_analysis_channel(goodLeptons, goodFatJets, splitByFlavour=False)

        # Skip events that don't match any analysis channel
        ac_counts["initial"] += 1
        if ac_key is None or ac_key not in trees.keys():
            ac_counts["dropped"] += 1
            continue  
        ac_counts[ac_key] += 1

        # Reset all branches of this tree to NaN
        b = buffers[ac_key]
        for name in b:
            b[name][0] = np.nan
        b["weight"][0] = 0.0

        # A dictonary to hold the selected objects' four-momenta for easier access
        P4s = {}

        # Fill lepton data
        if len(goodLeptons) > 0:
            for idx, lep in enumerate(goodLeptons[:nb_lep_max], start = 0):
                P4s[f"Lepton{idx}"] = lep.P4()
                b[f"PT_Lepton{idx}"][0]  = lep.PT
                b[f"Eta_Lepton{idx}"][0] = lep.Eta
                b[f"Phi_Lepton{idx}"][0] = lep.Phi
                
        
        # Fill fatjet data
        if len(goodFatJets) > 0:
            for idx, fj in enumerate(goodFatJets[:nb_fj_max], start = 0):
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

        trees[ac_key].Fill()

    # Print object selection cutflow
    if show_progress:
        PrintObjectSelectionSummary(objSel_cutflow, lum = luminosity, event_weight = eventWeight)

    # Print analysis channels yeilds
    if show_progress:
        total_events = ac_counts["initial"]
        df = pd.DataFrame(ac_counts.items(), columns=[" Analysis Channel/Region", "Events"])
        if (eventWeight != None) and (luminosity != None): 
            df[f"Yield ({int(luminosity)} fb^-1)"] = df["Events"] * eventWeight
            df[f"Cross Section (fb)"] = df[f"Yield ({int(luminosity)} fb^-1)"] / luminosity
        df["Fraction"] = df["Events"] / total_events
        df["Fraction"] = df["Fraction"].map(lambda x: f"{x*100:.2f}%")
        print(f"\n*** Analysis channels yields for {treeName} ***")
        print(tabulate(df, headers='keys', tablefmt="simple", showindex=False, colalign=("left",) * 4))
        
    if temp_dir_path is not None:
        paths = {}
        for ac_key, tree in trees.items():
            os.makedirs(temp_dir_path, exist_ok=True)
            path = os.path.join(temp_dir_path, f"{treeName}_{ac_key}.root")
            f_out = ROOT.TFile.Open(path, "RECREATE")
            tree.SetDirectory(f_out)
            tree.Write()
            # tree.Delete()
            f_out.Close()
            paths[ac_key] = path
        
        # Draw the object selection histograms
        out_path = os.path.join(temp_dir_path, "ObjectSelection", treeName)
        os.makedirs(out_path, exist_ok=True)
        DrawObjectSelectionHistograms(objSel_h, output_dir = out_path)
        
        return paths

    else:
        return trees

if __name__ == "__main__":
    
    from pprint import pprint
    load_delphes()  
    
    parser = argparse.ArgumentParser(description="Process a Delphes ROOT file and write a flat tree with selected events.")
    parser.add_argument("input_root_file",  type=str, help="Path to the input Delphes ROOT file.")
    parser.add_argument("--tree-name", type=str, default="Delphes", metavar="", help="Name of the output TTree (default: Delphes).")
    parser.add_argument("--output-dir", type=str, default=".", metavar="", help="Directory where the output ROOT file will be written (default: current directory).")
    parser.add_argument("--event-weight", type=float, default=None, metavar="", help="Weight to apply to each event. If omitted, calculate it from cross section and luminosity.")
    parser.add_argument("--cross-section", type=float, default=None, metavar="", help="Cross section for the process in fb. Used with --luminosity when --event-weight is omitted.")
    parser.add_argument("--luminosity", type=float, default=None, metavar="", help="Target integrated luminosity in fb^-1. Used with --cross_section when --event-weight is omitted.")
    parser.add_argument("--start-entry", type=int, default=0, metavar="", help="Entry to start processing from (default: 0).")
    parser.add_argument("--end-entry", type=int, default=None, metavar="", help="Entry to stop processing at (default: None, meaning process all entries).")
    parser.add_argument("--show-progress", action="store_true", help="Show a progress bar during processing.")


    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run the loop
    out_path = loop_tree(
        inputRootFile=args.input_root_file,
        treeName=args.tree_name,
        eventWeight=args.event_weight,
        cross_section=args.cross_section,
        luminosity=args.luminosity,
        start_entry=args.start_entry,
        end_entry=args.end_entry,
        show_progress=args.show_progress,
        temp_dir_path=args.output_dir,
    )

    print(f"\nOutput written to:")
    pprint(out_path)