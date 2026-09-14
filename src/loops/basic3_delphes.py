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
import numpy as np
import pandas as pd
from tqdm import tqdm
from delphes import load_delphes, build_chain
from kinematics import DeltaR, DeltaPhi, DeltaEta
from object_selection import select_objects
from object_selection import PrintObjectSelectionSummary, PrintAnalysisChannelYields
from object_selection import BookObjectSelectionHistograms, DrawObjectSelectionHistograms, serialize_object_selection_histograms
from analysis_channels import get_analysis_channel_keys, classify_analysis_channel
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
    luminosity = loop_args["luminosity"]
    collect_summary = loop_args.pop("collect_summary", False)

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
        tree = ROOT.TTree(treeName, treeName)
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

    # Print object selection cutflow & analysis channels yeilds
    if show_progress:
        PrintObjectSelectionSummary(objSel_cutflow, lum = luminosity, event_weight = eventWeight)
        PrintAnalysisChannelYields(ac_counts, treeName, event_weight = eventWeight, lum = luminosity)
        
    # if output_dir is given, then write the trees into root files.
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        for ac, ac_rgs in ac_dict.items():
            for ac_r in ac_rgs:
                ac_key = f"{ac}_{ac_r}"
                path = os.path.join(output_dir, ac, ac_r, output_file_name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                f_out = ROOT.TFile.Open(path, "RECREATE")
                trees[ac_key].SetDirectory(f_out)
                trees[ac_key].Write()
                # trees[ac_key].Delete()
                f_out.Close()
                paths[ac_key] = path
        
        # Draw the object selection histograms
        if show_progress:
            out_path = os.path.join(output_dir, "ObjectSelection", treeName)
            os.makedirs(out_path, exist_ok=True)
            DrawObjectSelectionHistograms(objSel_h, output_dir = out_path)
        
        result = paths
    else:
        result = trees

    if collect_summary:
        return result, {
            "objSel_cutflow": objSel_cutflow,
            "ac_counts": ac_counts,
            "objSel_h": serialize_object_selection_histograms(objSel_h),
        }
    return result

if __name__ == "__main__":
    from loop_utilis import run_loop_cli
    load_delphes()
    run_loop_cli(loop_tree)