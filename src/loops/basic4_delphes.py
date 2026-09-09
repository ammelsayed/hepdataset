#!/usr/bin/env python3
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

def loop_tree(
    inputRootFile,
    treeName,
    eventWeight = 1.0,
    start_entry = 0,
    end_entry = None,
    show_progress = True,
    temp_dir_path = None,
    nb_lep_max = 3,
    nb_fj_max = 2,
    prune_empty_branches = True,
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
    branch_names  = [f"{k}_Lepton{i}" for k in ["PT", "Eta", "Phi"] for i in range(nb_lep_max)]
    branch_names += [f"{k}_FatJet{i}" for k in ["PT", "Eta", "Phi", "Mass"] for i in range(nb_fj_max)]
    branch_names += [f"{k}_{'_'.join(comb)}" for k in ["DeltaR", "DeltaPhi", "DeltaEta", "M"] for comb in combinations([f"Lepton{i}" for i in range(nb_lep_max)] + [f"FatJet{i}" for i in range(nb_fj_max)], 2)]
    branch_names += ["MET", "HT", "LT", "ST", "Meff"]
    branch_names += ["weight", "gen_weight"]

    # book the trees and create the branch buffers
    trees, buffers = {}, {}
    ac_keys = ["1L", "2L", "3L"]
    filled_branches = {ac_key: set() for ac_key in ac_keys}
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

    # Event loop
    numberOfEntries = TreeReader.GetEntries()
    if end_entry is None or end_entry > numberOfEntries:
        end_entry = numberOfEntries

    rng = range(start_entry, end_entry)
    for entry in (tqdm(rng) if show_progress else rng):
        
        TreeReader.ReadEntry(entry)

        # Object selection
        selected_objects = select_objects(FatJet_branch, Electron_branch, Muon_branch)
        goodFatJets = selected_objects["goodFatJets"]
        goodLeptons = selected_objects["goodLeptons"]

        # Identify the analysis channel key
        nb_lep, nb_fj = len(goodLeptons), len(goodFatJets)
        if nb_lep == 1 and nb_fj >= 1: ac_key = "1L"
        elif nb_lep == 2 and nb_fj >= 1: ac_key = "2L"
        elif nb_lep == 3 and nb_fj >= 1: ac_key = "3L"
        else: ac_key = None

        # Skip events that don't match any analysis channel
        if ac_key is None or ac_key not in trees.keys():
            continue  
        
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

        # Record which branches got real (non-NaN) values
        for name, arr in b.items():
            if not np.isnan(arr[0]):
                filled_branches[ac_key].add(name)

        # Fill the data into the tree
        trees[ac_key].Fill()

    if temp_dir_path is not None:
        paths = {}
        for ac_key, tree in trees.items():
            # Disable unfilled branches
            if prune_empty_branches:
                for branch in tree.GetListOfBranches():
                    name = branch.GetName()
                    if name not in filled_branches[ac_key]:
                        tree.SetBranchStatus(name, 0)

            # Write into file
            os.makedirs(temp_dir_path, exist_ok=True)
            path = os.path.join(temp_dir_path, f"{treeName}_{ac_key}.root")
            f_out = ROOT.TFile.Open(path, "RECREATE")
            tree.SetDirectory(f_out)
            tree.Write()
            tree.Delete()
            f_out.Close()
            paths[ac_key] = path
        return paths

    else:
        # Disable unfilled branches
        if prune_empty_branches:
            for ac_key, tree in trees.items():
                for branch in tree.GetListOfBranches():
                    name = branch.GetName()
                    if name not in filled_branches[ac_key]:
                        tree.SetBranchStatus(name, 0)
        return trees

if __name__ == "__main__":
    
    load_delphes()  
    
    parser = argparse.ArgumentParser(description="Process a Delphes ROOT file and write a flat tree with selected events.")
    parser.add_argument("input_root_file",  type=str, help="Path to the input Delphes ROOT file.")
    parser.add_argument("--tree-name", type=str, default="Delphes", help="Name of the output TTree (default: Delphes).")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory where the output ROOT file will be written (default: current directory).")
    parser.add_argument("--event-weight", type=float, default=1.0, help="Weight to apply to each event (default: 1.0).")
    parser.add_argument("--nb-lep-max", type=int, default=3, help="Maximum number of leptons to store (default: 3).")
    parser.add_argument("--nb-fj-max", type=int, default=2, help="Maximum number of fat jets to store (default: 2).")
    parser.add_argument("--prune-empty-branches", action="store_true", help="Disable branches that have no filled values (default: True).")
    parser.add_argument("--start-entry", type=int, default=0, help="Entry to start processing from (default: 0).")
    parser.add_argument("--end-entry", type=int, default=None, help="Entry to stop processing at (default: None, meaning process all entries).")
    parser.add_argument("--show-progress", action="store_true", help="Show a progress bar during processing.")


    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run the loop
    out_path = loop_tree(
        inputRootFile=args.input_root_file,
        treeName=args.tree_name,
        eventWeight=args.event_weight,
        start_entry=args.start_entry,
        end_entry=args.end_entry,
        show_progress=args.show_progress,
        temp_dir_path=args.output_dir,
        nb_lep_max=args.nb_lep_max,
        nb_fj_max=args.nb_fj_max,
    )

    print(f"Output written to: {out_path}")