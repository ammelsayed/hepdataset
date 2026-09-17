#!/usr/bin/env python3
import os
import ROOT
import numpy as np
from tqdm import tqdm
from delphes_utilis import build_chain
from loop_utilis import check_loop_args
from object_selection import ObjectSelector
from event_selection import EventSelector
from itertools import combinations

def loop_tree(**loop_args):
    inputRootFile = loop_args["inputRootFile"]
    collect_summary = loop_args.pop("collect_summary", False)

    # Read the input file
    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)

    # Fill defaults, validate entries, compute weight, prepare output dir, print summary
    loop_args = check_loop_args(loop_args, TreeReader.GetEntries())
    start_entry, end_entry = loop_args["start_entry"], loop_args["end_entry"]
    eventWeight = loop_args["eventWeight"]
    treeName = loop_args["treeName"]
    show_progress = loop_args["show_progress"]
    debug_loop = loop_args["debug_loop"]
    output_dir = loop_args["output_dir"]
    output_file_name = loop_args["output_file_name"]
    luminosity = loop_args["luminosity"]
    
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

    # Analysis channel bookkeeping
    eventSel = EventSelector()
    ac_keys, ac_dict = eventSel.ac_keys, eventSel.ac_dict

    # Initialize object selector
    objSel = ObjectSelector()

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

    # Event loop
    rng = range(start_entry, end_entry)
    for entry in (tqdm(rng) if show_progress else rng):
        
        TreeReader.ReadEntry(entry)

        # Object selection
        selected_objects = objSel.Select(Muon_branch, Electron_branch, FatJet_branch, Jet_branch, event_weight = eventWeight)
        goodFatJets = selected_objects["goodFatJets"]
        goodLeptons = selected_objects["goodLeptons"]
        goodJets = selected_objects["goodJets"]
        goodBJets = selected_objects["goodBJets"]
        goodTauJets = selected_objects["goodTauJets"]

        # Identify the analysis channel key
        ac_key = eventSel.Select(goodLeptons, goodFatJets, valid_keys = trees.keys())

        # Skip events that don't match any analysis channel
        if ac_key is None:
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

        trees[ac_key].Fill()

    # Print object selection cutflow & analysis channels yeilds
    if show_progress:
        objSel.PrintObjectSelectionSummary(lum=luminosity, event_weight=eventWeight)
        eventSel.PrintEventSelectionSummary(treeName, event_weight = eventWeight, lum = luminosity)
        
    # if output_dir is given, then write the trees into root files.
    root_paths = {}
    if output_dir is not None:
        
        # Write the trees into .root files at:
        # <output_dir>/<analysis_channel_name>/<analysis_region_name>/<output_root_file_name>.root
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
                root_paths[ac_key] = path
        
        # Write object-selection histograms + cutflow into a dedicated root file at:
        # <output_dir>/ObjectSelection/<treeName>.root
        out_objsel = os.path.join(output_dir, "ObjectSelection")
        os.makedirs(out_objsel, exist_ok=True)
        f_objsel = ROOT.TFile.Open(os.path.join(out_objsel, f"{treeName}.root"), "RECREATE")
        objSel.WriteObjectSelectionHistograms(f_objsel)
        objSel.WriteObjectSelectionSummary(f_objsel)
        f_objsel.Close()
        # Draw the object selection histograms (PNGs go in the same directory)
        if show_progress:
            objSel.DrawObjectSelectionHistograms(output_dir = out_objsel)

        # Write event-selection cutflow into a dedicated root file at:
        # <output_dir>/EventSelection/<treeName>.root
        out_evtsel = os.path.join(output_dir, "EventSelection")
        os.makedirs(out_evtsel, exist_ok=True)
        f_evtsel = ROOT.TFile.Open(os.path.join(out_evtsel, f"{treeName}.root"), "RECREATE")
        eventSel.WriteEventSelectionSummary(f_evtsel, treeName)
        f_evtsel.Close()
        
    return {
        "trees" : trees,
        "root_paths" : root_paths,
        "ObjectSelector": objSel,
        "EventSelector": eventSel
    }


def main():
    from delphes_utilis import load_delphes
    from loop_utilis import run_loop_cli
    load_delphes()
    return run_loop_cli(loop_tree)

if __name__ == "__main__":

    main()