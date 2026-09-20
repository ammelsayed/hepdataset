#!/usr/bin/env python3
"""
Build ML-ready tabular datasets from Delphes ROOT files.
Author : A.M.M. Elsayed (University of Science and Technology of China)
Email  : ammelsayed@mail.ustc.edu.cn / ahmedphysica@outlook.com
"""

import os
import ROOT
import time
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from mt2 import mt2
from datetime import datetime
from samples_reader   import SamplesReader
from object_selection import ObjectSelector
from event_selection  import EventSelector
from branches_reader  import BranchesHandler
from kinematics       import EventShapes, Centrality, MtW
from parallelization  import parallel_runs, format_time
DELPHES_PATH = os.environ.get("DELPHES_HOME", "/home/ammelsayed/softwares/MG5_aMC_v3_5_15/Delphes")
ROOT.gInterpreter.AddIncludePath(DELPHES_PATH)
ROOT.gInterpreter.AddIncludePath(f"{DELPHES_PATH}/classes")
ROOT.gInterpreter.AddIncludePath(f"{DELPHES_PATH}/external")
ROOT.gSystem.Load("libDelphes")
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
ROOT.gInterpreter.Declare('#include "classes/SortableObject.h"')
ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("ATLAS")
ROOT.DisableImplicitMT() 
print("Using ROOT version:", ROOT.__version__)
print("Using Delphes libraries found at:", DELPHES_PATH)

def build_chain(inputRootFile):
    chain = ROOT.TChain("Delphes")
    if isinstance(inputRootFile, list):
        for file in inputRootFile:
            chain.Add(file)
    elif isinstance(inputRootFile, str):
        chain.Add(inputRootFile)
    else:
        raise TypeError(f"inputRootFile must be str or list, got {type(inputRootFile).__name__}")
    return chain

def count_entries(inputRootFile):
    return build_chain(inputRootFile).GetEntries()

def loop_tree(
    inputRootFile , 
    treeName = "Delphes", 
    eventWeight = None, 
    cross_section = None, 
    luminosity = None,
    start_entry = 0, 
    end_entry = None, 
    show_progress = False,
    debug_loop = False,
    output_dir = "HepDataset", 
    output_file_name = "events.root", 
    overwrite_output_dir = False,
    branches_config_path = None,
    ):

    # Read the input file
    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)

    # Check start and end entries
    numberOfEntries = TreeReader.GetEntries()
    if start_entry < 0 or start_entry >= numberOfEntries:
        raise ValueError(f"start_entry must be between 0 and {numberOfEntries}")
    if end_entry is None or end_entry > numberOfEntries:
        end_entry = numberOfEntries
    if end_entry < start_entry:
        raise ValueError("end_entry must be greater than or equal to start_entry")

    # Actual number of entries going to processed
    numberOfProcessedEntries = end_entry - start_entry

    # Check event weights
    if eventWeight is None:
        if (cross_section is None) != (luminosity is None):
            raise ValueError("cross section and luminosity must be provided together")
        if cross_section is not None and luminosity is not None:
            if numberOfProcessedEntries == 0:
                raise ValueError("Cannot compute an event weight when no entries will be processed")
            eventWeight = cross_section * luminosity / numberOfProcessedEntries
        else:
            eventWeight = 1.0

    # Validate / create output_dir 
    if output_dir is not None:
        if os.path.exists(output_dir):
            if not os.path.isdir(output_dir):
                raise ValueError(f"output_dir is not a directory: {output_dir}")
            if not overwrite_output_dir:
                raise FileExistsError(
                    f"Output directory already exists, cannot write there: {output_dir}"
                )
        else:
            os.makedirs(output_dir)
            if show_progress:
                print(f"Created output directory : {output_dir}")

    if show_progress:
        print(f"Reading ROOT file: {inputRootFile}")
        print(f"Total number of events: {numberOfEntries:,}")
        print(f"Processing events: {start_entry:,} to {end_entry - 1:,} ({numberOfProcessedEntries:,} events)")
        print(f"Weight per-event: {eventWeight:.3f}")
    
    # Branches to read
    Electron_branch  = TreeReader.UseBranch("Electron")
    Muon_branch      = TreeReader.UseBranch("Muon")
    FatJet_branch    = TreeReader.UseBranch("FatJet")
    Jet_branch       = TreeReader.UseBranch("Jet")
    MissingET_branch = TreeReader.UseBranch("MissingET")
    ScalarHT_branch  = TreeReader.UseBranch("ScalarHT")
    Weight_branch    = TreeReader.UseBranch("Weight")

    # get the branch names
    BR = BranchesHandler(branches_config_path)
    if not BR.is_valid():
        BR.print_validation()
        raise ValueError(f"Invalid branches configuration: {branches_config_path}")
    float_branch_names = BR.get_float_branch_names()
    int_branch_names = BR.get_int_branch_names()

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
        for branch_name in int_branch_names:
            b[branch_name] = np.zeros(1, dtype=np.int32)
            tree.Branch(branch_name, b[branch_name], f"{branch_name}/I")  # /I for Integer
        for branch_name in float_branch_names:
            b[branch_name] = np.zeros(1, dtype=np.float64)
            tree.Branch(branch_name, b[branch_name], f"{branch_name}/D")  # /D for Double (64-bit float)
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
        
        # Reset branches to np.nan (weight defaults to 0.0 for skipped events)
        b = buffers[ac_key]
        for branch_name in int_branch_names:
            b[branch_name][0] = -1
        for branch_name in float_branch_names:
            b[branch_name][0] = np.nan
        b["weight"][0] = 0.0
        b["gen_weight"][0] = 0.0

        # ================================================================
        # Fill the branches
        # ================================================================

        # A dictonary to hold the selected objects' four-momenta for easier access
        P4s = {}

        # Count number of objects before quality selections AND store them locally
        nPreQS_Muon = Muon_branch.GetEntries()
        nPreQS_Electron = Electron_branch.GetEntries()
        nPreQS_Lepton = nPreQS_Muon + nPreQS_Electron
        nPreQS_FatJet = FatJet_branch.GetEntries()
        nPreQS_Jet = Jet_branch.GetEntries()

        b["nPreQS_Muon"][0] = nPreQS_Muon
        b["nPreQS_Electron"][0] = nPreQS_Electron
        b["nPreQS_Lepton"][0] = nPreQS_Lepton
        b["nPreQS_FatJet"][0] = nPreQS_FatJet
        b["nPreQS_Jet"][0] = nPreQS_Jet

        if debug_loop:
            print(f"Number of objects before quality selections:")
            print(f" mu = {nPreQS_Muon}, e = {nPreQS_Electron}, J = {nPreQS_FatJet}, j = {nPreQS_Jet}")

       
        # ----------------------------------------------------------------
        # Fill leptons kinematics
        # ---------------------------------------------------------------

        muonMass = 0.1057
        electronMass = 0.511E-3

        for lep_idx, lepton in enumerate(goodLeptons[:BR.get_obj_count("Lepton")]):

            mainClassNames = ["Lepton"] # branches including these class names will be filled

            lepClassName = lepton.ClassName()
            if lepClassName in BR.get_obj_repr("Lepton"):
                mainClassNames.append(lepClassName)

            for lepType in mainClassNames:

                inst = f"{lepType}{lep_idx}"

                # Correct P4 object with mass
                if debug_loop:
                    print(inst)
                    print(f"> Before p4 correction: pt = {lepton.P4().Pt()}, eta = {lepton.P4().Eta()}, phi = {lepton.P4().Phi()}, m = {lepton.P4().M()}.")

                if lepClassName == 'Muon':
                    p4 = ROOT.TLorentzVector()
                    p4.SetPtEtaPhiM(lepton.PT, lepton.Eta, lepton.Phi, muonMass)
                    P4s[inst] = p4
                    if debug_loop:
                        print(f"> After p4 correction: pt = {p4.Pt()}, eta = {p4.Eta()}, phi = {p4.Phi()}, m = {p4.M()}.")

                elif lepClassName == 'Electron':
                    p4 = ROOT.TLorentzVector()
                    p4.SetPtEtaPhiM(lepton.PT, lepton.Eta, lepton.Phi, electronMass)
                    P4s[inst] = p4
                    if debug_loop:
                        print(f"> After p4 correction: pt = {p4.Pt()}, eta = {p4.Eta()}, phi = {p4.Phi()}, m = {p4.M()}.")
                else:
                    continue 

                # Fill kinematics asked for
                for k in BR.get_obj_kinematics("Lepton"):

                    branch_name = f"{k}_{inst}"

                    try:
                        b[branch_name][0] = getattr(lepton, k) # example: lep.PT 
                    except AttributeError:
                        try:
                            b[branch_name][0] = getattr(p4, k)() # example: lep.P4().Px() 
                        except AttributeError as e:
                            # leave as NaN (already set at reset time)
                            if debug_loop:
                                print(f"> Skipping {branch_name}: not available on {lepClassName} or its TLorentzVector.")
                            continue

        # ----------------------------------------------------------------
        # Fill Large-R (FatJets) kinematics
        # ---------------------------------------------------------------

        for fj_idx, fatjet in enumerate(goodFatJets[:BR.get_obj_count("FatJet")]):
            
            classname = fatjet.ClassName()

            p4_fj = fatjet.P4()
            P4s[f"FatJet{fj_idx}"] = p4_fj

            # Log softdropped large-R jet four-momenta
            # Required only if softdropped fatjets are configured as a representaion of large-R jets.
            if hasattr(fatjet, 'SoftDroppedP4') and ("SoftDroppedFatJet" in BR.get_obj_repr("FatJet")):
                p4_sd = fatjet.SoftDroppedP4[0]
                P4s[f"SoftDroppedFatJet{fj_idx}"] = p4_sd

            # Log trimmed large-R jet four-momenta
            if hasattr(fatjet, 'TrimmedP4') and ("TrimmedFatJet" in BR.get_obj_repr("FatJet")):
                p4_tr = fatjet.TrimmedP4[0]            
                P4s[f"TrimmedFatJet{fj_idx}"] = p4_tr

            # Log pruned large-R jet four-momenta
            if hasattr(fatjet, 'PrunedP4') and ("PrunedFatJet" in BR.get_obj_repr("FatJet")):
                p4_pr = fatjet.PrunedP4[0]
                P4s[f"PrunedFatJet{fj_idx}"] = p4_pr

            # Fill kinematics asked for
            for k in BR.get_obj_kinematics("FatJet"):

                # n-subjetness variables are handeled seperatly
                if k in ["Tau1", "Tau2", "Tau3", "Tau21", "Tau32"]: 
                    continue
                
                # Fill basic kinematics of the fatjet
                branch_name = f"{k}_FatJet{fj_idx}"

                try:
                    b[branch_name][0] = getattr(fatjet, k)  
                except AttributeError:
                    try:
                        b[branch_name][0] = getattr(p4_fj, k)()  
                    except AttributeError as e:
                        if debug_loop:
                            print(f"> Skipping {branch_name}: not available on {classname} or its TLorentzVector.")
                        continue

                # Fill same kinematics for the softdropped jet
                # grooming variants do not have direct methods
                # so we just use one layer of reading 
                if hasattr(fatjet, 'SoftDroppedP4') and ("SoftDroppedFatJet" in BR.get_obj_repr("FatJet")):

                    branch_name = f"{k}_SoftDroppedFatJet{fj_idx}"

                    try:
                        b[branch_name][0] = getattr(p4_sd, k)()  
                    except AttributeError as e:
                        if debug_loop:
                            print(f"> Skipping {branch_name}: not available on {classname} or its TLorentzVector.")
                        continue

                # Fill same kinematics for the trimmed jet
                if hasattr(fatjet, 'TrimmedP4') and ("TrimmedFatJet" in BR.get_obj_repr("FatJet")):

                    branch_name = f"{k}_TrimmedFatJet{fj_idx}"

                    try:
                        b[branch_name][0] = getattr(p4_tr, k)()  
                    except AttributeError as e:
                        if debug_loop:
                            print(f"> Skipping {branch_name}: not available on {classname} or its TLorentzVector.")
                        continue

                # Fill same kinematics for the pruned jet
                if hasattr(fatjet, 'PrunedP4') and ("PrunedFatJet" in BR.get_obj_repr("FatJet")):

                    branch_name = f"{k}_PrunedFatJet{fj_idx}"

                    try:
                        b[branch_name][0] = getattr(p4_pr, k)()  
                    except AttributeError as e:
                        if debug_loop:
                            print(f"> Skipping {branch_name}: not available on {classname} or its TLorentzVector.")
                        continue

            # Fill n-subjetness substructure variables
            # Those are only filled for FatJet
            # The groomed variants do not carry these methods
            for t_idx in [1, 2, 3]:
                b[f"Tau{t_idx}_FatJet{fj_idx}"][0] = fatjet.Tau[t_idx - 1]

                if debug_loop:
                    print(f"> Filled branch Tau{t_idx}_FatJet{fj_idx}: value = {fatjet.Tau[t_idx - 1]}.")

            t1, t2, t3 = fatjet.Tau[0], fatjet.Tau[1], fatjet.Tau[2]
            b[f"Tau21_FatJet{fj_idx}"][0] = t2 / t1 if t1 > 0 else 1.0
            b[f"Tau32_FatJet{fj_idx}"][0] = t3 / t2 if t2 > 0 else 1.0

        # ----------------------------------------------------------------
        # Fill Small-R Jets kinematics
        # ----------------------------------------------------------------

        for goodJetsList, prefix in zip([goodJets, goodBJets, goodTauJets], ["Jet", "BJet", "TauJet"]):

            if prefix not in BR.get_obj_repr("Jet"):
                continue

            for jet_idx, jet in enumerate(goodJetsList[:BR.get_obj_count("Jet")]):
                classname = jet.ClassName()
                inst = f"{prefix}{jet_idx}"
                p4 = jet.P4()
                P4s[inst] = p4

                for k in BR.get_obj_kinematics("Jet"):

                    branch_name = f"{k}_{inst}"

                    try:
                        b[branch_name][0] = getattr(jet, k)  
                    except AttributeError:
                        try:
                            b[branch_name][0] = getattr(p4, k)()  
                        except AttributeError as e:
                            if debug_loop:
                                print(f"> Skipping {branch_name}: not available on {classname} or its TLorentzVector.")
                            continue

        # ----------------------------------------------------------------
        # Fill scalars and event shapes
        # ----------------------------------------------------------------

        # Fill the met kinematics
        met = MissingET_branch.At(0)
        p4_met = met.P4()
        P4s["MET"] = p4_met
        for k in BR.get_obj_kinematics("MET"):
            branch_name = f"{k}_MET"
            try:
                b[branch_name][0] = getattr(met, k)  
            except AttributeError:
                try:
                    b[branch_name][0] = getattr(p4_met, k)()  
                except AttributeError as e:
                    if debug_loop:
                        print(f"> Skipping {branch_name}: not available on {met.ClassName()} or its TLorentzVector.")
                    continue

        # Fill other global scalars        
        ht = ScalarHT_branch.At(0).HT
        b["HT"][0] = ht
        
        lt = sum([lep.PT for lep in goodLeptons])
        st = lt + ht
        meff = st + met.MET
        sig_met = ( met.MET / np.sqrt(ht) ) if ht > 0 else np.nan
        b["LT"][0] = lt
        b["ST"][0] = st
        b["Significance_MET"][0] = sig_met
        b["Meff"][0] = meff

        sumPT_jets = sum([jet.PT for jet in goodJets])
        sumPT_fjs = sum([fj.PT for fj in goodFatJets])
        sumPT_sdfjs = sum([fj.SoftDroppedP4[0].Pt() for fj in goodFatJets])

        b["ScalarSumPT_Jets"][0] = sumPT_jets
        b["ScalarSumPT_FatJets"][0] = sumPT_fjs
        b["ScalarSumPT_SoftDroppedFatJets"][0] = sumPT_sdfjs
        b["ScalarSumPT_Hadronic"][0] = sumPT_jets + sumPT_sdfjs # should equal HT
        
        # Fill event shapes
        vis_p4s = [lep.P4() for lep in goodLeptons] + [fj.P4() for fj in goodFatJets] + [jet.P4() for jet in goodJets] + [met.P4()]
        px_arr = np.array([p.Px() for p in vis_p4s], dtype=np.float64)
        py_arr = np.array([p.Py() for p in vis_p4s], dtype=np.float64)
        pz_arr = np.array([p.Pz() for p in vis_p4s], dtype=np.float64)
        e_arr  = np.array([p.Energy() for p in vis_p4s], dtype=np.float64)
    
        S, A, C = EventShapes(px_arr, py_arr, pz_arr)
        b["Sphericity"][0] = S
        b["Aplanarity"][0] = A
        b["Circularity"][0] = C
        b["Centrality"][0] = Centrality(px_arr, py_arr, pz_arr, e_arr)

        # ----------------------------------------------------------------
        # Fill N-body relations
        # ----------------------------------------------------------------

        if BR.multiObjects_Nmax >= 2:

            for N in range(2, BR.multiObjects_Nmax + 1, 1):

                for p_names in BR.get_nbody_combinations(N):

                    p4_list = [P4s.get(n) for n in p_names]

                    if all(p4_list):

                        sufx = '_'.join(p_names)

                        # Without start, sum() defaults to  0 (integer), 
                        # which would fail because you can't add 0 + TLorentzVector()
                        total_p4 = sum(p4_list[1:], p4_list[0])

                        # fill basic kinematics first 
                        for k in BR.get_nbody_kinematics(N):
                            if k not in BR.multiObject_2body_kinematics:
                                branch_name = f"{k}_{sufx}"
                                try:
                                    b[branch_name][0] = getattr(total_p4, k)()  
                                except AttributeError as e:
                                    if debug_loop:
                                        print(f"Skipping {k}: not available for {sufx}.")
                                    continue

                        # by default 2body kinematics should be included for any 2-body objects
                        # those are just ["DeltaR", "DeltaPhi", "DeltaEta", "MtW"]
                        # we fill them manullay
                        if N == 2:
                            p1 = p4_list[0]; p2 = p4_list[1]
                            if "DeltaR" in BR.multiObject_2body_kinematics:
                                b[f"DeltaR_{sufx}"][0] = p1.DeltaR(p2)
                            if "DeltaPhi" in BR.multiObject_2body_kinematics:
                                b[f"DeltaPhi_{sufx}"][0] = p1.DeltaPhi(p2)
                            if "DeltaEta" in BR.multiObject_2body_kinematics:
                                b[f"DeltaEta_{sufx}"][0] = p1.Eta() - p2.Eta()
                            if "MtW" in BR.multiObject_2body_kinematics:
                                b[f"MtW_{sufx}"][0] = MtW(p1.Pt(), p1.Phi(), p2.Pt(), p2.Phi())
                        

                        # Here we wish to calculate the stransverse mass
                        # We just calclate it for 2 body objects, 
                        # and only for lepton and fatjet pars
                        # this is because calculating mt2 can be slow
                        if BR.multiObjects_include_mt2 and N == 2:
                            supported = BR.get_obj_instances("Lepton") + BR.get_obj_instances("FatJet")
                            p1 = p4_list[0]; p2 = p4_list[1]
                            p1_name = p_names[0]; p2_name = p_names[1]
                            if (p1_name in supported) and (p2_name in supported):
                                b[f"MT2_{sufx}"][0] = mt2(
                                    p1.M(), p1.Px(), p1.Py(),
                                    p2.M(), p2.Px(), p2.Py(),
                                    p4_met.Px(), p4_met.Py(),
                                    0.0, 0.0   # neutrino masses
                                )
                        
                        # only fill the event shapes if they are asked for
                        if BR.multiObjects_include_trival_kinematics:
                            px_arr = np.array([p.Px() for p in p4_list], dtype=np.float64)
                            py_arr = np.array([p.Py() for p in p4_list], dtype=np.float64)
                            pz_arr = np.array([p.Pz() for p in p4_list], dtype=np.float64)
                            e_arr  = np.array([p.Energy() for p in p4_list], dtype=np.float64)
                            S, A, C = EventShapes(px_arr, py_arr, pz_arr)
                            b[f"Sphericity_{sufx}"][0] = S
                            b[f"Aplanarity_{sufx}"][0] = A
                            b[f"Circularity_{sufx}"][0] = C
                            b[f"Centrality_{sufx}"][0] = Centrality(px_arr, py_arr, pz_arr, e_arr)
                            b[f"ScalarSumPT_{sufx}"][0] = sum([p.Pt() for p in p4_list])
                            b[f"VectorSumPT_{sufx}"][0] = total_p4.Pt()


        # fill weight branches 
        w_gen = Weight_branch.At(0).Weight
        b["gen_weight"][0] = w_gen
        b["weight"][0] = eventWeight

        # Fill the trees
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

def check_process_pool_executor_results(results):
    for r in results:
        if isinstance(r, Exception):
            raise r

def hadd_files(out_RootFile, in_RootFilesList, max_workers=None):
    j = str(max_workers or os.cpu_count())
    cmd = ["hadd", "-f", "-j", j, out_RootFile] + list(in_RootFilesList)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"hadd FAILED (rc={result.returncode}):\n{result.stderr}")
        return False
    return True

def hadd_chunks(results, max_workers, output_file_name):
    print("Merging chunks with hadd ..")
    start = time.perf_counter()

    # learn structure from first worker
    ac_keys = list(results[0].keys()) 

    # Group chunk files by ac_key
    ac_files = {ac_key: [] for ac_key in ac_keys}
    for chunk_paths in results:
        for ac_key, path in chunk_paths.items():
            if ac_key in ac_files and path is not None:
                ac_files[ac_key].append(path)

    merged = {}
    for ac_key, files in tqdm(ac_files.items()):
        if not files:
            continue

        # Chunk files already live in the destination directory, e.g.
        # <output_dir>/<ac>/<ac_r>/<treeName>_split_<i>_events_<s>_<e>.root.
        out_path = os.path.join(os.path.dirname(files[0]), output_file_name)
        ok = hadd_files(out_path, files, max_workers=max_workers)
        if ok:
            merged[ac_key] = out_path
            # hadd never deletes its inputs; clean them up ourselves.
            for f in files:
                try:
                    os.remove(f)
                except OSError as exc:
                    print(f"  WARNING: could not remove chunk {f}: {exc}")
        else:
            print(f"  WARNING: hadd failed for {ac_key}")

    print(f"Merged {len(results)} chunks x {len(merged)} SR(s) in {format_time(time.perf_counter() - start)}")
    return merged

def split_range(total, n):
    n = max(1, min(n, total))
    base, rem = divmod(total, n)
    out, start = [], 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        out.append((start, start + size))
        start += size
    return out


def loop_tree_parallel(
    inputRootFile , 
    treeName = "Delphes", 
    eventWeight = None, 
    cross_section = None, 
    luminosity = None,
    start_entry = 0, 
    end_entry = None, 
    show_progress = False, 
    debug_loop = False,
    output_dir = "HepDataset", 
    output_file_name = "events.root", 
    overwrite_output_dir = False,
    branches_config_path = None,
    n_chunks = None, 
    max_workers = None
    ):

    total = count_entries(inputRootFile)
    n_chunks = n_chunks or (max_workers or os.cpu_count())
    
    split_args = []
    for i, (s, e) in enumerate(split_range(total, n_chunks)):
        split_args.append((
            inputRootFile, 
            treeName, 
            eventWeight, 
            None,  # cross_section
            None,  # luminosity
            s,     # start_entry
            e,     # end_entry
            False, # show_progress
            False, # debug loop
            output_dir,  # output_dir
            f"tmp_split{i}_{output_file_name}", 
            True, # overwrite_output_dir
            branches_config_path,
        ))
    
    results = parallel_runs(
        loop_tree, 
        split_args, 
        max_workers = max_workers, 
        info="", 
        mpContext="fork"
    )
    check_process_pool_executor_results(results)

    merged_trees  = None
    merged_paths = hadd_chunks([r["root_paths"] for r in results], max_workers = max_workers, output_file_name = output_file_name)
    merged_objSel = ObjectSelector.Merge([r["ObjectSelector"] for r in results])
    merged_evtSel = EventSelector.Merge([r["EventSelector"] for r in results])
        
    return {
        "trees": merged_trees,
        "root_paths": merged_paths,
        "ObjectSelector": merged_objSel,
        "EventSelector": merged_evtSel,
    }

def resolve_category_name(category):
    cat = str(category).lower()
    return "bkg" if cat == "background" else "sig" if cat == "signal" else "Unknown"

def make_dataset(
    samples_file,
    branches_config_file = "/data/ammelsayed/hepdataset/src/defaults/branches_config.yml", 
    run_in_parallel = False,
    output_dir = "HEPDataset", 
    working_luminosity = 400.0,
    max_workers = None, 
    n_chunks = None,
    merge_proc_samples = True,
    ):

    for category, processes in SamplesReader(str(samples_file)).read().items():
        
        prefix = resolve_category_name(category)

        for proc_name, proc_meta in processes.items():
            proc_RootFiles      = proc_meta["files"]
            proc_NbRootFiles    = len(proc_RootFiles)
            proc_totalNbEvents  = proc_meta["nb_events"]
            proc_CrossSection   = proc_meta["cross_section"] * 1000

            if not proc_RootFiles or not proc_totalNbEvents:
                continue

            proc_eventWeight = proc_CrossSection * working_luminosity / proc_totalNbEvents
            proc_treeName    = f"{prefix}_{proc_name}"

            print("-" * 80)
            print(f"Processing : {proc_name} ({prefix})")
            print(f" Cross section                  : {proc_CrossSection:.3f} fb")
            print(f" Total number of .root files:   : {proc_NbRootFiles}")
            print(f" Total number of events         : {proc_totalNbEvents:,}")
            print(f" Average weight per event       : {proc_eventWeight:.3f} (at {working_luminosity} fb^-1)")
            print("-" * 80)

            proc_paths = {}
            proc_objSel, proc_evtSel = [], []
            for i, sampleRootFile in enumerate(proc_RootFiles):
                if i > 0: print("-"*80)
                print(f"Sample {i+1}/{proc_NbRootFiles}")   
                print("-"*80)

                sample_treeName = f"{proc_treeName}_sample{i}"
                sample_fileName = f"{proc_treeName}_sample{i}.root"
            

                if run_in_parallel:
                    result = loop_tree_parallel(
                        inputRootFile = str(sampleRootFile), 
                        treeName = proc_treeName,  # <-- same tree name for every sample
                        eventWeight = proc_eventWeight, 
                        cross_section = None, 
                        luminosity = None,
                        start_entry = 0, 
                        end_entry = None, 
                        show_progress = False, 
                        output_dir = str(output_dir), 
                        output_file_name = sample_fileName, 
                        overwrite_output_dir = True,
                        branches_config_path = branches_config_file,
                        n_chunks = n_chunks, 
                        max_workers = max_workers
                    )

                # Serial mode
                else:
                    result =  loop_tree(
                        inputRootFile = str(sampleRootFile), 
                        treeName = proc_treeName,  # <-- same tree name for every sample
                        eventWeight = proc_eventWeight, 
                        cross_section = None, 
                        luminosity = None,
                        start_entry = 0, 
                        end_entry = None, 
                        show_progress = True, 
                        output_dir = str(output_dir), 
                        output_file_name = sample_fileName, 
                        overwrite_output_dir = True,
                        branches_config_path = branches_config_file,
                    )


                proc_objSel.append(result["ObjectSelector"])
                proc_evtSel.append(result["EventSelector"])
                for key, path in result["root_paths"].items():
                    proc_paths.setdefault(key, []).append(path)

            # Merge per-sample files with hadd 
            if merge_proc_samples:
                print("Merging sample .root files with hadd ..")
                start = time.perf_counter()
                for key, paths in proc_paths.items():
                    if key is None:
                        ac, ac_r = "", ""
                    else:
                        ac, _, ac_r = key.rpartition("_")
                    target = os.path.join(output_dir , ac , ac_r , f"{proc_treeName}.root")

                    ok = hadd_files(str(target), paths, max_workers=max_workers)
                    if ok:
                        for p in paths:
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                print(f"Finished merged in {format_time(time.perf_counter() - start)}")

            # Merge the selectors across samples and print 
            if proc_objSel and proc_evtSel:
                merged_objSel = ObjectSelector.Merge(proc_objSel)
                merged_evtSel = EventSelector.Merge(proc_evtSel)
                merged_objSel.PrintObjectSelectionSummary(lum=working_luminosity, event_weight=proc_eventWeight)
                merged_evtSel.PrintEventSelectionSummary(proc_treeName, event_weight=proc_eventWeight, lum=working_luminosity)
                obj_dir = os.path.join(output_dir , "ObjectSelection" , proc_treeName)
                os.makedirs(os.path.dirname(obj_dir), exist_ok=True)
                merged_objSel.DrawObjectSelectionHistograms(output_dir=str(obj_dir))
            
            print(f"Finished working on {proc_name}.\n")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("samples_file", help="Samples YAML file.")
    # p.add_argument("branches_config_file", help="Branch configuration (YAML) file describing which objects/variables to extract.")
    # p.add_argument("--output-dir", default = "HEPDataset", help="Output directory.")
    # p.add_argument("--working-luminosity", type=float, default=400.0)
    # p.add_argument("--max-workers", type=int, default=None)
    # p.add_argument("--n-chunks", type=int, default=None)
    # p.add_argument("--show-progress", action="store_true")
    # p.add_argument("--merge-proc-samples", action="store_true")
    args = p.parse_args()

    started = datetime.now()
    print(f"=== Started at {started.isoformat(timespec='seconds')} ===")
    kwargs = vars(args).copy()

    try:
        make_dataset(**kwargs)
    except BaseException:
        finished = datetime.now()
        print(f"=== CRASHED at {finished.isoformat(timespec='seconds')} (after {finished - started}) ===")
        raise
    else:
        finished = datetime.now()
        print(f"=== Finished at {finished.isoformat(timespec='seconds')} (total {finished - started}) ===")

if __name__ == "__main__":
    main()