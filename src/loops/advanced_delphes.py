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

    # get the branch names
    branch_names = get_branch_names()

    # get the analysis channels keys and definitions
    ac_keys = get_analysis_channel_keys()

    # book the trees and create the branch buffers
    trees, buffers = {}, {}
    ac_keys = ["1L", "2L", "3L"]
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











        # Reset branches to np.nan (weight defaults to 0.0 for skipped events)
        for branch_name in get_float_branch_names():
            b[branch_name][0] = np.nan
        for branch_name in get_int_branch_names():
            b[branch_name][0] = -1

        # Count number of objects before quality selections AND store them locally
        nPreQS_Muon = Muon_branch.GetEntries()
        nPreQS_Electron = Electron_branch.GetEntries()
        nPreQS_Lepton = nPreQS_Muon + nPreQS_Electron
        nPreQS_FatJet = FatJet_branch.GetEntries()
        nPreQS_Jet = Jet_branch.GetEntries()

        b["nMuons_PreQS"][0] = nPreQS_Muon
        b["nElectrons_PreQS"][0] = nPreQS_Electron
        b["nLeptons_PreQS"][0] = nPreQS_Lepton
        b["nFatJets_PreQS"][0] = nPreQS_FatJet
        b["nJets_PreQS"][0] = nPreQS_Jet

        if debug_loop:
            print(f"Number of objects before quality selections:")
            print(f" mu = {nPreQS_Muon}, e = {nPreQS_Electron}, J = {nPreQS_FatJet}, j = {nPreQS_Jet}")

        ##################################################################
        # Quality Selections
        ##################################################################

        # ----------------------------------------------------------------
        # Collect good leptons
        # ----------------------------------------------------------------
        goodLeptons, goodMuons, goodElectrons = [], [], []

        # Muons
        for i in range(Muon_branch.GetEntries()):
            muon = Muon_branch.At(i)
            if isGoodMuon(i, muon.PT, muon.Eta, muon.IsolationVar):
                goodLeptons.append(muon)
                goodMuons.append(muon)

        # Electrons
        for i in range(Electron_branch.GetEntries()):
            electron = Electron_branch.At(i)
            if isGoodElectron(i, electron.PT, electron.Eta, electron.IsolationVar):
                goodLeptons.append(electron)
                goodElectrons.append(electron)

        # Sort leptons by PT (descending)
        goodLeptons.sort(key=lambda lep: lep.PT, reverse=True)
        nL = len(goodLeptons)
        nM = len(goodMuons)
        nE = len(goodElectrons)
        b["nMuons"][0] = nM
        b["nElectrons"][0] = nE
        b["nLeptons"][0] = nL

        if debug_loop:
            print(f"After QS mu = {nM}, e = {nE}, L = {nL}")
            for i, lep in enumerate(goodLeptons):
                lt = lepton_type_fn(i, goodLeptons, isLooseSR) if lepton_type_fn else "?"
                print(f"  Lep {i}: pt={lep.PT:.1f} eta={lep.Eta:.2f} type={lt}")

        # ----------------------------------------------------------------
        # Collect good fat jets
        # ----------------------------------------------------------------
        goodFatJets = []
        for i in range(FatJet_branch.GetEntries()):
            fj = FatJet_branch.At(i)
            if fj.PT >= 200 and abs(fj.Eta) <= 2.0:
                goodFatJets.append(fj)
        goodFatJets.sort(key=lambda fj: fj.PT, reverse=True)
        nFJ = len(goodFatJets)
        b["nFatJets"][0] = nFJ

        # ----------------------------------------------------------------
        # Collect good (AK4) jets
        # ----------------------------------------------------------------
        goodJetsList = []
        goodJets = goodJetsList
        for i in range(Jet_branch.GetEntries()):
            jet = Jet_branch.At(i)
            if jet.PT >= 30 and abs(jet.Eta) <= 2.5:
                goodJetsList.append(jet)
        goodJetsList.sort(key=lambda j: j.PT, reverse=True)
        nJ = len(goodJetsList)
        b["nJets"][0] = nJ

        if debug_loop:
            print(f"After QS FJ={nFJ}, J={nJ}")

        # ----------------------------------------------------------------
        # Classify into signal region and bail if none
        # ----------------------------------------------------------------
        sr_key = classify_signal_region_key(goodLeptons, goodFatJets, isLooseSR)
        if sr_key is None or sr_key not in signal_regions_keys:
            continue

        # ----------------------------------------------------------------
        # Fill individual object kinematics  (Lepton / FatJet / Jet)
        # ----------------------------------------------------------------
        P4s = {}

        # --- Leptons ---
        for lep_idx, lep in enumerate(goodLeptons[: get_obj_count("Lepton")]):
            classname = lep.ClassName()
            prefix = get_obj_repr("Lepton")[lep_idx]
            inst = prefix
            p4 = lep.P4()
            P4s[inst] = p4

            if lepton_type_fn is not None:
                lt = lepton_type_fn(lep_idx, goodLeptons, isLooseSR)
                b[f"LeptonType_{inst}"][0] = int(lt)

            b[f"Charge_{inst}"][0] = int(lep.Charge)
            b[f"IsolationVar_{inst}"][0] = lep.IsolationVar

            if debug_loop:
                print(inst)
                print(
                    f"> Lep: pt={lep.PT:.1f} eta={lep.Eta:.2f} phi={lep.Phi:.2f} "
                    f"m={p4.M():.2f} Q={lep.Charge} iso={lep.IsolationVar:.3f}"
                )

            for k in get_obj_kinematics("Lepton"):
                branch_name = f"{k}_{inst}"
                try:
                    value = getattr(lep, k)
                    b[branch_name][0] = value
                    if debug_loop:
                        print(f"> Filled branch {branch_name}: value = {value}.")
                except AttributeError:
                    try:
                        value = getattr(p4, k)()
                        b[branch_name][0] = value
                        if debug_loop:
                            print(f"> Filled branch {branch_name}: value = {value}.")
                    except AttributeError as e:
                        print(f"Delphes {classname} object has no attribute {k} or .P4().{k}().")
                        raise AttributeError(e)

        # --- FatJets ---
        for fj_idx, fj in enumerate(goodFatJets[: get_obj_count("FatJet")]):
            classname = fj.ClassName()
            prefix = get_obj_repr("FatJet")[fj_idx]
            inst = prefix
            p4 = fj.P4()
            P4s[inst] = p4

            b[f"Tau1_{inst}"][0] = fj.Tau1
            b[f"Tau2_{inst}"][0] = fj.Tau2
            b[f"Tau3_{inst}"][0] = fj.Tau3
            try:
                b[f"Tau21_{inst}"][0] = fj.Tau2 / fj.Tau1 if fj.Tau1 > 0 else np.nan
                b[f"Tau32_{inst}"][0] = fj.Tau3 / fj.Tau2 if fj.Tau2 > 0 else np.nan
            except Exception:
                pass
            if hasattr(fj, "SoftDroppedP4") and fj.SoftDroppedP4.GetEntries() > 0:
                sd_p4 = fj.SoftDroppedP4[0]
                b[f"SoftDropPT_{inst}"][0] = sd_p4.Pt()
                b[f"SoftDropMass_{inst}"][0] = sd_p4.M()

            if debug_loop:
                print(inst)
                print(f"> FJ: pt={fj.PT:.1f} eta={fj.Eta:.2f} phi={fj.Phi:.2f} m={p4.M():.2f}")

            for k in get_obj_kinematics("FatJet"):
                branch_name = f"{k}_{inst}"
                try:
                    value = getattr(fj, k)
                    b[branch_name][0] = value
                    if debug_loop:
                        print(f"> Filled branch {branch_name}: value = {value}.")
                except AttributeError:
                    try:
                        value = getattr(p4, k)()
                        b[branch_name][0] = value
                        if debug_loop:
                            print(f"> Filled branch {branch_name}: value = {value}.")
                    except AttributeError as e:
                        print(f"Delphes {classname} object has no attribute {k} or .P4().{k}().")
                        raise AttributeError(e)

        # --- (AK4) Jets ---
        for jet_idx, jet in enumerate(goodJetsList[: get_obj_count("Jet")]):
            classname = jet.ClassName()
            inst = get_obj_repr("Jet")[jet_idx]
            p4 = jet.P4()
            P4s[inst] = p4

            if debug_loop:
                print(inst)
                print(f"> Jet: pt = {jet.PT}, eta = {jet.Eta}, phi = {jet.Phi}, m = {p4.M()}.")

            for k in get_obj_kinematics("Jet"):
                branch_name = f"{k}_{inst}"
                try:
                    value = getattr(jet, k)
                    b[branch_name][0] = value
                    if debug_loop:
                        print(f"> Filled branch {branch_name}: value = {value}.")
                except AttributeError:
                    try:
                        value = getattr(p4, k)()
                        b[branch_name][0] = value
                        if debug_loop:
                            print(f"> Filled branch {branch_name}: value = {value}.")
                    except AttributeError as e:
                        print(f"Delphes {classname} object has no attribute {k} or .P4().{k}().")
                        raise AttributeError(e)

        # ----------------------------------------------------------------
        # Fill scalars and event shapes
        # ----------------------------------------------------------------

        # Fill the met kinematics
        met = MissingET_branch.At(0)
        p4_met = met.P4()
        P4s["MET"] = p4_met
        for k in get_obj_kinematics("MET"):
            branch_name = f"{k}_MET"
            try:
                value = getattr(met, k)
                b[branch_name][0] = value
            except AttributeError:
                try:
                    value = getattr(p4_met, k)()
                    b[branch_name][0] = value
                except AttributeError as e:
                    print(f"Delphes {met.ClassName()} object has no attribute {k} or .P4().{k}().")
                    raise AttributeError(e)

        # Fill other global scalars
        ht = ScalarHT_branch.At(0).HT
        b["HT"][0] = ht

        lt = sum([lep.PT for lep in goodLeptons])
        st = lt + ht
        meff = st + met.MET
        sig_met = (met.MET / np.sqrt(ht)) if ht > 0 else np.nan
        b["LT"][0] = lt
        b["ST"][0] = st
        b["Significance_MET"][0] = sig_met
        b["Meff"][0] = meff

        sumPT_jets = sum([jet.PT for jet in goodJets])
        sumPT_fjs = sum([fj.PT for fj in goodFatJets])
        sumPT_sdfjs = 0.0
        for fj in goodFatJets:
            if hasattr(fj, "SoftDroppedP4") and fj.SoftDroppedP4.GetEntries() > 0:
                sumPT_sdfjs += fj.SoftDroppedP4[0].Pt()

        b["ScalarSumPT_Jets"][0] = sumPT_jets
        b["ScalarSumPT_FatJets"][0] = sumPT_fjs
        b["ScalarSumPT_SoftDroppedFatJets"][0] = sumPT_sdfjs
        b["ScalarSumPT_Hadronic"][0] = sumPT_jets + sumPT_sdfjs

        # Fill event shapes
        vis_p4s = (
            [lep.P4() for lep in goodLeptons]
            + [fj.P4() for fj in goodFatJets]
            + [jet.P4() for jet in goodJets]
            + [met.P4()]
        )
        px_arr = np.array([p.Px() for p in vis_p4s], dtype=np.float64)
        py_arr = np.array([p.Py() for p in vis_p4s], dtype=np.float64)
        pz_arr = np.array([p.Pz() for p in vis_p4s], dtype=np.float64)
        e_arr = np.array([p.Energy() for p in vis_p4s], dtype=np.float64)

        S, A, C = EventShapes(px_arr, py_arr, pz_arr)
        b["Sphericity"][0] = S
        b["Aplanarity"][0] = A
        b["Circularity"][0] = C
        b["Centrality"][0] = Centrality(px_arr, py_arr, pz_arr, e_arr)

        # ----------------------------------------------------------------
        # Fill N-body relations
        # ----------------------------------------------------------------

        if multiObjects_Nmax >= 2:
            for N in range(2, multiObjects_Nmax + 1, 1):
                for p_names in get_nbody_combinations(
                    N,
                    different_types_only=(not multiObjects_include_same_represenations),
                    combo_objects=multiObjects_combo_objects_set,
                ):
                    p4_list = [P4s.get(n) for n in p_names]
                    if all(p4_list):
                        sufx = "_".join(p_names)

                        # Without start, sum() defaults to 0 (integer),
                        # which would fail because you can't add 0 + TLorentzVector()
                        total_p4 = sum(p4_list[1:], p4_list[0])

                        # fill basic kinematics first
                        for k in get_nbody_kinematics(
                            N, return_basic=True, include_trivial=False
                        ):
                            branch_name = f"{k}_{sufx}"
                            try:
                                value = getattr(total_p4, k)()
                                b[branch_name][0] = value
                            except AttributeError as e:
                                print(f"Cannot fill {k} or .P4().{k}() for {sufx}")
                                raise AttributeError(e)

                        # Default 2-body kinematics (included for any 2-body objects):
                        # ["DeltaR", "DeltaPhi", "DeltaEta", "MtW"] — filled manually.
                        if N == 2:
                            p1 = p4_list[0]
                            p2 = p4_list[1]
                            b[f"DeltaR_{sufx}"][0] = p1.DeltaR(p2)
                            b[f"DeltaPhi_{sufx}"][0] = p1.DeltaPhi(p2)
                            b[f"DeltaEta_{sufx}"][0] = p1.Eta() - p2.Eta()
                            b[f"MtW_{sufx}"][0] = MtW(p1.Pt(), p1.Phi(), p2.Pt(), p2.Phi())

                        # Stransverse mass — only for 2-body leptons/fatjets (slow)
                        if multiObjects_include_mt2 and N == 2:
                            supported = get_obj_instances("Lepton") + get_obj_instances("FatJet")
                            p1 = p4_list[0]
                            p2 = p4_list[1]
                            p1_name = p_names[0]
                            p2_name = p_names[1]
                            if (p1_name in supported) and (p2_name in supported):
                                b[f"MT2_{sufx}"][0] = mt2(
                                    p1.M(),
                                    p1.Px(),
                                    p1.Py(),
                                    p2.M(),
                                    p2.Px(),
                                    p2.Py(),
                                    p4_met.Px(),
                                    p4_met.Py(),
                                    0.0,
                                    0.0,  # neutrino masses
                                )

                        # Extra N-body event shapes if requested
                        if multiObjects_include_trival_kinematics:
                            px_arr = np.array([p.Px() for p in p4_list], dtype=np.float64)
                            py_arr = np.array([p.Py() for p in p4_list], dtype=np.float64)
                            pz_arr = np.array([p.Pz() for p in p4_list], dtype=np.float64)
                            e_arr = np.array([p.Energy() for p in p4_list], dtype=np.float64)
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

        b["weight"][0] = sampleWeight

        # Fill the trees
        trees[sr_key].Fill()
        counts[sr_key] += 1

    summary = " | ".join(f"{k}={counts[k]}" for k in signal_regions_keys)
    if progress:
        print(f"Region yields for {treeName}: {summary}")

    # If a temp directory is requested, spill each non-empty per-SR tree to its
    # own .root file inside that directory and return the list of file paths.
    if temp_dir_path is not None:
        written_paths = []
        for sr_key, tree in trees.items():
            if tree.GetEntries() == 0:
                continue
            out_path = os.path.join(temp_dir_path, sr_key, f"{treeName}.root")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            f_out = ROOT.TFile.Open(out_path, "RECREATE")
            tree.SetDirectory(f_out)
            tree.Write()
            f_out.Close()
            written_paths.append(out_path)
        return written_paths

    return trees
