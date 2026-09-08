import os
import numpy as np
import ROOT
from tqdm import tqdm
from mt2 import mt2

from ..kinematics import EventShapes, Centrality, MtW
from ..branch_names import (
    get_float_branch_names,
    get_int_branch_names,
    get_obj_count,
    get_obj_kinematics,
    get_obj_instances,
    get_obj_repr,
    get_nbody_combinations,
    get_nbody_kinematics,
)


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
print("Using ROOT version:", ROOT.__version__)
print("Using Delphes libraries found at:", DELPHES_PATH)
script_nb_version = 3
print(f"Making datasets with script version : {script_nb_version}")

def isGoodMuon(muonIdx, muonPT, muonEta, muonIso):
    idx_pt_map = {0: 30, 1: 20} #  PT thresholds for leading, subleading, etc. muons
    IsoCutMuon = 0.1 ## looseCut = 0.3, mediumCut = 0.2, tightCut = 0.1
    return (muonIso <= IsoCutMuon) and (muonPT >= idx_pt_map.get(muonIdx, 10)) and (abs(muonEta) <= 2.5)

def isGoodElectron(electronIdx, electronPT, electronEta, electronIso):
    idx_pt_map = {0: 30, 1: 20} #  PT thresholds for leading, subleading, etc. electrons
    IsoCutElectron = 0.2 ## looseCut = 0.3, mediumCut = 0.2, tightCut = 0.1
    return (electronIso < IsoCutElectron) and (electronPT >= idx_pt_map.get(electronIdx, 10)) and (abs(electronEta) <= 2.5)

def loop_tree(
    inputRootFile,
    treeName,
    sampleWeight,
    signal_regions_keys,
    start_entry=0,
    end_entry=None,
    progress=True,
    debug_loop=False,
    max_entries=None,
    temp_dir_path=None,
    # --- helpers (passed in so this module is decoupled from make_dataset_vN.py) ---
    build_chain=None,
    isGoodMuon=None,
    isGoodElectron=None,
    classify_signal_region_key=None,
    # --- configuration knobs (mirror the defaults used in make_dataset_v4.py) ---
    multiObjects_Nmax=3,
    multiObjects_include_same_represenations=True,
    multiObjects_include_trival_kinematics=False,
    multiObjects_include_mt2=True,
    multiObjects_combo_objects_set=("Lepton", "FatJet", "MET"),
    isLooseSR=True,
    lepton_type_fn=None,
):
    """Loop over the Delphes tree and fill the required data.

    If temp_dir_path is given (parallel / disk-spill mode), each per-SR tree is
    written to its own .root file inside temp_dir_path (which mirrors the
    ReaderOutput directory layout).  The function returns a list of created
    file paths instead of in-memory TTrees, keeping the parent process lightweight.

    Helper hooks
    ------------
    build_chain, isGoodMuon, isGoodElectron, classify_signal_region_key,
    lepton_type_fn are intentionally taken as parameters so this function can be
    reused across vN script variants without a circular import.
    """

    # --- safety: the helpers must come from the caller ---
    missing = [
        n
        for n, v in (
            ("build_chain", build_chain),
            ("isGoodMuon", isGoodMuon),
            ("isGoodElectron", isGoodElectron),
            ("classify_signal_region_key", classify_signal_region_key),
        )
        if v is None
    ]
    if missing:
        raise TypeError(
            "loop_tree() missing required helper arguments: " + ", ".join(missing)
        )

    # Setup Delphes
    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)
    FatJet_branch = TreeReader.UseBranch("FatJet")
    Electron_branch = TreeReader.UseBranch("Electron")
    Muon_branch = TreeReader.UseBranch("Muon")
    Jet_branch = TreeReader.UseBranch("Jet")
    MissingET_branch = TreeReader.UseBranch("MissingET")
    ScalarHT_branch = TreeReader.UseBranch("ScalarHT")
    Weight_branch = TreeReader.UseBranch("Weight")

    # Create the branch buffers, books trees and branches
    b = {}
    for branch_name in get_float_branch_names():
        b[branch_name] = np.zeros(1, dtype=np.float64)
    for branch_name in get_int_branch_names():
        b[branch_name] = np.zeros(1, dtype=np.int32)

    trees = {}
    for sr_key in signal_regions_keys:
        tree = ROOT.TTree(treeName, sr_key)
        tree.SetDirectory(0)  # keep in-memory until spill
        for branch_name in get_float_branch_names():
            tree.Branch(branch_name, b[branch_name], f"{branch_name}/D")
        for branch_name in get_int_branch_names():
            tree.Branch(branch_name, b[branch_name], f"{branch_name}/I")
        trees[sr_key] = tree

    # Event loop
    numberOfEntries = TreeReader.GetEntries()
    if end_entry is None or end_entry > numberOfEntries:
        end_entry = numberOfEntries

    counts = {k: 0 for k in signal_regions_keys}
    rng = range(start_entry, end_entry)
    for entry in (tqdm(rng) if progress else rng):
        TreeReader.ReadEntry(entry)

        if max_entries is not None and entry > max_entries:
            break

        if debug_loop:
            print("-" * 80)
            print(f"Processing entry {entry}")

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
