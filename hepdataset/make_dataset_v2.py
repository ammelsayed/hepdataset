#!/usr/bin/env python3
"""
make_dataset.py

Build ML-ready tabular datasets from Delphes ROOT files.

Responsibilities:
    - Read background ROOT paths from bkg_directories.yml
    - Select objects according to SR definitions, read the required kinematics
    - Flatten event-level information into a tabular format (ROOT Tree Bracnhs)
    - Write output datasets in Parquet or CSV format using Pandas.

Author : A.M.M. Elsayed (University of Science and Technology of China)
Email  : ammelsayed@mail.ustc.edu.cn / ahmedphysica@outlook.com
"""

import os
import sys
import time
import uuid
import shutil
import tempfile
import subprocess
import numpy as np
import pandas as pd
import itertools
import tabulate
from kinematics import EventShapes, Centrality, MtW
from branch_names import (

    get_float_branch_names, get_int_branch_names, get_obj_repr, get_obj_count, get_obj_kinematics,
    get_obj_instances, get_nbody_combinations, get_nbody_kinematics, print_summary
)
from yaml import safe_load as yml_safe_load
from tqdm import tqdm
from mt2 import mt2
import ROOT
from parallelization import parallel_runs, format_time
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

## =============================
## Run options
## =============================

debug = 1
debug_loop = 0
LooseSR = True
multiObjects_Nmax = 3
multiObjects_include_same_represenations = True
multiObjects_include_trival_kinematics = False
multiObjects_include_mt2 = True # only calculated for 2-body objects (leptons and fatjets (see main lop))
multiObjects_combo_objects_set = ("Lepton", "FatJet", "MET")

print_summary()


# test options (set to false for production)
testCode = False
testSamplePaths = ["/data/ammelsayed/stuff/root_files/tt012j_MLM_run_01_01.root", "/data/ammelsayed/stuff/root_files/tt012j_MLM_run_01_02.root"]

# output directories
baseDir = "ReaderOutput" if not testCode else "ReaderOutput_Test"
outputRootFileName = "mc_events.root"

## =============================
## Background processes involved
## =============================

# check root health only on first time running this script
def clean(args, check_root=True):
    valid = [] 
    for path in args:
        if not os.path.isfile(path):
            continue
        if not check_root:
            valid.append(path)
            continue
        root_file = ROOT.TFile.Open(path)
        try:
            if not root_file or root_file.IsZombie():
                continue
            if not root_file.Get("Delphes"):
                continue
            valid.append(path)
        finally:
            if root_file and not root_file.IsZombie():
                root_file.Close()

    n_unvalid = len(args) - len(valid)
    if n_unvalid > 0:
        print(f"Found {n_unvalid} unvalid files.")
        print(f" > Input: {args}")

    return list(set(valid)) # use set to auto-remove duplicates


def get_paths_from_yml(bkg_proc_name):
    print(f"Reading .root files paths for {bkg_proc_name}")
    with open('/data/ammelsayed/stuff/bkgModeling/bkg_directories.yml', 'r') as f:
        data = yml_safe_load(f)
    return clean(data[bkg_proc_name])


def get_processes():

    bkg_processes = {
        "tt": {
            "dir": get_paths_from_yml("tt"),
            "cross_section_[pb]": 348.3,
        },
        "tW": {
            "dir": get_paths_from_yml("tW"),
            "cross_section_[pb]": 526.9,
        },
        "ttX": {
            "dir": get_paths_from_yml("ttX"),
            "cross_section_[pb]": 1.485,
        },
        "SSWW": {
            "dir": get_paths_from_yml("SSWW"),
            "cross_section_[pb]": 0.6805,
        },
        "OSWW": {
            "dir": get_paths_from_yml("OSWW"),
            "cross_section_[pb]": 247.6,
        },
        "WZ": {
            "dir": get_paths_from_yml("WZ"),
            "cross_section_[pb]": 37.45,
        },
        "ZZ": {
            "dir": get_paths_from_yml("ZZ"),
            "cross_section_[pb]": 13.11,
        },
        "W_012j": {
            "dir": get_paths_from_yml("W+012j"),
            "cross_section_[pb]": 1.274e+04,
        },
        "Z_012j": {
            "dir": get_paths_from_yml("Z+012j"),
            "cross_section_[pb]": 1784,
        }
    }


    signal_cross_sections = pd.read_csv("./signal_cross_sections.csv")
    def get_signal_cross_section(mass):
        return float(signal_cross_sections.loc[signal_cross_sections["mass"] == mass, "nlo_xsec"].values[0]/1000)

    sig_processes = {}
    M_min, M_max, M_step = 1000, 2000, 50
    for mass in range(M_min, M_max + M_step, M_step):
        sig_processes[f"sig{mass}"] = {
            "dir" : [f"/data/ammelsayed/Framework/Samples/signal/FDM/Events/sig{mass}/tag_1_delphes_events.root"],
            "cross_section_[pb]" : get_signal_cross_section(mass)
        }

    processes = {}
    for key, values in bkg_processes.items():
        processes[key] = values
    for key, values in sig_processes.items():
        processes[key] = values
    
    return processes


## =============================
## Define signal regions
## =============================

def get_signal_regions(isLoose = True):
    if isLoose:
        signal_regions = {
            "0L"  : ["JJ"],
            "1L"  : ["lepJ", "lepJJ"],
            "2OS" : ["leplepJ"],
            "2SS" : ["leplepJ"],
            "3L"  : ["lepleplep"],
            # "4L"  : ["leplepleplep"]
        }
    else:
        signal_regions = {
            "0L"  : ["JJ"],
            "1L"  : ["eJ", "muJ", "eJJ", "muJJ"],
            "2OS" : ["eeJ", "emuJ", "mumuJ"],
            "2SS" : ["eeJ", "emuJ", "mumuJ"],
            "3L"  : ["eee", "eemu", "emumu", "mumumu"],
            # "4L"  : ["eeee", "eeemu", "eemumu", "emumumu", "mumumumu"]
        }

    signal_regions_keys = [
        f"{channel}_{region}"  
        for channel, regions in signal_regions.items()
        for region in regions
    ]

    return signal_regions, signal_regions_keys

def lepton_type(idx, leptons, isLoose = True):
    if isLoose:
        return "lep"
    else:
        return "e" if leptons[idx].ClassName().startswith("Electron") else "mu"

def classify_signal_region_key(goodLeptons, goodFatJets, isLoose = True):
    n_leps = len(goodLeptons)
    n_fatjets = len(goodFatJets)

    ## Add this to test code
    if testCode:
        return "test_test1"

    # 0 leptons channel
    if n_leps == 0:
        if n_fatjets >= 2: return "0L_JJ"
        else: return None
    
    # 1 lepton channel
    if n_leps == 1:
        lep_type = lepton_type(0, goodLeptons, isLoose)
        if n_fatjets == 1: return f"1L_{lep_type}J"
        elif n_fatjets >= 2: return f"1L_{lep_type}JJ"
        else: return None

    # 2 leptons channel
    elif n_leps == 2:
        lep1_type = lepton_type(0, goodLeptons, isLoose)
        lep2_type = lepton_type(1, goodLeptons, isLoose)
        lep1_charge = goodLeptons[0].Charge
        lep2_charge = goodLeptons[1].Charge
        total_charge = lep1_charge + lep2_charge
        lep_types = sorted([lep1_type, lep2_type])
        if total_charge == 0: # opposite sign
            if n_fatjets >= 1: return f"2OS_{lep_types[0]}{lep_types[1]}J"
            else: return None
        else: # same sign
            if n_fatjets >= 1: return f"2SS_{lep_types[0]}{lep_types[1]}J"
            else: return None
    
    # 3 leptons channel
    elif n_leps >= 3:
        lep1_type = lepton_type(0, goodLeptons, isLoose)
        lep2_type = lepton_type(1, goodLeptons, isLoose)
        lep3_type = lepton_type(2, goodLeptons, isLoose)
        lep_types = sorted([lep1_type, lep2_type, lep3_type])
        if n_fatjets >= 0: return f"3L_{lep_types[0]}{lep_types[1]}{lep_types[2]}"
        else:return None
    
    # elif n_leps == 4:
    #     lep1_type = lepton_type(0, goodLeptons, isLoose)
    #     lep2_type = lepton_type(1, goodLeptons, isLoose)
    #     lep3_type = lepton_type(2, goodLeptons, isLoose)
    #     lep4_type = lepton_type(3, goodLeptons, isLoose)
    #     lep_types = sorted([lep1_type, lep2_type, lep3_type, lep4_type])
    #     if n_fatjets >= 0: return f"4L_{lep_types[0]}{lep_types[1]}{lep_types[2]}{lep_types[3]}"
    #     else:return None
    
    else:
        return None


## =============================
## Helper functions for parallelization
## =============================

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

def split_range(total, n):
    n = max(1, min(n, total))
    base, rem = divmod(total, n)
    out, start = [], 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        out.append((start, start + size))
        start += size
    return out


## Main reader loop and quality selections
## ==========================================

def isGoodMuon(muonIdx, muonPT, muonEta, muonIso):
    idx_pt_map = {0: 30, 1: 20} #  PT thresholds for leading, subleading, etc. muons
    IsoCutMuon = 0.1 ## looseCut = 0.3, mediumCut = 0.2, tightCut = 0.1
    return (muonIso <= IsoCutMuon) and (muonPT >= idx_pt_map.get(muonIdx, 10)) and (abs(muonEta) <= 2.5)

def isGoodElectron(electronIdx, electronPT, electronEta, electronIso):
    idx_pt_map = {0: 30, 1: 20} #  PT thresholds for leading, subleading, etc. electrons
    IsoCutElectron = 0.2 ## looseCut = 0.3, mediumCut = 0.2, tightCut = 0.1
    return (electronIso < IsoCutElectron) and (electronPT >= idx_pt_map.get(electronIdx, 10)) and (abs(electronEta) <= 2.5)

def loop_tree(inputRootFile, treeName, sampleWeight, signal_regions_keys, start_entry=0, end_entry=None, progress=True, debug_loop = False, max_entries=None):
    """ Loop over the Delphes tree and fill the required data. """

    # Setup Delphes
    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)
    FatJet_branch    = TreeReader.UseBranch("FatJet")
    Electron_branch  = TreeReader.UseBranch("Electron")
    Muon_branch      = TreeReader.UseBranch("Muon")
    Jet_branch       = TreeReader.UseBranch("Jet")
    MissingET_branch = TreeReader.UseBranch("MissingET")
    ScalarHT_branch  = TreeReader.UseBranch("ScalarHT")
    Weight_branch     = TreeReader.UseBranch("Weight")

    ## Create the branch buffers, books trees and branches
    b = {} 
    for branch_name in get_float_branch_names():
        b[branch_name] = np.zeros(1, dtype=np.float64)
    for branch_name in get_int_branch_names():
        b[branch_name] = np.zeros(1, dtype=np.int32)

    trees = {}
    for sr_key in signal_regions_keys:
        tree = ROOT.TTree(treeName, sr_key)
        tree.SetDirectory(0)
        for branch_name in get_float_branch_names():
            tree.Branch(branch_name, b[branch_name], f"{branch_name}/D")  # /D for Double (64-bit float)
        for branch_name in get_int_branch_names():
            tree.Branch(branch_name, b[branch_name], f"{branch_name}/I")  # /I for Integer
        trees[sr_key] = tree
    
    ## Event loop
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
            print("-"*80)
            print(f"Processing entry {entry}")
   
        # Reset branches to np.nan (weight defaults to 0.0 for skipped events)
        for branch_name in get_float_branch_names():
            b[branch_name][0] = np.nan
        for branch_name in get_int_branch_names():
            b[branch_name][0] = -1 # Use -1 as sentinel for integer branches since np.nan is unsupported
        

        # Count number of objects before quality selections AND store them locally
        nPreQS_Muon = Muon_branch.GetEntries()
        nPreQS_Electron = Electron_branch.GetEntries()
        nPreQS_Lepton = nPreQS_Muon + nPreQS_Electron
        nPreQS_FatJet = FatJet_branch.GetEntries()
        nPreQS_Jet = Jet_branch.GetEntries()

        if debug_loop:
            print(f"Number of objects before quality selections:")
            print(f" mu = {nPreQS_Muon}, e = {nPreQS_Electron}, J = {nPreQS_FatJet}, j = {nPreQS_Jet}")
       
        ##################################################################
        ##################################################################
        # Quality Selections
        ##################################################################
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
        
        # ----------------------------------------------------------------
        # Collect good fatjets
        # ----------------------------------------------------------------
        goodFatJets = []
        mWindow = 25
        mW, mZ, mH = 80.4, 91.2, 125
        for i in range(FatJet_branch.GetEntries()):
            fatjet = FatJet_branch.At(i)
            MassOk = mW - mWindow <= fatjet.SoftDroppedP4[0].M() <= mH + mWindow
            if len(goodLeptons) > 0:
                DeltaROk = min(fatjet.P4().DeltaR(lep.P4()) for lep in goodLeptons) > 0.4
            else:
                DeltaROk = True
            if DeltaROk and MassOk and fatjet.PT > 200.0 and abs(fatjet.Eta) <= 2.5:
                goodFatJets.append(fatjet)

        goodFatJets.sort(key=lambda fj: fj.PT, reverse=True)

        # ----------------------------------------------------------------
        # Collect good jets, bjets, and taujets
        # ----------------------------------------------------------------
        goodJets, goodBJets, goodTauJets = [], [], []
        nPreQS_Jet, nPreQS_BJet, nPreQS_TauJet = 0, 0, 0
        bWP = 0     ## Working points: 0 - Loose , 1 - Medium, 2 - Tight (Use 0 for standard Delphes)
        tauWP = 0   ## Working points: 0 - Loose , 1 - Medium, 2 - Tight (Use 0 for standard Delphes)
        for i in range(Jet_branch.GetEntries()):
            jet = Jet_branch.At(i)
            nPreQS_Jet += 1
            BtagOk = jet.BTag & (1 << bWP)
            TautagOk = jet.TauTag & (1 << tauWP)
            # All jets must pass basic quality requirments
            if jet.PT > 30 and abs(jet.Eta) <= 2.5:
                goodJets.append(jet)
                if BtagOk:
                    nPreQS_BJet += 1
                    goodBJets.append(jet)
                if TautagOk:
                    nPreQS_TauJet += 1
                    goodTauJets.append(jet)

        goodJets.sort(key=lambda jet: jet.PT, reverse=True) 
        goodBJets.sort(key=lambda jet: jet.PT, reverse=True) 
        goodTauJets.sort(key=lambda jet: jet.PT, reverse=True)

        if debug_loop:
            print(f"Number of objects after quality selections:")
            print(f" mu = {len(goodMuons)}, e = {len(goodElectrons)}, J = {len(goodFatJets)}, j = {len(goodJets)}")
       
        ##################################################################
        ##################################################################
        # Signal Region Selections
        ##################################################################
        ##################################################################

        sr_key = classify_signal_region_key(goodLeptons, goodFatJets)
        if sr_key is None:
            continue

        if debug_loop:
            print(f"Number of objects after object selections:")
            print(f" mu = {len(goodMuons)}, e = {len(goodElectrons)}, J = {len(goodFatJets)}, j = {len(goodJets)}, b = {len(goodBJets)}, tau = {len(goodTauJets)}")
            print(f"Signal Region: {sr_key}")
       

        ################################################
        # Fill the branches
        ################################################

        P4s = {} # Dictionary to cache 4-momenta for combinations

        # Fill number of objects: the local variables nPreQS_* safely hold the counts from before the 'continue's
        b["nPreQS_Muon"][0] = nPreQS_Muon;          b["nPostES_Muon"][0] = len(goodMuons)
        b["nPreQS_Electron"][0] = nPreQS_Electron;  b["nPostES_Electron"][0] = len(goodElectrons)
        b["nPreQS_Jet"][0] = nPreQS_Jet;            b["nPostES_Jet"][0] = len(goodJets)
        b["nPreQS_BJet"][0] = nPreQS_BJet;          b["nPostES_BJet"][0] = len(goodBJets)
        b["nPreQS_TauJet"][0] = nPreQS_TauJet;      b["nPostES_TauJet"][0] = len(goodTauJets)
        b["nPreQS_FatJet"][0] = nPreQS_FatJet;      b["nPostES_FatJet"][0] = len(goodFatJets)

        # ----------------------------------------------------------------
        # Fill leptons kinematics
        # ---------------------------------------------------------------

        muonMass = 0.1057
        electronMass = 0.511E-3

        for lep_idx, lepton in enumerate(goodLeptons[:get_obj_count("Lepton")]):

            mainClassNames = ["Lepton"] # branches including these class names will be filled

            lepClassName = lepton.ClassName()
            if lepClassName in get_obj_repr("Lepton"):
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
                for k in get_obj_kinematics("Lepton"):

                    branch_name = f"{k}_{inst}"

                    # Optional: check if this branch name is in get_float_branch_names() or not, skipped here,
                    # because if not, then it will not filled in the tree anyways, and will raise a key error.

                    try:

                        value = getattr(lepton, k) # example: lep.PT
                        b[branch_name][0] = value 

                        if debug_loop:
                            print(f"> Filled branch {branch_name}: value = {value}.")

                    except AttributeError:

                        try:

                            value = getattr(p4, k)() # example: lep.P4().Px()
                            b[branch_name][0] = value 

                            if debug_loop:
                                print(f"> Filled branch {branch_name}: value = {value}.")

                        except AttributeError as e:
                            print(f"Delphes {lepClassName} object has no attribute {k} or .P4().{k}(). Error: {e}")
                            raise AttributeError(e)
                            

        # ----------------------------------------------------------------
        # Fill fatjet kinematics
        # ----------------------------------------------------------------

        fill_softdroped = True
        fill_trimmed = False

        for fj_idx, fatjet in enumerate(goodFatJets[:get_obj_count("FatJet")]):
            
            classname = fatjet.ClassName()

            p4_fj = fatjet.P4()
            P4s[f"FatJet{fj_idx}"] = p4_fj
            if debug_loop:
                print(f"FatJet{fj_idx}")
                print(f"> FatJet: pt = {fatjet.PT}, eta = {fatjet.Eta}, phi = {fatjet.Phi}, m = {fatjet.Mass}.")

            if hasattr(fatjet, 'SoftDroppedP4') and ("SoftDroppedFatJet" in get_obj_repr("FatJet")):
                p4_sd = fatjet.SoftDroppedP4[0]
                P4s[f"SoftDroppedFatJet{fj_idx}"] = p4_sd
                if debug_loop:
                    print(f"> SoftDroppedFatJet: pt = {p4_sd.Pt()}, eta = {p4_sd.Eta()}, phi = {p4_sd.Phi()}, m = {p4_sd.M()}.")

            if hasattr(fatjet, 'TrimmedP4') and ("TrimmedFatJet" in get_obj_repr("FatJet")):
                p4_tr = fatjet.TrimmedP4[0]            
                P4s[f"TrimmedFatJet{fj_idx}"] = p4_tr
                if debug_loop:
                    print(f"> TrimmedFatJet: pt = {p4_tr.Pt()}, eta = {p4_tr.Eta()}, phi = {p4_tr.Phi()}, m = {p4_tr.M()}.")

            # Fill kinematics asked for
            for k in get_obj_kinematics("FatJet"):

                # n-subjetness variables are handeled seperatly
                if k in ["Tau1", "Tau2", "Tau3", "Tau21", "Tau32"]: 
                    continue
                
                # Fill kinematics of the FatJet
                try:
                    value = getattr(fatjet, k)
                    b[f"{k}_FatJet{fj_idx}"][0] = value
                    if debug_loop:
                        print(f"> Filled branch {k}_FatJet{fj_idx}: value = {value}.")
                except AttributeError:
                    try:
                        value = getattr(p4_fj, k)()
                        b[f"{k}_FatJet{fj_idx}"][0] = value
                        if debug_loop:
                            print(f"> Filled branch {k}_FatJet{fj_idx} from P4: value = {value}.")
                    except AttributeError as e:
                        print(f"Delphes {classname} object has no attribute {k} or .P4().{k}().")
                        raise AttributeError(e)

                # Fill same kinematics for the softdropped jet
                # grooming variants do not have direct methods
                # so we just use one layer of reading 
                if hasattr(fatjet, 'SoftDroppedP4') and ("SoftDroppedFatJet" in get_obj_repr("FatJet")):

                    try:
                        value_sd = getattr(p4_sd, k)()
                        b[f"{k}_SoftDroppedFatJet{fj_idx}"][0] = value_sd

                        if debug_loop:
                            print(f"> Filled branch {k}_SoftDroppedFatJet{fj_idx} from SoftDroppedP4: value = {value_sd}.")

                    except AttributeError as e:
                        print(f"Delphes {classname} object has no attribute .SoftDroppedP4[0]().{k}().")
                        raise AttributeError(e)

                # Fill same kinematics for the trimmed jet
                if hasattr(fatjet, 'TrimmedP4') and ("TrimmedFatJet" in get_obj_repr("FatJet")):

                    try:
                        value_tr = getattr(p4_tr, k)()
                        b[f"{k}_TrimmedFatJet{fj_idx}"][0] = value_tr

                        if debug_loop:
                            print(f"> Filled branch {k}_TrimmedFatJet{fj_idx} from TrimmedP4: value = {value_tr}.")

                    except AttributeError as e:
                        print(f"Delphes {classname} object has no attribute .TrimmedP4[0]().{k}().")
                        raise AttributeError(e)

            # Tau substructures
            # Those are only filled for FatJet
            # The groomed variants do not carry these methods
            for t_idx in [1, 2, 3]:
                b[f"Tau{t_idx}_FatJet{fj_idx}"][0] = fatjet.Tau[t_idx - 1]

                if debug_loop:
                    print(f"> Filled branch Tau{t_idx}_FatJet{fj_idx}: value = {fatjet.Tau[t_idx - 1]}.")

            t1, t2, t3 = fatjet.Tau[0], fatjet.Tau[1], fatjet.Tau[2]
            b[f"Tau21_FatJet{fj_idx}"][0] = t2 / t1 if t1 > 0 else np.nan
            b[f"Tau32_FatJet{fj_idx}"][0] = t3 / t2 if t2 > 0 else np.nan
            if debug_loop:
                print(f"> Filled branch Tau21_FatJet{fj_idx}: value = {b[f'Tau21_FatJet{fj_idx}'][0]}.")
                print(f"> Filled branch Tau32_FatJet{fj_idx}: value = {b[f'Tau32_FatJet{fj_idx}'][0]}.")

        # ----------------------------------------------------------------
        # Fill jet kinematics
        # ----------------------------------------------------------------
        for goodJetsList, prefix in zip([goodJets, goodBJets, goodTauJets], ["Jet", "BJet", "TauJet"]):

            if prefix not in get_obj_repr("Jet"):
                continue

            for jet_idx, jet in enumerate(goodJetsList[:get_obj_count("Jet")]):
                classname = jet.ClassName()
                inst = f"{prefix}{jet_idx}"
                p4 = jet.P4()
                P4s[inst] = p4

                if debug_loop:
                    print(inst)
                    print(f"> Jet: pt = {jet.PT}, eta = {jet.Eta}, phi = {jet.Phi}, m = {p4.M()}.")

                for k in get_obj_kinematics("Jet"):

                    branch_name = f"{k}_{inst}"

                    try:

                        value = getattr(jet, k) # example: jet.PT
                        b[branch_name][0] = value 

                        if debug_loop:
                            print(f"> Filled branch {branch_name}: value = {value}.")

                    except AttributeError:

                        try:

                            value = getattr(p4, k)() # example: jet.P4().PT()
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
        sig_met = ( met.MET / np.sqrt(ht) ) if ht > 0 else np.nan
        b["LT"][0] = lt
        b["ST"][0] = st
        b["Significance_MET"][0] = sig_met
        b["Meff"][0] = meff

        sumPT_jets = sum([jet.PT for lep in goodJets])
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

        if multiObjects_Nmax >= 2:

            for N in range(2, multiObjects_Nmax + 1, 1):

                for p_names in get_nbody_combinations(N, different_types_only = multiObjects_include_same_represenations == False, combo_objects= multiObjects_combo_objects_set):

                    p4_list = [P4s.get(n) for n in p_names]

                    if all(p4_list):

                        sufx = '_'.join(p_names)

                        # Without start, sum() defaults to  0 (integer), 
                        # which would fail because you can't add 0 + TLorentzVector()
                        total_p4 = sum(p4_list[1:], p4_list[0])

                        # fill basic kinematics first 
                        for k in get_nbody_kinematics(N, return_basic = True, include_trivial = False):
                            branch_name = f"{k}_{sufx}"
                            try:
                                value = getattr(total_p4, k)() 
                                b[branch_name][0] = value 
                            except AttributeError as e:
                                print(f"Cannot fill {k} or .P4().{k}() for {sufx}")
                                raise AttributeError(e)

                        # by default 2body kinematics should be included for any 2-body objects
                        # those are just ["DeltaR", "DeltaPhi", "DeltaEta", "MtW"]
                        # we fill them manullay
                        if N == 2:
                            p1 = p4_list[0]; p2 = p4_list[1]
                            b[f"DeltaR_{sufx}"][0] = p1.DeltaR(p2)
                            b[f"DeltaPhi_{sufx}"][0] = p1.DeltaPhi(p2)
                            b[f"DeltaEta_{sufx}"][0] = p1.Eta() - p2.Eta()
                            b[f"MtW_{sufx}"][0] = MtW(p1.Pt(), p1.Phi(), p2.Pt(), p2.Phi())
                        

                        ## Here we wish to calculate the stransverse mass
                        ## We just calclate it for 2 body objects, 
                        ## and only for lepton and fatjet pars
                        ## this because calculating mt2 can be slow
                        if multiObjects_include_mt2 and N == 2:
                            supported = get_obj_instances("Lepton") + get_obj_instances("FatJet")
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
                        if multiObjects_include_trival_kinematics:
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
        
        b["weight"][0] = sampleWeight

        # Fill the trees
        trees[sr_key].Fill()
        # counts[sr_key] += w_gen
        counts[sr_key] += 1

    summary = " | ".join(f"{k}={counts[k]}" for k in signal_regions_keys)
    if progress:
        print(f"Region yields for {treeName}: {summary}")
    return trees


def merge_outputs(results, signal_regions_keys, treeName):
    """results = list of {sr_key: TTree} dicts from parallel workers."""
    print("Merging ..")
    merged_trees = {}
    for sr_key in signal_regions_keys:
        tl = ROOT.TList()
        for trees in results:
            tl.Add(trees[sr_key])
        merged = ROOT.TTree.MergeTrees(tl)
        merged.SetName(treeName)
        merged.SetTitle(sr_key)
        merged.SetDirectory(0)   # keep detached; read() will cd() + Write()
        merged_trees[sr_key] = merged
    return merged_trees


def loop_tree_advanced(inputRootFile, treeName, sampleWeight, signal_regions_keys, run_parallel=True, n_chunks=None, max_workers=None, max_entries=None):
    if not run_parallel:
        return loop_tree(inputRootFile, treeName, sampleWeight, signal_regions_keys, max_entries=max_entries)

    total = count_entries(inputRootFile)
    n_chunks = n_chunks or (max_workers or os.cpu_count())
    chunks = [(inputRootFile, treeName, sampleWeight, signal_regions_keys, s, e, False) for s, e in split_range(total, n_chunks)]
    results = parallel_runs(loop_tree, chunks, max_workers=max_workers, info="", mpContext="fork")
    for r in results:
        if isinstance(r, Exception):
            raise r

    return merge_outputs(results, signal_regions_keys, treeName)


def read(processes, signal_regions, signal_regions_keys):

    print("\n")
    
    # output root files paths
    paths = {}
    for channelName, regionNamesList in signal_regions.items():
        for regionName in regionNamesList:
            key = f"{channelName}_{regionName}"
            paths[key] = os.path.join(baseDir, channelName, regionName, outputRootFileName)

    # build output root files
    outputRootFilesDict = {}
    for key, path in paths.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        outputRootFile = ROOT.TFile.Open(path, "RECREATE")
        if not outputRootFile or outputRootFile.IsZombie():
            print(f"Cannot create ROOT file: {path}")
            return
        outputRootFilesDict[key] = outputRootFile
    
    for proc, metadata in processes.items():
        inputRootFilesList = metadata['dir']
        TotalCrossSection = metadata['cross_section_[pb]']

        if not inputRootFilesList:
            print(f"Skipping {proc}: no input files.")
            continue

        print(f"Processing {proc}: {len(inputRootFilesList)} sample root files.")
        numEvents = count_entries(inputRootFilesList)
        if numEvents == 0:
            print(f"Skipping {proc}: zero events.")
            continue
        perEventWeight = TotalCrossSection * 1000 * 400 / numEvents
        print(f"    > Total number of events : {numEvents}")
        print(f"    > Per-event weight : {perEventWeight}")

        for idx, inputRootFile in enumerate(inputRootFilesList, start = 1):
            treeName = f"{proc}_sample{idx}"
            trees = loop_tree_advanced(inputRootFile, treeName, sampleWeight=perEventWeight, signal_regions_keys=signal_regions_keys)
            for sr_key, tree in trees.items():
                nEntries = tree.GetEntries()
                outputRootFilesDict[sr_key].cd()
                tree.Write(treeName, ROOT.TObject.kOverwrite)
                print(f"Written {nEntries} entries into TTree {treeName} at TFile {paths[sr_key]}")
            print("\n")
    for sr_key in signal_regions_keys:
        outputRootFilesDict[sr_key].Close()


def make_datasets():

    signal_regions, signal_regions_keys = get_signal_regions(isLoose = LooseSR)
    print("Signal regions:")
    for key, values in signal_regions.items():
        print(f"{key:<5} : {values}")

    read(get_processes(), signal_regions, signal_regions_keys)

def test():
    proc = {
    "test1_s": {
        "dir": clean(testSamplePaths), "cross_section_[pb]": 1,
    }}
    read(proc, {"test" : ["test1"]}, ["test_test1"])


if __name__ == '__main__':

    if testCode:
        test()
    else:
        make_datasets()


