import os
import math
import numpy as np
import ROOT
from parallelization import build_chain
DELPHES_PATH = os.environ.get("DELPHES_PATH", "/home/elsayed/Delphes-3.5.1")
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

def loop_tree(
    inputRootFile,
    treeName,
    eventWeight = 1.0,
    start_entry = 0,
    end_entry = None,
    show_progress = True,
    temp_dir_path = None,
):

    # Read the input file
    Chain = build_chain(inputRootFile)

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

    b = {}   # numpy branch arrays
    for name in branch_names:
        b[name] = np.zeros(1, dtype=np.float64)
        tree.Branch(name, b[name], f"{name}/D")  # /D for Double (64-bit float)

    # Event loop
    numberOfEntries = TreeReader.GetEntries()
    if end_entry is None or end_entry > numberOfEntries:
        end_entry = numberOfEntries
        
    rng = range(start_entry, end_entry)
    for entry in (tqdm(rng) if show_progress else rng):
        
        TreeReader.ReadEntry(entry)

        # Reset branches to np.nan (weight defaults to 0.0 for skipped events)
        for name in b:
            b[name][0] = np.nan
        b["weight"][0] = 0.0

        # Object selection
        # Collect good fatjets (pT > 300 GeV, |eta| < 2.5, in mass window)
        goodFatJets = []
        mWindow = 20
        for i in range(FatJet_branch.GetEntries()):
            fatjet = FatJet_branch.At(i)
            MassOk = mW - mWindow <= fatjet.SoftDroppedP4[0].M() <= mH + mWindow
            if MassOk and fatjet.PT > 300 and abs(fatjet.Eta) <= 2.5:
                goodFatJets.append(fatjet)
        if len(goodFatJets) == 0:
            continue

        # Collect good leptons
        goodLeptons = []
        IsoCutMuon     = 0.1
        IsoCutElectron = 0.2
        for i in range(Muon_branch.GetEntries()):
            muon = Muon_branch.At(i)
            if muon.IsolationVar < IsoCutMuon and muon.PT > 25 and abs(muon.Eta) <= 2.5:
                goodLeptons.append(muon)
        for i in range(Electron_branch.GetEntries()):
            electron = Electron_branch.At(i)
            if electron.IsolationVar < IsoCutElectron and electron.PT > 30 and abs(electron.Eta) <= 2.5:
                goodLeptons.append(electron)
        if len(goodLeptons) == 0:
            continue

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

    # Write and close
    nEntries = tree.GetEntries()
    outFile.cd()
    tree.Write()
    outFile.Close()
    print(f"Written {nEntries} entries to {outputFile}")

    return out