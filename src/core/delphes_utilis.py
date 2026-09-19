import os
import ROOT

def load_delphes():
    DELPHES_PATH = os.environ.get("DELPHES_HOME", "/home/ammelsayed/softwares/MG5_aMC_v3_7_2/Delphes")
    ROOT.gInterpreter.AddIncludePath(DELPHES_PATH)
    ROOT.gInterpreter.AddIncludePath(f"{DELPHES_PATH}/classes")
    ROOT.gInterpreter.AddIncludePath(f"{DELPHES_PATH}/external")
    ROOT.gSystem.Load("libDelphes")
    ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
    ROOT.gInterpreter.Declare('#include "classes/SortableObject.h"')
    ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
    ROOT.gROOT.SetBatch(True)
    # print("Using ROOT version:", ROOT.__version__)
    # print("Using Delphes libraries found at:", DELPHES_PATH)
    # print("\n")
    return None

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