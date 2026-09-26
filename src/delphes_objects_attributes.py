#!/usr/bin/env python3
import os, sys, ROOT

DELPHES = os.environ.get("DELPHES_HOME", "/home/ammelsayed/softwares/MG5_aMC_v3_5_15/Delphes")
for p in (DELPHES, f"{DELPHES}/classes", f"{DELPHES}/external"):
    ROOT.gInterpreter.AddIncludePath(p)
ROOT.gSystem.Load("libDelphes")
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gROOT.SetBatch(True)

chain  = ROOT.TChain("Delphes"); chain.Add(sys.argv[1])
reader = ROOT.ExRootTreeReader(chain)

NAMES    = ["Muon", "Electron", "Jet", "FatJet", "MissingET", "ScalarHT"]
branches = {n: reader.UseBranch(n) for n in NAMES}      # register ONCE, before ANY ReadEntry

def dump_object(obj):
    cls = ROOT.TClass.GetClass(obj.ClassName())

    print(f"\n=== {obj.ClassName()} ===")

    # ---- 1. public data members of the Delphes object itself ----
    print("  -- data members --")
    for dm in cls.GetListOfDataMembers():
        if not (dm.Property() & ROOT.kIsPublic):
            continue
        n = dm.GetName()
        try:
            print(f"    {n:20s} = {getattr(obj, n)}")
        except Exception as e:
            print(f"    {n:20s}   <err: {e}>")

    # ---- 2. TLorentzVector returned by P4() ----
    has_p4 = any(m.GetName() == "P4" for m in cls.GetListOfMethods())
    if not has_p4:
        return   # e.g. ScalarHT, MissingET (no P4)

    try:
        p4 = obj.P4()
    except Exception as e:
        print(f"  P4() failed: {e}")
        return

    print("  -- P4() TLorentzVector methods (no-arg, public) --")
    p4cls = ROOT.TClass.GetClass("TLorentzVector")
    seen = set()
    for mt in p4cls.GetListOfMethods():
        if not (mt.Property() & ROOT.kIsPublic):
            continue
        n = mt.GetName()
        if n.startswith("_") or "operator" in n:
            continue
        # only methods that need zero arguments (overloads with defaults excluded)
        if mt.GetNargs() != 0 or mt.GetNargsOpt() != 0:
            continue
        if n in seen:
            continue
        seen.add(n)
        try:
            v = getattr(p4, n)()
            print(f"    P4().{n:20s}() = {v}")
        except Exception as e:
            print(f"    P4().{n:20s}()   <err: {e}>")

# in your main loop:
for name in NAMES:
    br = branches[name]
    obj = None
    for i in range(reader.GetEntries()):
        reader.ReadEntry(i)
        if br.GetEntries() > 0:
            obj = br.At(0)
            break
    if obj is None:
        print(f"\n=== {name}: no objects found ===")
        continue
    dump_object(obj)