import os
import ROOT
import argparse
from collections import Counter
from itertools import combinations
from tqdm import tqdm

DELPHES_PATH = os.environ.get("DELPHES_HOME", "/home/ammelsayed/softwares/Delphes-3.5.1")
ROOT.gInterpreter.AddIncludePath(DELPHES_PATH)
ROOT.gInterpreter.AddIncludePath(f"{DELPHES_PATH}/classes")
ROOT.gInterpreter.AddIncludePath(f"{DELPHES_PATH}/external")
ROOT.gSystem.Load("libDelphes")
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
ROOT.gInterpreter.Declare('#include "classes/SortableObject.h"')
ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("ATLAS")
ROOT.gStyle.SetPalette(ROOT.kRainbow)
ROOT.TLorentzVector.P4 = lambda self: self  # Z candidate is its own 4-vector, like Delphes objects
print("Using ROOT version:", ROOT.__version__)
print("Using Delphes libraries found at:", DELPHES_PATH)

def build_chain(inputRootFile):
    chain = ROOT.TChain("Delphes")
    if isinstance(inputRootFile, list):
        for f in inputRootFile:
            chain.Add(f)
    elif isinstance(inputRootFile, str):
        chain.Add(inputRootFile)
    else:
        raise TypeError(f"Input must be str or list of root file(s), got {type(inputRootFile).__name__}")
    return chain

def book_histograms():
    h_dict = {}
    for i in range(1, 5):
        name = f"pt_l{i}"
        h_dict[name] = ROOT.TH1D(name, f"; p_{{T}} (l_{{{i}}}) [GeV]; Weighted Events", 50, 0.0, 500.0)

    h_dict["z1Mass"] = ROOT.TH1D("z1Mass", "; m(Z_{1}) [GeV]; Weighted Events", 50, 40.0, 140.0)
    h_dict["z2Mass"] = ROOT.TH1D("z2Mass", "; m(Z_{2}) [GeV]; Weighted Events", 50, 0.0, 120.0)
    h_dict["lepPtSum"] = ROOT.TH1D("lepPtSum", "; #sum p_{T}(l) [GeV]; Weighted Events", 50, 0.0, 1000.0)
    h_dict["m4l"] = ROOT.TH1D("m4l", "; m(4l) [GeV]; Weighted Events", 60, 0.0, 600.0)
    h_dict["absDeltaPhiZ1Z2"] = ROOT.TH1D("absDeltaPhiZ1Z2", "; |#Delta#phi(Z_{1}, Z_{2})|; Weighted Events", 30, 0.0, ROOT.TMath.Pi())

    for val in h_dict.values():
        val.Sumw2()
    return h_dict

def loop_tree(inputRootFile, xsecXlumi=1.0):
    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)
    Jet_branch       = TreeReader.UseBranch("Jet")
    FatJet_branch    = TreeReader.UseBranch("FatJet")
    Electron_branch  = TreeReader.UseBranch("Electron")
    Muon_branch      = TreeReader.UseBranch("Muon")
    MissingET_branch = TreeReader.UseBranch("MissingET")
    ScalarHT_branch  = TreeReader.UseBranch("ScalarHT")
    Weight_branch    = TreeReader.UseBranch("Weight")

    nb_events = TreeReader.GetEntries()

    # Sum of generator weights over the sample: the normalisation anchor.
    sum_w = 0.0
    for entry in range(nb_events):
        TreeReader.ReadEntry(entry)
        sum_w += Weight_branch.At(0).Weight

    # Book histograms for each channel
    hs = {
        "2e2mu": book_histograms(),
        "4e"   : book_histograms(),
        "4mu"  : book_histograms()
    }

    for entry in tqdm(range(TreeReader.GetEntries()), desc="Processing .."):
        TreeReader.ReadEntry(entry)

        if xsecXlumi is not None:
            # xsecXlumi = sigma[fb] * L[fb^-1] is the TOTAL expected yield.
            # Distributing it flat over N events is only correct when every
            # event has the same generator weight.  With event-generation
            # biasing (bias_module = ptl1_bias, ...) the weights are not flat,
            # so the luminosity factor must be
            #     factor = sigma*L / sum_i(w_i)
            # and each event keeps its own w_i on top of it.
            w_gen = xsecXlumi * Weight_branch.At(0).Weight / sum_w
        else:
            w_gen = Weight_branch.At(0).Weight

        # Object selection
        # -----------------------
        goodLeptons = []
        for i in range(Muon_branch.GetEntries()):
            mu = Muon_branch.At(i)
            if abs(mu.Eta) <= 2.7 and mu.PT > 10.0:
                goodLeptons.append((mu, "mu"))
        for i in range(Electron_branch.GetEntries()):
            e = Electron_branch.At(i)
            if abs(e.Eta) <= 2.7 and e.PT > 10.0:
                goodLeptons.append((e, "e"))
        goodLeptons.sort(key=lambda lf: lf[0].PT, reverse=True)
        if len(goodLeptons) < 4:
            continue
        goodLeptons = goodLeptons[:4]

        # Event selection
        # -----------------------
        # count OSSF pairs per flavor (one + and one - each).
        # (2,0)=two e-pairs, (0,2)=two mu-pairs, (1,1)=one of each -> 2e2mu.
        charge_counts = Counter((fl, 1 if lep.Charge > 0 else -1) for lep, fl in goodLeptons)
        e_pairs = min(charge_counts[("e", 1)], charge_counts[("e", -1)])
        mu_pairs = min(charge_counts[("mu", 1)], charge_counts[("mu", -1)])
        ch = {(2, 0): "4e", (0, 2): "4mu", (1, 1): "2e2mu"}.get((e_pairs, mu_pairs))
        if ch is None:
            continue

        # Fill histograms
        # -----------------------
        for i, (lep, _) in enumerate(goodLeptons, start=1):
            hs[ch][f"pt_l{i}"].Fill(lep.PT, w_gen)
            
        hs[ch]["lepPtSum"].Fill(sum([lep[0].PT for lep in goodLeptons]), w_gen)
        hs[ch]["m4l"].Fill(sum([lep[0].P4() for lep in goodLeptons]).M(), w_gen)

        # In 2e2mu channel, each Z is the one corresponding to a OSSF lepton pair
        # In the 4e and 4mu channels, we pick the opposite sign pair, whose invariant
        # mass is closer to mZ = 91.1 GeV.=
        ossf = [(goodLeptons[i][0].P4() + goodLeptons[j][0].P4(), frozenset((i, j)))
                for i, j in combinations(range(4), 2)
                if goodLeptons[i][1] == goodLeptons[j][1] and goodLeptons[i][0].Charge + goodLeptons[j][0].Charge == 0]
        Z1, Z2 = min((sorted((a, b), key=lambda z: abs(z.M() - 91.1))
                      for (a, ia), (b, ib) in combinations(ossf, 2) if ia.isdisjoint(ib)),
                     key=lambda zs: abs(zs[0].M() - 91.1))
        hs[ch]["z1Mass"].Fill(Z1.M(), w_gen)
        hs[ch]["z2Mass"].Fill(Z2.M(), w_gen)
        hs[ch]["absDeltaPhiZ1Z2"].Fill(abs(Z1.P4().DeltaPhi(Z2.P4())), w_gen)

    # save to root file
    for ch, hs_ch in hs.items():
        os.makedirs(f"./Validation/{ch}", exist_ok=True)
        f_out = ROOT.TFile(f"./Validation/{ch}/historgams.root", "RECREATE")
        for name, h in hs_ch.items():
            f_out.cd()
            h.Write(f"h_{name}")
        f_out.Close()

    return hs

def draw(hist, name="hist"):
    c = ROOT.TCanvas("c", "c", 800, 600)
    c.cd()
    c.SetLogy(True)
    hist.SetLineColor(ROOT.kRed)
    hist.SetLineWidth(3)
    hist.Draw("hist")
    c.Update()
    c.SaveAs(f"{name}.png")
    c.Close()

def main():
    parser = argparse.ArgumentParser(description="Analyze Delphes ROOT files") 
    parser.add_argument("--lumi", type=float, default=500.0, help="Luminosity (fb^-1)")
    args = parser.parse_args()

    processes = {
        "Signal": {
            "ggH_4L" : {
                "name"  : "ggH #rightarrow 4l"
                "files" : "./ggH_4L_LO/Events/test1/tag_1_delphes_events.root"
            }
        },
        "Background": {
            "ggZZ_4L": {
                "name"  : "ggZZ #rightarrow 4l"
                "files" : "./ggZZ_4L_LO/Events/test1/tag_1_delphes_events.root"
            },
            "qqZZ_4L": {
                "name"  : "q#bar{q}ZZ #rightarrow 4l"
                "files" : "./qqZZ_4L_LO/Events/test1/tag_1_delphes_events.root"
            } 
        }
    }

    for proc_catagory, proc_data in processes.items():
        if proc_catagory == "Signal"



if __name__ == '__main__':
    main()
