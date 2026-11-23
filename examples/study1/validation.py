import os
import ROOT
import argparse
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from pathlib import Path

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
            chain.Add(str(Path(f).resolve()))
    elif isinstance(inputRootFile, str):
        chain.Add(str(Path(inputRootFile).resolve()))
    else:
        raise TypeError(f"Input must be str or list of root file(s), got {type(inputRootFile).__name__}")
    return chain

def book_histograms(prefix=""):
    h_dict = {}
    for i in range(1, 5):
        name = f"pt_l{i}_{prefix}"
        h_dict[name] = ROOT.TH1D(name, f"; p_{{T}} (l_{{{i}}}) [GeV]; Weighted Events", 50, 0.0, 500.0)

    h_dict[f"z1Mass_{prefix}"] = ROOT.TH1D(f"z1Mass_{prefix}", "; m(Z_{1}) [GeV]; Weighted Events", 50, 40.0, 140.0)
    h_dict[f"z2Mass_{prefix}"] = ROOT.TH1D(f"z2Mass_{prefix}", "; m(Z_{2}) [GeV]; Weighted Events", 50, 0.0, 120.0)
    h_dict[f"lepPtSum_{prefix}"] = ROOT.TH1D(f"lepPtSum_{prefix}", "; #sum p_{T}(l) [GeV]; Weighted Events", 50, 0.0, 1000.0)
    h_dict[f"m4l_{prefix}"] = ROOT.TH1D(f"m4l_{prefix}", "; m(4l) [GeV]; Weighted Events", 60, 0.0, 600.0)
    h_dict[f"absDeltaPhiZ1Z2_{prefix}"] = ROOT.TH1D(f"absDeltaPhiZ1Z2_{prefix}", "; |#Delta#phi(Z_{1}, Z_{2})|; Weighted Events", 30, 0.0, ROOT.TMath.Pi())

    for val in h_dict.values():
        val.Sumw2()
    return h_dict

def loop_tree(inputRootFile, xsec=1.0):
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
        "2e2mu": book_histograms(prefix="2e2mu"),
        "4e"   : book_histograms(prefix="4e"),
        "4mu"  : book_histograms(prefix="4mu")
    }

    for entry in tqdm(range(TreeReader.GetEntries()), desc="Processing .."):
        TreeReader.ReadEntry(entry)
        w_gen = xsec * Weight_branch.At(0).Weight / sum_w

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
            hs[ch][f"pt_l{i}_{ch}"].Fill(lep.PT, w_gen)
            
        hs[ch][f"lepPtSum_{ch}"].Fill(sum([lep[0].PT for lep in goodLeptons]), w_gen)
        hs[ch][f"m4l_{ch}"].Fill(sum([lep[0].P4() for lep in goodLeptons], ROOT.TLorentzVector()).M(), w_gen)

        # In 2e2mu channel, each Z is the one corresponding to a OSSF lepton pair
        # In the 4e and 4mu channels, we pick the opposite sign pair, whose invariant
        # mass is closer to mZ = 91.1 GeV.=
        ossf = [(goodLeptons[i][0].P4() + goodLeptons[j][0].P4(), frozenset((i, j)))
                for i, j in combinations(range(4), 2)
                if goodLeptons[i][1] == goodLeptons[j][1] and goodLeptons[i][0].Charge + goodLeptons[j][0].Charge == 0]
        Z1, Z2 = min((sorted((a, b), key=lambda z: abs(z.M() - 91.1))
                      for (a, ia), (b, ib) in combinations(ossf, 2) if ia.isdisjoint(ib)),
                     key=lambda zs: abs(zs[0].M() - 91.1))
        hs[ch][f"z1Mass_{ch}"].Fill(Z1.M(), w_gen)
        hs[ch][f"z2Mass_{ch}"].Fill(Z2.M(), w_gen)
        hs[ch][f"absDeltaPhiZ1Z2_{ch}"].Fill(abs(Z1.P4().DeltaPhi(Z2.P4())), w_gen)

    # # save to root file
    # for ch, hs_ch in hs.items():
    #     os.makedirs(f"./Validation/{ch}", exist_ok=True)
    #     f_out = ROOT.TFile(f"./Validation/{ch}/historgams.root", "RECREATE")
    #     for name, h in hs_ch.items():
    #         f_out.cd()
    #         h.Write(f"h_{name}")
    #     f_out.Close()

    return hs


def main():
    parser = argparse.ArgumentParser(description="Analyze Delphes ROOT files") 
    parser.add_argument("--lumi", type=float, default=10.0, help="Luminosity (fb^-1)")
    args = parser.parse_args()

    sqrt_s = 13.6

    processes = {
        "Signal": {
            "ggH_4L" : {
                "name"  : "ggH #rightarrow 4l",
                "files" : "./ggH_4L_LO/Events/test1/tag_1_delphes_events.root",
                "xsec_fb" : 0.0004303 * 1e3
            }
        },
        "Background": {
            "ggZZ_4L": {
                "name"  : "ggZZ #rightarrow 4l",
                "files" : "./ggZZ_4L_LO/Events/test1_decayed_1/tag_1_delphes_events.root",
                "xsec_fb"  : 1.0
            },
            "qqZZ_4L": {   
                "name"  : "q#bar{q}ZZ #rightarrow 4l",
                "files" : "./qqZZ_4L_LO/Events/test1/tag_1_delphes_events.root",
                "xsec_fb"  : 0.03468 * 1e3
            }
        }
    }

    signal = {}
    signal_name = ""
    backgrounds = []

    # Build the summed signal and the individual backgrounds
    for proc_catagory, proc_data in processes.items():
        if proc_catagory == "Signal":
            signal_name = ", ".join(proc["name"] for proc in proc_data.values())
            for proc in proc_data.values():
                hs = loop_tree(proc["files"], xsec=proc["xsec_fb"])
                for ch, hs_ch in hs.items():
                    for name, h in hs_ch.items():
                        h.Scale(args.lumi)
                        if (ch, name) in signal:
                            signal[(ch, name)].Add(h)
                        else:
                            signal[(ch, name)] = h.Clone(f"sig_{ch}_{name}")
        else:
            for proc in proc_data.values():
                hs = loop_tree(proc["files"], xsec=proc["xsec_fb"])
                for hs_ch in hs.values():
                    for h in hs_ch.values():
                        h.Scale(args.lumi)
                backgrounds.append((proc["name"], hs))

    # Plot each observable in each channel
    for (ch, name), sig in signal.items():
        os.makedirs(f"./Validation/{ch}", exist_ok=True)

        # Signal
        sig.SetLineColor(ROOT.kRed)
        sig.SetLineWidth(3)

        # Background stacked histogram
        bkg_refs = {}  # keep clones alive; THStack does not own them
        stack = ROOT.THStack(f"stack_{ch}_{name}", sig.GetTitle())
        for i, (bkg_name, hs_bkg) in enumerate(backgrounds):
            hb = hs_bkg[ch][name].Clone(f"bkg_{i}_{ch}_{name}")
            stack.Add(hb)
            bkg_refs[bkg_name] = hb

        # Legends
        legend = ROOT.TLegend(0.60, 0.70, 0.89, 0.88)
        legend.SetBorderSize(0)
        legend.AddEntry(sig, signal_name, "l")
        for bkg_name, hist_bkg in bkg_refs.items():
            legend.AddEntry(hist_bkg, bkg_name, "f")
        
        # Labels
        label = ROOT.TLatex()
        label.SetNDC(True)
        label.SetTextSize(0.035)

        # Draw
        c = ROOT.TCanvas("c", "c", 800, 600)
        c.cd()
        c.SetLogy(True)
        stack.SetMaximum(10 * max(stack.GetMaximum(), sig.GetMaximum()))
        stack.Draw("hist pfc plc")
        sig.Draw("hist same")
        legend.Draw()
        label.DrawLatex(0.2, 0.88, "#scale[1.1]{#bf{#sqrt{s} = %s TeV, %s fb^{-1}}}" % (sqrt_s, args.lumi))
        c.SaveAs(f"./Validation/{ch}/{name}.png")
        c.Close()


if __name__ == '__main__':
    main()
