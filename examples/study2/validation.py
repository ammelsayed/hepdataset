import os
import ROOT
import argparse
from tqdm import tqdm
from object_selection import ObjectSelector

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

def book_histograms(nbins=30, xmin=10.0, xmax=2500, nb_lep=2, nb_fj=2):
    binwidth = abs(xmax-xmin)/nbins
    h_dict = {      
        "met": ROOT.TH1D("h_met", f"MET; MET [GeV]; Weighted Events / {binwidth:.2f} [GeV]", nbins, xmin, xmax),
        "ht": ROOT.TH1D("h_ht", f"HT; HT [GeV]; Weighted Events / {binwidth:.2f} [GeV]", nbins, xmin, xmax),
    }
    for i in range(1, nb_lep + 1):
        name = f"l{i}Pt"
        h_dict[name] = ROOT.TH1D(name, f"; p_{{T}} (l_{{{i}}}) [GeV]; Weighted Events / {binwidth:.2f} [GeV]", nbins, xmin, xmax)
    for i in range(1, nb_fj + 1):
        name = f"fj{i}Pt"
        h_dict[name] = ROOT.TH1D(name, f"; p_{{T}} (J_{{i}}) [GeV]; Weighted Events / {binwidth:.2f} [GeV]", nbins, xmin, xmax)

    for val in h_dict.values():
        val.Sumw2()
    return h_dict

def loop_tree(inputRootFile, xsecXlumi=1.0, nb_lep=2, nb_fj=2):
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

    hs = book_histograms(nb_lep=nb_lep, nb_fj=nb_fj)
    objSel = ObjectSelector()

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
        selected_objects = objSel.Select(Muon_branch, Electron_branch, FatJet_branch, Jet_branch, event_weight = w_gen)
        goodFatJets = selected_objects["goodFatJets"]
        goodLeptons = selected_objects["goodLeptons"]

        # Fill histograms
        for i in range(min(nb_lep, len(goodLeptons))):
            hs[f"l{i+1}Pt"].Fill(goodLeptons[i].PT, w_gen)

        for i in range(min(nb_fj, len(goodFatJets))):
            hs[f"fj{i+1}Pt"].Fill(goodFatJets[i].PT, w_gen)

        met = MissingET_branch.At(0).MET
        hs["met"].Fill(met, w_gen)

        ht = ScalarHT_branch.At(0).HT
        hs["ht"].Fill(ht, w_gen)
    
    # save to root file
    f_out = ROOT.TFile("./historgams.root", "RECREATE")
    for name, h in hs.items():
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
    parser.add_argument("root_files", nargs="+", help="Input ROOT file(s)")
    parser.add_argument("--xsec", type=float, default=None, help="Cross section (pb)")
    parser.add_argument("--lumi", type=float, default=500.0, help="Luminosity (fb^-1)")
    parser.add_argument("--nb-lep", type=int, default=2, help="Number of leptons to draw")
    parser.add_argument("--nb-fj", type=int, default=2, help="Number of fatjets to draw")
    parser.add_argument("--draw", action="store_true", help="Draw histograms")
    args = parser.parse_args()

    for f in args.root_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Input file does not exist: {f}")

    if args.xsec is not None:
        xsecXlumi = args.xsec * 1e3 * args.lumi
    else:
        xsecXlumi = None
        
    hs = loop_tree(args.root_files, xsecXlumi=xsecXlumi, nb_lep=args.nb_lep, nb_fj=args.nb_fj)

    if args.draw:
        for name, h in hs.items():
            draw(h, name=name)


if __name__ == '__main__':
    main()