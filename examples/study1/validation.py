import os
import ROOT
import numpy as np
import atlasplots as aplt
from pathlib import Path
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

sqrt_s = 13.0
lumi = 36.1  # fb^-1, ATLAS Open Data 2015+2016: the same lumi for data and MC

# ATLAS Open Data 2015+2016, exactly4lep skim (record 93924)
OPEN_DATA = "root://eospublic.cern.ch//eos/opendata/atlas/rucio/opendata/ODEO_FEB2025_v0_exactly4lep_"

channels = ["4e", "4mu", "2e2mu"]
branches = [f"{k}_lep{i}" for k in ["Pt", "Phi", "Eta", "E"] for i in range(1, 5)]
branches+= ["lepPtSum", "m4l", "z1Mass", "z2Mass", "absDeltaPhiZ1Z2", "weight"]

channel_names = {
    "4e" : "e^{+}e^{-}e^{+}e^{-}",
    "4mu" : "#mu^{+}#mu^{-}#mu^{+}#mu^{-}",
    "2e2mu" : "e^{+}e^{-}#mu^{+}#mu^{-}",
}

processes = {
    "Signal": {
        "ggH_4e" : {
            "name"  : "ggH #rightarrow 4e",
            "files" : "./ggH_4e_LO/Events/test1/tag_1_delphes_events.root",
            "xsec_fb" : 0.0001048 * 1e3
        },
        "ggH_4mu" : {
            "name"  : "ggH #rightarrow 4#mu",
            "files" : "./ggH_4mu_LO/Events/test1/tag_1_delphes_events.root",
            "xsec_fb" : 0.0001048 * 1e3
        },
        "ggH_2e2mu" : {
            "name"  : "ggH #rightarrow ee#mu#mu",
            "files" : "./ggH_2e2mu_LO/Events/test1/tag_1_delphes_events.root",
            "xsec_fb" : 0.0001837 * 1e3
        }
    },
    "Background": {
        "ggZZ_4e": {
            "name"  : "ggZZ #rightarrow 4e",
            "files" : "./ggZZ_4e_LO/Events/test1_decayed_1/tag_1_delphes_events.root",
            "xsec_fb"  : 0.001187 * 1e3
        },
        "ggZZ_4mu": {
            "name"  : "ggZZ #rightarrow 4#mu",
            "files" : "./ggZZ_4mu_LO/Events/test1_decayed_1/tag_1_delphes_events.root",
            "xsec_fb"  : 0.001187 * 1e3
        },
        "ggZZ_2e2mu" : {
            "name"  : "ggZZ #rightarrow ee#mu#mu",
            "files" : "./ggZZ_2e2mu_LO/Events/test1_decayed_1/tag_1_delphes_events.root",
            "xsec_fb" : 0.002374 * 1e3
        },
        "qqZZ_4e" : {
            "name"  : "qqZZ #rightarrow 4e",
            "files" : "./qqZZ_4e_LO/Events/test1/tag_1_delphes_events.root",
            "xsec_fb" : 0.008266 * 1e3
        },
        "qqZZ_4mu" : {
            "name"  : "qqZZ #rightarrow 4#mu",
            "files" : "./qqZZ_4mu_LO/Events/test1/tag_1_delphes_events.root",
            "xsec_fb" : 0.008277 * 1e3
        },
        "qqZZ_2e2mu" : {
            "name"  : "qqZZ #rightarrow ee#mu#mu",
            "files" : "./qqZZ_2e2mu_LO/Events/test1/tag_1_delphes_events.root",
            "xsec_fb" : 0.01616 * 1e3
        }
    },
    "Data": {
        "exactly4lep" : {
            "name"  : "Data",
            "files" : [OPEN_DATA + p + ".exactly4lep.root"
                       for p in ("data15_periodD", "data15_periodE", "data15_periodF",
                                 "data15_periodG", "data15_periodH", "data15_periodJ",
                                 "data16_periodA", "data16_periodB", "data16_periodC",
                                 "data16_periodD", "data16_periodE", "data16_periodF",
                                 "data16_periodG", "data16_periodI", "data16_periodK",
                                 "data16_periodL")]
        }
    }
}


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

def book_tree(tree_name, float_branch_names=None, int_branch_names=None):
    tree = ROOT.TTree(tree_name, tree_name)
    tree.SetDirectory(0)

    b = {}
    for branch_name in int_branch_names or ():
        b[branch_name] = np.zeros(1, dtype=np.int32)
        tree.Branch(branch_name, b[branch_name], f"{branch_name}/I")  # /I for Integer
    for branch_name in float_branch_names or ():
        b[branch_name] = np.zeros(1, dtype=np.float64)
        tree.Branch(branch_name, b[branch_name], f"{branch_name}/D")  # /D for Double (64-bit float)

    return tree, b

def write_tree(tree, out):
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    f_out = ROOT.TFile(str(out), "RECREATE")
    tree.SetDirectory(f_out)
    tree.Write()
    f_out.Close()
    return None


def loop_real_data(inputRootFile, trees, branches):
    """Record 93924 (ATLAS 2025 O&E skim, tree "analysis"): RVec branches, GeV units.
    Data carry no generator weight, so every event counts one (weight = 1) and the
    normalisation to the MC luminosity is done at plotting time (lumi / lumi_data).
    """
    files = inputRootFile if isinstance(inputRootFile, list) else [inputRootFile]
    Chain = ROOT.TChain("analysis")
    for f in files:
        Chain.Add(f)  # root:// URLs go in verbatim; Path().resolve() would break them
    Chain.SetBranchStatus("*", 0)  # 117 branches in the file, we touch 8

    pt, eta, phi, energy = (ROOT.ROOT.VecOps.RVec("float")(), ROOT.ROOT.VecOps.RVec("float")(),
                            ROOT.ROOT.VecOps.RVec("float")(), ROOT.ROOT.VecOps.RVec("float")())
    charge   = ROOT.ROOT.VecOps.RVec("int")()
    lep_type = ROOT.ROOT.VecOps.RVec("int")()
    loose_id  = ROOT.ROOT.VecOps.RVec("bool")()
    loose_iso = ROOT.ROOT.VecOps.RVec("bool")()
    for name, buf in (("lep_pt", pt), ("lep_eta", eta), ("lep_phi", phi), ("lep_e", energy),
                      ("lep_charge", charge), ("lep_type", lep_type),
                      ("lep_isLooseID", loose_id), ("lep_isLooseIso", loose_iso)):
        Chain.SetBranchStatus(name, 1)
        Chain.SetBranchAddress(name, buf)

    for entry in tqdm(range(Chain.GetEntries()), desc="Processing data .."):
        Chain.GetEntry(entry)

        # Object selection
        # -----------------------
        # Same kinematics as loop_mc_data, plus the skim's own loose ID/isolation flags:
        # this skim keeps every 4-lepton event down to 7 GeV, most of them fake.
        goodLeptons = []
        for i in range(pt.size()):
            if abs(eta[i]) > 2.7 or pt[i] <= 10.0:
                continue
            if not (loose_id[i] and loose_iso[i]):
                continue
            fl = {11: "e", 13: "mu"}.get(int(lep_type[i]))
            if fl is None:
                continue
            p4 = ROOT.TLorentzVector()
            p4.SetPtEtaPhiE(pt[i], eta[i], phi[i], energy[i])
            goodLeptons.append((p4, fl, int(charge[i])))
        goodLeptons.sort(key=lambda lfc: lfc[0].Pt(), reverse=True)
        if len(goodLeptons) < 4:
            continue
        goodLeptons = goodLeptons[:4]

        # Event selection
        # -----------------------
        # Same OSSF-pair counting as loop_mc_data: (2,0)=4e, (0,2)=4mu, (1,1)=2e2mu.
        charge_counts = Counter((fl, 1 if q > 0 else -1) for _, fl, q in goodLeptons)
        e_pairs  = min(charge_counts[("e", 1)],  charge_counts[("e", -1)])
        mu_pairs = min(charge_counts[("mu", 1)], charge_counts[("mu", -1)])
        ch = {(2, 0): "4e", (0, 2): "4mu", (1, 1): "2e2mu"}.get((e_pairs, mu_pairs))
        if ch is None:
            continue

        # Fill trees
        # -----------------------
        tree, b = trees[ch], branches[ch]
        b["weight"][0] = 1.0  # raw yield, no generator weight in data

        for i, (p4, _, _) in enumerate(goodLeptons, start=1):
            b[f"Pt_lep{i}"][0]  = p4.Pt()
            b[f"Phi_lep{i}"][0] = p4.Phi()
            b[f"Eta_lep{i}"][0] = p4.Eta()
            b[f"E_lep{i}"][0]   = p4.Energy()

        # driven features
        b["lepPtSum"][0] = sum(p4.Pt() for p4, _, _ in goodLeptons)
        b["m4l"][0] = sum((p4 for p4, _, _ in goodLeptons), ROOT.TLorentzVector()).M()

        ossf = [(goodLeptons[j][0] + goodLeptons[k][0], frozenset((j, k)))
                for j, k in combinations(range(4), 2)
                if goodLeptons[j][1] == goodLeptons[k][1]
                and goodLeptons[j][2] + goodLeptons[k][2] == 0]

        Z1, Z2 = min((sorted((z1, z2), key=lambda z: abs(z.M() - 91.1))
                      for (z1, iz1), (z2, iz2) in combinations(ossf, 2) if iz1.isdisjoint(iz2)),
                     key=lambda zs: abs(zs[0].M() - 91.1))
        b["z1Mass"][0] = Z1.M()
        b["z2Mass"][0] = Z2.M()
        b["absDeltaPhiZ1Z2"][0] = abs(Z1.P4().DeltaPhi(Z2.P4()))

        tree.Fill()

    return trees

def loop_mc_data(inputRootFile, trees, branches, xsec=1.0):
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

        # Fill trees
        # -----------------------
        tree, b = trees[ch], branches[ch]
        b["weight"][0] = w_gen

        # raw data 
        for i, (lep, _) in enumerate(goodLeptons, start=1):
                b[f"Pt_lep{i}"][0] = lep.PT
                b[f"Phi_lep{i}"][0] = lep.Phi
                b[f"Eta_lep{i}"][0] = lep.Eta
                b[f"E_lep{i}"][0] = lep.P4().Energy()
        
        # driven features
        b["lepPtSum"][0] = sum(lep.PT for lep, _ in goodLeptons)
        b["m4l"][0] = sum((lep.P4() for lep, _ in goodLeptons), ROOT.TLorentzVector()).M()


        # In 2e2mu channel, each Z is the one corresponding to a OSSF lepton pair
        # In the 4e and 4mu channels, we pick the opposite sign pair, whose invariant
        # mass is closer to mZ = 91.1 GeV.=
        ossf = [(goodLeptons[j][0].P4() + goodLeptons[k][0].P4(), frozenset((j, k)))
                for j, k in combinations(range(4), 2)
                if goodLeptons[j][1] == goodLeptons[k][1]
                and goodLeptons[j][0].Charge + goodLeptons[k][0].Charge == 0]

        Z1, Z2 = min((sorted((z1, z2), key=lambda z: abs(z.M() - 91.1))
                      for (z1, iz1), (z2, iz2) in combinations(ossf, 2) if iz1.isdisjoint(iz2)),
                     key=lambda zs: abs(zs[0].M() - 91.1))
        b["z1Mass"][0] = Z1.M()
        b["z2Mass"][0] = Z2.M()
        b["absDeltaPhiZ1Z2"][0] = abs(Z1.P4().DeltaPhi(Z2.P4()))

        tree.Fill()
    
    return trees

def draw_histograms():
    aplt.set_atlas_style()
    out = Path("./Validation/histograms").resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out.parent / "data"  # ./Validation/data/{ch}/, written by main

    def get_hist(path, branch, nbins, lo, hi):
        """Weighted histogram from one tree; None if the file is missing."""
        if not path.exists():
            return None
        stem = path.stem  # file name == tree name, e.g. sig_ggH_4e.root -> sig_ggH_4e
        return ROOT.RDataFrame(stem, str(path)).Histo1D(
            (f"h_{stem}_{branch}", "", nbins, lo, hi), branch, "weight").GetValue()

    observables = {
        "m4l":             (50,  60.0, 660.0, "m_{4l} [GeV]"),
        "z1Mass":          (50,  40.0, 200.0, "m_{Z_{1}} [GeV]"),
        "z2Mass":          (50,  10.0, 200.0, "m_{Z_{2}} [GeV]"),
        "lepPtSum":        (50,   0.0, 800.0, "#Sigma |p_{T}^{lep}| [GeV]"),
        "absDeltaPhiZ1Z2": (32,   0.0,   3.2, "|#Delta#phi(Z_{1},Z_{2})|"),
    }
    for lep in range(1, 5):
        observables[f"Pt_lep{lep}"]  = (50, 0.0, 400.0, "p_{T}^{lep%d} [GeV]" % lep)
        observables[f"Phi_lep{lep}"] = (32, -np.pi, np.pi, "#phi^{lep%d}" % lep)
        observables[f"Eta_lep{lep}"] = (50, -2.7, 2.7, "#eta^{lep%d}" % lep)
        observables[f"E_lep{lep}"]   = (50, 0.0, 500.0, "E^{lep%d} [GeV]" % lep)

    for ch in channels:
        (out / ch).mkdir(parents=True, exist_ok=True)

        for branch in branches:
            if branch == "weight" or branch not in observables:
                continue
            nb, lo, hi, xtitle = observables[branch]
            ytitle = ("Events / %.0f GeV" % ((hi - lo) / nb)) if "[GeV]" in xtitle else "Events / bin"

            sig = get_hist(data_dir / ch / f"sig_ggH_{ch}.root", branch, nb, lo, hi)
            if sig is None or sig.GetEntries() == 0:
                continue
            sig.Scale(lumi)
            sig.SetLineColor(ROOT.kRed)
            sig.SetLineWidth(3)

            bkg_refs = []  # THStack keeps only pointers
            stack = ROOT.THStack(f"stack_{ch}_{branch}", "")
            bkg_total = ROOT.TH1D(f"bkg_total_{ch}_{branch}", "", nb, lo, hi)
            bkg_total.Sumw2()
            for proc in ("ggZZ", "qqZZ"):
                h = get_hist(data_dir / ch / f"bkg_{proc}_{ch}.root", branch, nb, lo, hi)
                if h is None or h.GetEntries() == 0:
                    continue
                h.Scale(lumi)
                stack.Add(h)  # pfc/plc colour the fills
                bkg_total.Add(h)
                bkg_refs.append((proc, h))

            dst = None  # data are raw counts at lumi = 36.1 fb^-1, so no Scale here
            for key in processes["Data"]:
                dst = get_hist(data_dir / ch / f"data_{key}.root", branch, nb, lo, hi)
                if dst is not None and dst.GetEntries() > 0:
                    break
                dst = None
            if dst is None:
                continue

            # MC total = backgrounds + signal, used only for the uncertainty band
            mc = bkg_total.Clone(f"mc_total_{ch}_{branch}")
            mc.Sumw2()
            mc.Add(sig)

            # ATLAS Plots recipe (examples/data_vs_mc.py)
            band = aplt.root_helpers.hist_to_graph(mc, show_bin_width=True)
            band_ratio = aplt.root_helpers.hist_to_graph(mc, show_bin_width=True, norm=True)
            dst_graph = aplt.root_helpers.hist_to_graph(dst)
            ratio_hist = dst.Clone(f"ratio_{ch}_{branch}")
            ratio_hist.Divide(mc)  # y err = ratio*sqrt((e_d/d)^2 + (e_mc/mc)^2)
            # hist_to_graph equivalent, but bins with data or MC = 0 are skipped:
            # Divide puts them at 0, which would draw a false row of points at y = 0
            ratio_graph = ROOT.TGraphAsymmErrors()
            k = 0
            for ib in range(1, nb + 1):
                if mc.GetBinContent(ib) <= 0.0 or dst.GetBinContent(ib) <= 0.0:
                    continue
                ratio_graph.SetPoint(k, ratio_hist.GetBinCenter(ib), ratio_hist.GetBinContent(ib))
                ratio_graph.SetPointError(k, 0.0, 0.0,
                                          ratio_hist.GetBinError(ib), ratio_hist.GetBinError(ib))
                k += 1
            ratio_graph.SetMarkerStyle(ROOT.kFullCircle)
            ratio_graph.SetMarkerSize(1.2)
            ratio_graph.SetLineColor(ROOT.kBlack)

            # signal as a step line: a TH1 in log-y drags a red line along the bottom
            # axis on every empty bin, a graph is just clipped
            sig_line = ROOT.TGraph(2 * nb)
            for ib in range(1, nb + 1):
                sig_line.SetPoint(2 * (ib - 1), sig.GetBinLowEdge(ib), sig.GetBinContent(ib))
                sig_line.SetPoint(2 * (ib - 1) + 1, sig.GetBinLowEdge(ib) + sig.GetBinWidth(ib), sig.GetBinContent(ib))
            sig_line.SetLineColor(ROOT.kRed)
            sig_line.SetLineWidth(3)

            fig, (ax1, ax2) = aplt.ratio_plot(name=f"fig_{ch}_{branch}", figsize=(800, 800), hspace=0.05)

            ax1.plot(stack, " pfc plc")
            ax1.plot(band, "2", fillcolor=1, fillstyle=3254, linewidth=0)
            ax1.plot(sig_line, "L")
            ax1.plot(dst_graph, "P")

            ax1.set_xlim(lo, hi)
            ax1.set_yscale("log")
            ymax = max(mc.GetMaximum(), sig.GetMaximum(), dst.GetMaximum())
            ax1.set_ylim(1e-2, ymax * 1e2 if ymax > 0 else 10.0)
            ax1.set_ylabel(ytitle)
            ax1.add_margins(top=0.16)

            # ratio panel: data/MC points (data (+) MC stat) and the MC stat band at y=1
            ax2.plot(band_ratio, "2", fillcolor=1, fillstyle=3254, linewidth=0)
            ax2.plot(ratio_graph, "P0")
            ax2.plot(ROOT.TLine(lo, 1.0, hi, 1.0), linestyle=2, linecolor=ROOT.kGray + 1, expand=False)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(0.0, 5.0)
            ax2.set_xlabel(xtitle)
            ax2.set_ylabel("Data / MC", loc="centre")

            ax1.cd()
            ax1.text(0.2, 0.90, "#bf{#sqrt{s} = 13 TeV, %s fb^{-1}}" % lumi, size=25, align=13)
            ax1.text(0.2, 0.83, channel_names[ch], size=22, align=13)

            ax1._legend_entries = []
            ax1._legend_entries.append((dst_graph, "Data", "EP"))
            ax1._legend_entries.append((sig_line, "ggH", "L"))
            for proc, h in bkg_refs:
                ax1._legend_entries.append((h, proc, "F"))
            ax1._legend_entries.append((band, "MC Stat. Unc.", "F"))
            # legend placement exactly as in the ATLAS Plots Data/MC example
            ax1.legend(
                loc=(0.68, 0.65, 1 - ROOT.gPad.GetRightMargin() - 0.03,
                     1 - ROOT.gPad.GetTopMargin() - 0.03),
                textsize=22,
            )

            fig.savefig(str(out / ch / f"{branch}.png"))


def main():

    # for cat, procs in processes.items():
    #     tag = {"Signal": "sig", "Background": "bkg", "Data": "data"}[cat]
    #     for key, proc in procs.items():
    #         trees, buffers = {}, {}
    #         for ch in channels:
    #             trees[ch], buffers[ch] = book_tree(f"{tag}_{key}", float_branch_names=branches)
    #         if cat == "Data":
    #             loop_real_data(proc["files"], trees, buffers)
    #         else:
    #             loop_mc_data(proc["files"], trees, buffers, proc["xsec_fb"])
    #         for ch, tree in trees.items():
    #             write_tree(tree, out=f"./Validation/data/{ch}/{tag}_{key}.root")

    draw_histograms()

if __name__ == '__main__':
    main()
