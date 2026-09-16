import os
import ROOT
import pandas as pd
from math import fabs
from itertools import combinations, product
from tabulate import tabulate
from kinematics import Tau21, Tau32

def MergeObjectSelectors():
    """
    For each object selector object,
    read its cutflow and histograms, then,
    merge the cutflows
    merge the histograms
    this can be a class, with the PrintObjectSelectionSummary and DrawObjectSelectionHistograms methods (a child class of ObjectSelector with only two methods)
    """
    pass

class ObjectSelector:
    """Encapsulates the cutflow dict and object-selection histograms."""

    def __init__(self):
        self.cutflow = {
            "lepton" : {"initial" : 0},
            "fatjet" : {"initial" : 0}
        }
        self.hist = self.BookObjectSelectionHistograms()

    def BookObjectSelectionHistograms(self):
        h_dict = {
            "h_nL": ROOT.TH1D("h_nL", "; Lepton Multiplicity; Weighted Events", 6, -0.5, 5.5),
            "h_nL_raw" : ROOT.TH1D("h_nL_raw", "; Lepton Multiplicity; Weighted Events", 6, -0.5, 5.5),
            "h_nL_baseline": ROOT.TH1D("h_nL_baseline", "; Lepton Multiplicity; Events", 6, -0.5, 5.5),
            "h_nL_pass_isolation": ROOT.TH1D("h_nL_pass_isolation", "; Lepton Multiplicity; Events", 6, -0.5, 5.5),
            "h_nL_pass_em_olr": ROOT.TH1D("h_nL_pass_em_olr", "; Lepton Multiplicity; Events", 6, -0.5, 5.5),
            "h_nL_pass_fj_olr": ROOT.TH1D("h_nL_pass_fj_olr", "; Lepton Multiplicity; Events", 6, -0.5, 5.5),
            "h_nL_pass_pTCut": ROOT.TH1D("h_nL_pass_pTCut", "; Lepton Multiplicity; Events", 6, -0.5, 5.5),
            
            "h_nJ": ROOT.TH1D("h_nJ", " FatJet Multiplicity; Weighted Events", 11, -0.5, 10.5),
            "h_nJ_raw" : ROOT.TH1D("h_nJ_raw", "; FatJet Multiplicity; Weighted Events", 11, -0.5, 10.5),
            "h_nJ_baseline": ROOT.TH1D("h_nJ_baseline", "; FatJet Multiplicity; Events", 11, -0.5, 10.5),
            "h_nJ_pass_massCut": ROOT.TH1D("h_nJ_pass_massCut", "; FatJet Multiplicity; Events", 11, -0.5, 10.5),
            "h_nJ_pass_tau21Cut": ROOT.TH1D("h_nJ_pass_tau21Cut", "; FatJet Multiplicity; Events", 11, -0.5, 10.5),
            "h_nJ_pass_pTCut": ROOT.TH1D("h_nJ_pass_pTCut", "; FatJet Multiplicity; Events", 11, -0.5, 10.5),

            "h_nj": ROOT.TH1D("h_nj", " Jet Multiplicity; Weighted Events", 16, -0.5, 15.5),
            "h_nj_raw" : ROOT.TH1D("h_nj_raw", "; Jet Multiplicity; Weighted Events", 16, -0.5, 15.5),
        }

        for val in h_dict.values():
            val.Sumw2()
            val.SetDirectory(0)
                
        return h_dict

    def _fill(self, hist_name, value, event_weight = 1.0):
        self.hist[hist_name].Fill(value, event_weight)
        return None

    def PrintObjectSelectionSummary(self, lum = None, event_weight = None, merge=False):

        for obj, obj_cf in self.cutflow.items():
            total_events = obj_cf["initial"]
            cf = pd.DataFrame(obj_cf.items(), columns=["Stage", "Count"])

            if (event_weight != None) and (lum != None):
                cf[f"Yield ({int(lum)} fb^-1)"] = cf["Count"] * event_weight

            eff = [0.0] + [
                obj_cf[stage] / obj_cf[prev_stage] if obj_cf[prev_stage] > 0 else 0.0
                for stage, prev_stage in zip(list(obj_cf.keys())[1:], list(obj_cf.keys())[:-1])
            ]
            cf["Efficiency"] = eff
            cf["Cumulative_Efficiency"] = cf["Count"] / total_events

            # Convert the two efficiency columns to percentage strings
            cf["Efficiency"] = cf["Efficiency"].map(lambda x: f"{x*100:.2f}%")
            cf["Cumulative_Efficiency"] = cf["Cumulative_Efficiency"].map(lambda x: f"{x*100:.2f}%")

            print(f"\n*** {obj.title()} Selection Cutflow ***")
            print(tabulate(
                cf,
                headers='keys',
                tablefmt='simple',
                showindex=False,
                colalign=(("left",) + ("center",) * (len(cf.columns) - 1)),
            ))

        return None

    def Select(self, Muon_branch, Electron_branch, FatJet_branch, Jet_branch, event_weight = 1.0):

        cf = self.cutflow
        cf["lepton"]["initial"] += 1
        cf["fatjet"]["initial"] += 1
        self._fill("h_nL_raw", Muon_branch.GetEntries() + Electron_branch.GetEntries(), event_weight)
        self._fill("h_nJ_raw", FatJet_branch.GetEntries(), event_weight)
        self._fill("h_nj_raw", Jet_branch.GetEntries(), event_weight)

        # List of lists (or dict) to hold sequentially selected objects
        lep_stages = []
        fj_stages = []

        ## Lepton selections
        ## ---------------------

        # Preselection (baseline)
        leps = []
        # Muons
        for i in range(Muon_branch.GetEntries()):
            lep = Muon_branch.At(i)
            if fabs(lep.Eta) <= 2.7 and lep.PT > 10.0:
                leps.append(lep)
        # Electrons
        for i in range(Electron_branch.GetEntries()):
            lep = Electron_branch.At(i)
            if fabs(lep.Eta) <= 2.7 and lep.PT > 10.0:
                leps.append(lep)
        # Hadronic tau 
        # (not considered in this analysis, but can be added here if needed)
        leps.sort(key=lambda lep: lep.PT, reverse=True)
        lep_stages.append(leps)
        self._fill("h_nL_baseline", len(leps), event_weight)
        if len(leps) > 0: 
            cf["lepton"]["baseline"] = cf["lepton"].get("baseline", 0) + 1

        # Isolation requirements
        leps = []
        for lep in lep_stages[-1]:
            if lep.ClassName() == "Muon" and lep.IsolationVar < 0.15:
                leps.append(lep)
            elif lep.ClassName() == "Electron" and lep.IsolationVar < 0.20:
                leps.append(lep)
        leps.sort(key=lambda lep: lep.PT, reverse=True)
        lep_stages.append(leps)
        self._fill("h_nL_pass_isolation", len(leps), event_weight)
        if len(leps) > 0: 
            cf["lepton"]["pass_isolation"] = cf["lepton"].get("pass_isolation", 0) + 1

        # Electron-Muon Overlap Removal
        leps = lep_stages[-1][:]
        indices_to_remove = set()
        for i, j in combinations(range(len(leps)), 2):
            lep_i, lep_j = leps[i], leps[j]
            if lep_i.ClassName() != lep_j.ClassName():
                dr = lep_i.P4().DeltaR(lep_j.P4())
                if dr < 0.1:
                    indices_to_remove.add(j if lep_i.PT > lep_j.PT else i)
        leps = [lep for idx, lep in enumerate(leps) if idx not in indices_to_remove]
        leps.sort(key=lambda lep: lep.PT, reverse=True)
        lep_stages.append(leps)
        self._fill("h_nL_pass_em_olr", len(leps), event_weight)
        if len(leps) > 0: 
            cf["lepton"]["pass_em_olr"] = cf["lepton"].get("pass_em_olr", 0) + 1

        ## Ak10 Jet selections
        ## ---------------------

        # Preselection (baseline)
        fjs = []
        for i in range(FatJet_branch.GetEntries()):
            fj = FatJet_branch.At(i)
            if fabs(fj.Eta) <= 2.5 and fj.PT > 30.0:
                fjs.append(fj)
        fjs.sort(key=lambda fj: fj.PT, reverse=True)
        fj_stages.append(fjs)
        self._fill("h_nJ_baseline", len(fjs), event_weight)
        if len(fjs) > 0: 
            cf["fatjet"]["baseline"] = cf["fatjet"].get("baseline", 0) + 1

        # Mass window selections
        fjs = []
        mWin, mLow, mHigh = 20.0, 80.0, 125.0
        for fj in fj_stages[-1]:
            massOk = (mLow - mWin <= fj.SoftDroppedP4[0].M() <= mHigh + mWin)
            if massOk:
                fjs.append(fj)
        fjs.sort(key=lambda fj: fj.PT, reverse=True)
        fj_stages.append(fjs)
        self._fill("h_nJ_pass_massCut", len(fjs), event_weight)
        if len(fjs) > 0: 
            cf["fatjet"]["pass_massCut"] = cf["fatjet"].get("pass_massCut", 0) + 1

        # N-subjettiness selections
        # (This part can be updated to use MVA methods or other advanced techniques for better discrimination)
        fjs = []
        tau21_cut = 0.7 
        for fj in fj_stages[-1]:
            if Tau21(fj) < tau21_cut:
                fjs.append(fj)
        fjs.sort(key=lambda fj: fj.PT, reverse=True)
        fj_stages.append(fjs)
        self._fill("h_nJ_pass_tau21Cut", len(fjs), event_weight)
        if len(fjs) > 0: 
            cf["fatjet"]["pass_tau21Cut"] = cf["fatjet"].get("pass_tau21Cut", 0) + 1

        ## Cross-object Overlap Removal
        ## -----------------------------

        # Lepton-FatJet Overlap Removal
        leps = lep_stages[-1][:]
        fjs  = fj_stages[-1]
        lep_indices_to_remove = set()
        for i, j in product(range(len(leps)), range(len(fjs))):
            lep, fj = leps[i], fjs[j]
            dr = lep.P4().DeltaR(fj.P4())
            if dr < 1.0:
                lep_indices_to_remove.add(i)
        leps = [lep for idx, lep in enumerate(leps) if idx not in lep_indices_to_remove]
        leps.sort(key=lambda lep: lep.PT, reverse=True)
        lep_stages.append(leps)
        self._fill("h_nL_pass_fj_olr", len(leps), event_weight)
        if len(leps) > 0: 
            cf["lepton"]["pass_fj_olr"] = cf["lepton"].get("pass_fj_olr", 0) + 1

        ## Final Tight Selections
        ## -----------------------

        # Apply transverse momentum requirements to leptons
        leps = []
        pt_cuts = {"Muon" : {0: 25.0, 1: 20.0, 2: 10.0}, "Electron" : {0: 27.0, 1: 20.0, 2: 10.0}}
        # Explicitly sort by pT to guarantee correct leading/sub-leading ordering, 
        # removing any dependency on previous stages.
        current_leps = sorted(lep_stages[-1], key=lambda l: l.PT, reverse=True)
        for lep in current_leps:
            # Dynamically determine the index based on the leptons that have already passed
            idx = len(leps)
            min_pt = pt_cuts[lep.ClassName()].get(idx, 10.0)
            if lep.PT > min_pt:
                leps.append(lep)
            # If it fails, we do nothing and just move to the next lepton.
            # The next lepton will then be tested against the SAME rank threshold.
        # No need to sort again, as we appended them in descending pT order
        lep_stages.append(leps)
        self._fill("h_nL_pass_pTCut", len(leps), event_weight)
        if len(leps) > 0: 
            cf["lepton"]["pass_pTCut"] = cf["lepton"].get("pass_pTCut", 0) + 1

        # Apply transverse momentum requirements to fatjets
        fjs = []
        fj_pt_cuts = {0: 200.0, 1: 150.0, 2: 30.0}
        current_fjs = sorted(fj_stages[-1], key=lambda f: f.PT, reverse=True)
        for idx, fj in enumerate(current_fjs):
            idx = len(fjs)
            min_pt = fj_pt_cuts.get(idx, 30.0)
            if fj.PT > min_pt:
                fjs.append(fj)
        fj_stages.append(fjs)
        self._fill("h_nJ_pass_pTCut", len(fjs), event_weight)
        if len(fjs) > 0: 
            cf["fatjet"]["pass_pTCut"] = cf["fatjet"].get("pass_pTCut", 0) + 1

        # Extract final lists
        goodLeptons = lep_stages[-1]
        goodFatJets = fj_stages[-1]

        goodElectrons, goodMuons = [], []
        # goodElectrons = [lep for lep in goodLeptons if lep.ClassName() == "Electron"]
        # goodMuons = [lep for lep in goodLeptons if lep.ClassName() == "Muon"]

        goodWFatJets, goodZFatjets, goodHFatjets, goodTopFatjets, goodEWFatJets = [], [], [], [], []
        goodJets, goodBJets, goodCJets, goodSJets, goodTauJets = [], [], [], [], []


        return {
            "goodFatJets": goodFatJets,
            "goodWFatJets": goodWFatJets,
            "goodZFatjets": goodZFatjets,
            "goodHFatjets": goodHFatjets,
            "goodTopFatjets": goodTopFatjets,
            "goodEWFatJets": goodEWFatJets,

            "goodLeptons": goodLeptons,
            "goodElectrons": goodElectrons,
            "goodMuons": goodMuons,
            
            "goodJets": goodJets,
            "goodBJets": goodBJets,
            "goodCJets": goodCJets,
            "goodSJets": goodSJets,
            "goodTauJets": goodTauJets
        }

    def DrawObjectSelectionHistograms(self, 
        output_dir = ".", 
        formats = ["png"], 
        cH = 800, cW = 700
        ):
        
        ROOT.gROOT.SetBatch(True)
        ROOT.gROOT.SetStyle("ATLAS")
        ROOT.gStyle.SetPalette(ROOT.kRainbow)
        os.makedirs(output_dir, exist_ok=True)

        # Lepton multiplicty
        c1 = ROOT.TCanvas("c1", "Lepton Multiplicity", cH, cW)
        c1.SetLogy()
        yI = 0.9
        legend1 = ROOT.TLegend(0.55, yI-0.35, 0.92, yI)
        legend1.SetFillColor(0)
        legend1.SetBorderSize(0)
        legend1.SetTextSize(0.035)
        yMax = max(
            self.hist["h_nL_raw"].GetMaximum(), 
            self.hist["h_nL_baseline"].GetMaximum(), 
            self.hist["h_nL_pass_isolation"].GetMaximum(), 
            self.hist["h_nL_pass_em_olr"].GetMaximum(), 
            self.hist["h_nL_pass_fj_olr"].GetMaximum(), 
            self.hist["h_nL_pass_pTCut"].GetMaximum()
        )
        if self.hist["h_nL_raw"].Integral() > 0:
            self.hist["h_nL_raw"].SetMaximum(yMax * 1e5)
            self.hist["h_nL_raw"].SetLineColor(ROOT.kRed)
            self.hist["h_nL_raw"].SetLineWidth(3)
            self.hist["h_nL_raw"].SetLineStyle(2)
            self.hist["h_nL_raw"].Draw("HIST")
            legend1.AddEntry(self.hist["h_nL_raw"], "Raw", "l")
        if self.hist["h_nL_baseline"].Integral() > 0:
            self.hist["h_nL_baseline"].SetMaximum(yMax * 1e5)
            self.hist["h_nL_baseline"].SetLineColor(ROOT.kBlack)
            self.hist["h_nL_baseline"].SetLineWidth(2)
            self.hist["h_nL_baseline"].SetLineStyle(1)
            self.hist["h_nL_baseline"].Draw("HIST SAME")
            legend1.AddEntry(self.hist["h_nL_baseline"], "Baseline", "l")
        if self.hist["h_nL_pass_isolation"].Integral() > 0:
            self.hist["h_nL_pass_isolation"].SetMaximum(yMax * 1e5)
            self.hist["h_nL_pass_isolation"].SetLineColor(ROOT.kBlue)
            self.hist["h_nL_pass_isolation"].SetLineWidth(2)
            self.hist["h_nL_pass_isolation"].SetLineStyle(1)
            self.hist["h_nL_pass_isolation"].Draw("HIST SAME")
            legend1.AddEntry(self.hist["h_nL_pass_isolation"], "Pass Isolation", "l")
        if self.hist["h_nL_pass_em_olr"].Integral() > 0:
            self.hist["h_nL_pass_em_olr"].SetMaximum(yMax * 1e5)
            self.hist["h_nL_pass_em_olr"].SetLineColor(ROOT.kCyan+1)
            self.hist["h_nL_pass_em_olr"].SetLineWidth(2)
            self.hist["h_nL_pass_em_olr"].SetLineStyle(1)
            self.hist["h_nL_pass_em_olr"].Draw("HIST SAME")
            legend1.AddEntry(self.hist["h_nL_pass_em_olr"], "Pass e-#mu OLR", "l")
        if self.hist["h_nL_pass_fj_olr"].Integral() > 0:
            self.hist["h_nL_pass_fj_olr"].SetMaximum(yMax * 1e5)
            self.hist["h_nL_pass_fj_olr"].SetLineColor(ROOT.kOrange+1)
            self.hist["h_nL_pass_fj_olr"].SetLineWidth(2)
            self.hist["h_nL_pass_fj_olr"].SetLineStyle(1)
            self.hist["h_nL_pass_fj_olr"].Draw("HIST SAME")
            legend1.AddEntry(self.hist["h_nL_pass_fj_olr"], "Pass Lep-FatJet OLR", "l")
        if self.hist["h_nL_pass_pTCut"].Integral() > 0:
            self.hist["h_nL_pass_pTCut"].SetMaximum(yMax * 1e5)
            self.hist["h_nL_pass_pTCut"].SetLineColor(ROOT.kMagenta+1)
            self.hist["h_nL_pass_pTCut"].SetLineWidth(3)
            self.hist["h_nL_pass_pTCut"].SetLineStyle(1)
            self.hist["h_nL_pass_pTCut"].Draw("HIST SAME")
            legend1.AddEntry(self.hist["h_nL_pass_pTCut"], "Pass p_{T} Cut", "l")
        legend1.Draw()
        for fmt in formats:
            c1.SaveAs(os.path.join(output_dir, f"lepton_multiplicity.{fmt}"))
        c1.Close()

        # Ak4 jet multiplicty
        c2 = ROOT.TCanvas("c2", "Jet Multiplicity", cH, cW)
        c2.SetLogy()
        yMax = max(yMax, self.hist["h_nj_raw"].GetMaximum())
        self.hist["h_nj_raw"].SetMaximum(yMax * 1e3)
        self.hist["h_nj_raw"].SetLineColor(ROOT.kRed)
        self.hist["h_nj_raw"].SetLineWidth(3)
        self.hist["h_nj_raw"].Draw("HIST")
        for fmt in formats:
            c2.SaveAs(os.path.join(output_dir, f"jet_multiplicity.{fmt}"))
        c2.Close()

        # FatJet multiplicty
        c3 = ROOT.TCanvas("c3", "FatJet Multiplicity", cH, cW)
        c3.SetLogy()
        yI = 0.9
        legend1 = ROOT.TLegend(0.55, yI-0.35, 0.92, yI)
        legend1.SetFillColor(0)
        legend1.SetBorderSize(0)
        legend1.SetTextSize(0.035)
        yMax = max(
            self.hist["h_nJ_raw"].GetMaximum(), 
            self.hist["h_nJ_baseline"].GetMaximum(), 
            self.hist["h_nJ_pass_massCut"].GetMaximum(), 
            self.hist["h_nJ_pass_tau21Cut"].GetMaximum(), 
            self.hist["h_nJ_pass_pTCut"].GetMaximum()
        )
        self.hist["h_nJ_raw"].SetMaximum(yMax * 1e5)
        self.hist["h_nJ_raw"].SetLineColor(ROOT.kRed)
        self.hist["h_nJ_raw"].SetLineWidth(3)
        self.hist["h_nJ_raw"].SetLineStyle(2)
        self.hist["h_nJ_raw"].Draw("HIST")
        legend1.AddEntry(self.hist["h_nJ_raw"], "Raw", "l")
        self.hist["h_nJ_baseline"].SetMaximum(yMax * 1e5)
        self.hist["h_nJ_baseline"].SetLineColor(ROOT.kBlack)
        self.hist["h_nJ_baseline"].SetLineWidth(2)
        self.hist["h_nJ_baseline"].SetLineStyle(1)
        self.hist["h_nJ_baseline"].Draw("HIST SAME")
        legend1.AddEntry(self.hist["h_nJ_baseline"], "Baseline", "l")
        self.hist["h_nJ_pass_massCut"].SetMaximum(yMax * 1e5)
        self.hist["h_nJ_pass_massCut"].SetLineColor(ROOT.kMagenta)
        self.hist["h_nJ_pass_massCut"].SetLineWidth(2)
        self.hist["h_nJ_pass_massCut"].SetLineStyle(1)
        self.hist["h_nJ_pass_massCut"].Draw("HIST SAME")
        legend1.AddEntry(self.hist["h_nJ_pass_massCut"], "Pass Mass Cut", "l")
        self.hist["h_nJ_pass_tau21Cut"].SetMaximum(yMax * 1e5)
        self.hist["h_nJ_pass_tau21Cut"].SetLineColor(ROOT.kBlue)
        self.hist["h_nJ_pass_tau21Cut"].SetLineWidth(2)
        self.hist["h_nJ_pass_tau21Cut"].SetLineStyle(1)
        self.hist["h_nJ_pass_tau21Cut"].Draw("HIST SAME")
        legend1.AddEntry(self.hist["h_nJ_pass_tau21Cut"], "Pass #tau_{21} Cut", "l")
        self.hist["h_nJ_pass_pTCut"].SetMaximum(yMax * 1e5)
        self.hist["h_nJ_pass_pTCut"].SetLineColor(ROOT.kGreen+1)
        self.hist["h_nJ_pass_pTCut"].SetLineWidth(3)
        self.hist["h_nJ_pass_pTCut"].SetLineStyle(1)
        self.hist["h_nJ_pass_pTCut"].Draw("HIST SAME")
        legend1.AddEntry(self.hist["h_nJ_pass_pTCut"], "Pass p_{T} Cut", "l")
        legend1.Draw()
        for fmt in formats:
            c3.SaveAs(os.path.join(output_dir, f"fatjet_multiplicity.{fmt}"))
        c3.Close()

        return None

    def WriteObjectSelectionSummary(self, root_file):
        """
        Write the cutflow dict into root_file as one TH1D per object type.
        Bin i holds the (unweighted) event count for stage i
        bin labels carry the stage names.
        """
        root_file.cd()
        for obj, cf in self.cutflow.items():
            stages = list(cf.keys())
            h = ROOT.TH1D(f"cutflow_{obj}", f"{obj} selection cutflow", len(stages), 0.5, len(stages) + 0.5)
            for i, (stage, count) in enumerate(cf.items(), start=1):
                h.SetBinContent(i, count)
                h.GetXaxis().SetBinLabel(i, stage)
            h.Write()
        
        return None
    
    def WriteObjectSelectionHistograms(self, root_file):
        """Write every per-stage object-selection histogram into root_file at top level."""
        root_file.cd()
        for h in self.hist.values():
            h.Write()
        
        return None

    def SerializeObjectSelectionHistograms(self):
        """Return {name: (contents, sumw2)} as plain lists, safe to pickle."""
        out = {}
        for name, h in self.hist.items():
            nbins = h.GetNbinsX()
            contents = [h.GetBinContent(i) for i in range(1, nbins + 1)]
            sumw2    = [h.GetBinError(i) ** 2 for i in range(1, nbins + 1)]
            out[name] = (contents, sumw2)
        return out

    def MergeSerializedObjectSelectionHistograms(self, serialized_list):
        """Sum a list of {name: (contents, sumw2)} dicts into a fresh histogram dict."""
        merged = self.BookObjectSelectionHistograms()
        for name, h in merged.items():
            nbins = h.GetNbinsX()
            contents = [0.0] * nbins
            sumw2    = [0.0] * nbins
            for serialized in serialized_list:
                c_arr, w_arr = serialized[name]
                for i in range(nbins):
                    contents[i] += c_arr[i]
                    sumw2[i]    += w_arr[i]
            for i in range(nbins):
                h.SetBinContent(i + 1, contents[i])
                h.SetBinError  (i + 1, sumw2[i] ** 0.5)
        return merged

