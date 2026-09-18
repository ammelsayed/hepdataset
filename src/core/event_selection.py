# event_selection.py
import ROOT
import pandas as pd
from tabulate import tabulate

def MergeEventSelectors():
    """
    For each event selector object,
    read its cutflow and histograms, then,
    merge the cutflows
    merge the histograms
    """
    pass

class EventSelector:

    def __init__(self):
        self.splitByFlavour = False
        self.ac_keys, self.ac_dict = self.GetChannelKeys()
        self.ac_counts = dict.fromkeys(["initial", *self.ac_keys, "dropped"], 0)
    
    @classmethod
    def Merge(cls, selectors):
        """Return a new EventSelector with the ac_counts of `selectors` summed."""
        merged = cls()
        for sel in selectors:
            for k, v in sel.ac_counts.items():
                merged.ac_counts[k] = merged.ac_counts.get(k, 0) + v
        return merged
        
    def LeptonFlavour(self, lep):
        """
        Return 'lep' if we don't split by flavour, else 'e' / 'mu'.
        """
        return ("e" if lep.ClassName() == "Electron" else "mu") if self.splitByFlavour else "lep"

    def GetChannelKeys(self):
        if self.splitByFlavour:
            ac_dict = {
                "0L"   : ["Nothing", "J", "JJ", "JJJ"],
                "1L"   : ["e", "mu", "eJ", "muJ", "eJJ", "muJJ"],
                "2OSL ": ["ee", "emu", "mumu", "eeJ", "emuJ", "mumuJ"],
                "2SSL" : ["ee", "emu", "mumu", "eeJ", "emuJ", "mumuJ"],
                "3L"   : ["eee", "eemu", "emumu", "mumumu"],
                # "4L": ["eeee", "eeemu", "eemumu", "emumumu", "mumumumu"],
            }
        else:
            ac_dict = {
                "0L"   : ["Nothing", "J", "JJ", "JJJ"],
                "1L"   : ["lep", "lepJ", "lepJJ"],
                "2OSL" : ["leplep", "leplepJ"],
                "2SSL" : ["leplep", "leplepJ"],
                "3L"   : ["lepleplep"],
                # "4L": ["leplepleplep"],
            }

        ac_keys = [f"{ac}_{r}" for ac, acr in ac_dict.items() for r in acr]
        return ac_keys, ac_dict


    def ClassifyChannelKey(self, goodLeptons, goodFatJets):
        n_lep = len(goodLeptons)
        n_fj  = len(goodFatJets)

        # 0 leptons
        if n_lep == 0:
            if n_fj == 0: return "0L_Nothing"
            if n_fj == 1: return "0L_J"
            if n_fj == 2: return "0L_JJ"
            if n_fj >= 3: return "0L_JJJ"
            return None

        # 1 lepton
        if n_lep == 1:
            f = self.LeptonFlavour(goodLeptons[0])
            if n_fj == 0: return f"1L_{f}"
            if n_fj == 1: return f"1L_{f}J"
            if n_fj >= 2: return f"1L_{f}JJ"
            return None

        # 2 leptons
        if n_lep == 2:
            lep1, lep2 = goodLeptons[0], goodLeptons[1]
            q_tot_abs  = abs(lep1.Charge + lep2.Charge)
            fs = "".join(sorted(self.LeptonFlavour(l) for l in (lep1, lep2)))

            # opposite sign lepton pair (no requirment on the flavour)
            if q_tot_abs == 0: 
                if n_fj == 0: return f"2OSL_{fs}"
                if n_fj >= 1: return f"2OSL_{fs}J"
                return None

            # same sign lepton pair (no requirment on the flavour)
            if q_tot_abs == 2: 
                if n_fj == 0: return f"2SSL_{fs}"
                if n_fj >= 1: return f"2SSL_{fs}J"
                return None

        # 3 leptons
        if n_lep >= 3:
            fs = "".join(sorted(self.LeptonFlavour(l) for l in goodLeptons[:3]))
            if n_fj >= 0: return f"3L_{fs}"
            return None
        
        return None

    def Select(self, goodLeptons, goodFatJets, valid_keys = None):
        """Classify the event, update self.ac_counts, and return the ac_key (or None)."""
        if valid_keys is None:
            valid_keys = self.ac_keys
        self.ac_counts["initial"] += 1
        ac_key = self.ClassifyChannelKey(goodLeptons, goodFatJets)
        if ac_key is None or ac_key not in valid_keys:
            self.ac_counts["dropped"] += 1
            return None
        self.ac_counts[ac_key] += 1
        return ac_key

    def WriteEventSelectionSummary(self, root_file, treeName):
        root_file.cd()
        stages = list(self.ac_counts.keys())
        h = ROOT.TH1D(f"cutflow_{treeName}", f"analysis channels ({treeName})",
                      len(stages), 0.5, len(stages) + 0.5)
        for i, (stage, count) in enumerate(self.ac_counts.items(), start=1):
            h.SetBinContent(i, count)
            h.GetXaxis().SetBinLabel(i, stage)
        h.Write()

    def PrintEventSelectionSummary(self, treeName, event_weight = None, lum = None):
        total_events = self.ac_counts["initial"]
        if total_events == 0:
            return
        df = pd.DataFrame(self.ac_counts.items(), columns=[" Analysis Channel/Region", "Events"])
        if (event_weight is not None) and (lum is not None):
            df[f"Yield ({int(lum)} fb^-1)"] = df["Events"] * event_weight
            df[f"Cross Section (fb)"] = df[f"Yield ({int(lum)} fb^-1)"] / lum
        df["Fraction"] = df["Events"] / total_events
        df["Fraction"] = df["Fraction"].map(lambda x: f"{x*100:.2f}%")
        print(f"\n*** Analysis channels yields for {treeName} ***")
        print(tabulate(df, headers='keys', tablefmt="simple", showindex=False, colalign=("left",) * 4))

if __name__ == "__main__":

    import argparse
    from itertools import combinations_with_replacement as cwr

    # Simple class to mimic the Delphes Electron/Muon objects
    class Lepton:
        def __init__(self, flavour, charge):
            self._flavour = flavour
            self.Charge   = charge
        def ClassName(self):
            return self._flavour

    def _check(n_lep_max = 3, n_fj_max = 2):
        L = [Lepton(f, q) for f in ("Electron", "Muon") for q in (1, -1)]
        J = [object()]  # a "jet" — only len() matters

        samples = [(lp, nj * [J]) for nlp in range(n_lep_max + 1) for lp in cwr(L, nlp) for nj in range(n_fj_max + 1)]

        headers = ["Possible Combinations", "`splitByFlavour`=False", "`splitByFlavour`=True"]
        rows = []
        sel = EventSelector()
        rows = []
        for leps, jets in samples:
            tag = f"{' '.join([l.ClassName()[0].lower().replace("m", "mu") + ('+' if l.Charge > 0 else '-') for l in leps])}"
            seperator = "" if tag == "" else ", "
            tag += f"{seperator}{len(jets)}J" if len(jets) > 0 else ""
            sel.splitByFlavour = False
            r_false = sel.ClassifyChannelKey(leps, jets)
            sel.splitByFlavour = True
            r_true  = sel.ClassifyChannelKey(leps, jets)
            rows.append([tag, r_false, r_true])
        
        print(tabulate(rows, headers, tablefmt="github", colalign=("left",)*3))

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run channel-classification self-tests")
    args = parser.parse_args()
    if args.check:
        _check()