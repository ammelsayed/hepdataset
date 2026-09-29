# event_selection.py
import ast
from pathlib import Path

import ROOT
import yaml
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

    def __init__(self, splitByFlavour=False, config_path=None):
        self.splitByFlavour = splitByFlavour
        self.config_path = Path(config_path) if config_path else self._default_config_path()
        self._channel_config = self._load_channel_config(self.config_path)
        self.ac_keys, self.ac_dict = self.GetChannelKeys()
        self.ac_counts = dict.fromkeys(["initial", *self.ac_keys, "dropped"], 0)

    @staticmethod
    def _default_config_path():
        return Path(__file__).resolve().parents[1] / "defaults" / "event_selection.yml"

    @staticmethod
    def _load_channel_config(config_path):
        with open(config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        channels = config.get("Channels")
        if not isinstance(channels, dict):
            raise ValueError("event selection config must contain a Channels mapping")
        return channels

    @classmethod
    def Merge(cls, selectors, splitByFlavour=None):
        """Return a new EventSelector with the ac_counts of `selectors` summed."""
        if splitByFlavour is None:
            splitByFlavour = selectors[0].splitByFlavour if selectors else False
        merged = cls(splitByFlavour=splitByFlavour)
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
        """Build channel keys and classification rules from the YAML config."""
        ac_dict = {}
        self._channel_rules = []
        for channel, definition in self._channel_config.items():
            regions = (definition or {}).get("Regions", {})
            selected_regions = []
            for region, region_definition in regions.items():
                requirement = (region_definition or {}).get("requirement")
                if not isinstance(requirement, str):
                    raise ValueError(f"missing requirement for {channel}_{region}")
                names = {node.id for node in ast.walk(ast.parse(requirement, mode="eval"))
                         if isinstance(node, ast.Name)}
                uses_flavour = bool(names & {"ne", "nmu"})
                uses_generic_leptons = "nL" in names and channel != "0L"
                if uses_flavour and not self.splitByFlavour:
                    continue
                if uses_generic_leptons and not uses_flavour and self.splitByFlavour:
                    continue
                selected_regions.append(region)
                self._channel_rules.append((channel, region, requirement))
            if selected_regions:
                ac_dict[channel] = selected_regions
        ac_keys = [f"{channel}_{region}" for channel, regions in ac_dict.items()
                   for region in regions]
        return ac_keys, ac_dict

    def ClassifyChannelKey(self, goodLeptons, goodFatJets):
        """Return the first configured channel whose requirement matches the event."""
        charges = [lep.Charge for lep in goodLeptons]
        variables = {
            "nL": len(goodLeptons),
            "ne": sum(lep.ClassName() == "Electron" for lep in goodLeptons),
            "nmu": sum(lep.ClassName() == "Muon" for lep in goodLeptons),
            "QL": abs(sum(charges)),
            "nJ": len(goodFatJets),
        }
        for channel, region, requirement in self._channel_rules:
            try:
                matches = bool(eval(requirement, {"__builtins__": {}}, variables))
            except (NameError, SyntaxError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid event-selection requirement {requirement!r}") from exc
            if matches:
                return f"{channel}_{region}"
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
