# event_selection.py
import ROOT
import pandas as pd
from yaml import safe_load as yml_safe_load
from pathlib import Path
from tabulate import tabulate
from itertools import combinations_with_replacement as cwr

class Lepton:
    """Simple class to mimic the Delphes Electron/Muon objects"""
    def __init__(self, flavour, charge):
        self._flavour = flavour
        self.Charge   = charge
    def ClassName(self):
        return self._flavour

class EventSelector:

    def __init__(self, config_file=None):
        self.ReadConfigFile(config_file)

        # Regions must be mutually exclusive
        self.CheckExclusivity()
        
        # Classify search catagories into channels and regions
        self.ac_keys, self.ac_dict = self.GetChannelKeys()
        self.ac_counts = dict.fromkeys(["initial", *self.ac_keys, "dropped"], 0)
    
    def ReadConfigFile(self, config_file):
        # default config file
        if config_file == None:
            config_file = {
                "Channels" : {
                    "2OSL": {
                        "Regions" : {
                            "leplep": {
                                "requirement" : "(nL == 2) & (QL == 0)"
                            }
                        }
                    },
                    "2SSL": {
                        "Regions" : {
                            "leplep": {
                                "requirement" : "(nL == 2) & (QL == 2)"
                            }
                        }
                    }
                }
            }

        # a dict is accepted too, for cards that carry the Event-Selection block inline
        if isinstance(config_file, dict):
            self.config_path = None
            try:
                self.config_file = config_file["Channels"]
            except (KeyError, TypeError) as e:
                raise ValueError(f"Invalid event-selection block: {e}")
        else:
            # Require a valid YAML card path
            config_file = Path(config_file)
            if not config_file.is_file():
                raise FileNotFoundError(f"YAML card not found: {config_file}")
            self.config_path = config_file

            # Load the config file
            with open(config_file) as f:
                try:
                    self.config_file = yml_safe_load(f)["Channels"]
                except (KeyError, TypeError) as e:
                    raise ValueError(f"Invalid YAML card '{config_file}': {e}")
        
        return None
    
    def GetChannelKeys(self):
        ac_dict = {ch: list(data["Regions"]) for ch, data in self.config_file.items()}
        ac_keys = [f"{ch}_{r}" for ch, rs in ac_dict.items() for r in rs]
        return ac_keys, ac_dict

    @staticmethod
    def _event_vars(goodLeptons, goodFatJets, n_leading=5):
        # each entry is {"value": <number>, "description": <help text>}; eval uses the "value"
        _vars = {
            "nL":    {
                "value": len(goodLeptons), 
                "description": "Total number of leptons"
            },
            "nEle":  {
                "value": sum(1 for l in goodLeptons if l.ClassName() == "Electron"),
                "description": "Number of electrons"
            },
            "nMu":   {
                "value": sum(1 for l in goodLeptons if l.ClassName() == "Muon"),
                "description": "Number of muons"
            },
            "QL":    {
                "value": abs(sum(l.Charge for l in goodLeptons)),
                "description": "Absolute total charge of the leptons"
            },
            "nFatJ": {
                "value": len(goodFatJets),
                "description": "Number of fat-jets"
            },
            "nJ": {
                "value": 0,
                "description": "Number of jets (not filled yet)"
            },
            "nBJ": {
                "value": 0,
                "description": "Number of b-jets (not filled yet)"
            },
            "nCJ": {
                "value": 0,
                "description": "Number of c-jets (not filled yet)"
            },
            "nTauJ": {
                "value": 0,
                "description": "Number of tau-jets (not filled yet)"
            },
        }
        # leading-N flavour counts; assumes goodLeptons is pT-descending (object_selection.py)
        for n in range(1, n_leading + 1):
            _vars[f"nEle{n}"] = {
                "value": sum(1 for l in goodLeptons[:n] if l.ClassName() == "Electron"),
                "description": f"Number of electrons among the leading {n} lepton(s) in pT"
            }
            _vars[f"nMu{n}"] = {
                "value": sum(1 for l in goodLeptons[:n] if l.ClassName() == "Muon"),
                "description": f"Number of muons among the leading {n} lepton(s) in pT"
            }
        return _vars

    def ClassifyChannelKey(self, goodLeptons, goodFatJets):
        v = {k: d["value"] for k, d in self._event_vars(goodLeptons, goodFatJets).items()}
        for ch, data in self.config_file.items():
            for r, spec in data["Regions"].items():
                if eval(spec["requirement"], {}, v):
                    return f"{ch}_{r}"
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

    def CheckExclusivity(self, n_lep_max = 4, n_fj_max = 2):
        """Raise if any two regions match the same event."""
        L = [Lepton(f, q) for f in ("Electron", "Muon") for q in (1, -1)]
        J = [object()]  # a "jet" — only len() matters
        for nlp in range(n_lep_max + 1):
            for leps in cwr(L, nlp):
                for nj in range(n_fj_max + 1):
                    v = {k: d["value"] for k, d in self._event_vars(leps, nj * J).items()}
                    matches = [f"{ch}_{r}" for ch, data in self.config_file.items()
                               for r, spec in data["Regions"].items()
                               if eval(spec["requirement"], {}, v)]
                    if len(matches) > 1:
                        raise ValueError(f"Non-exclusive regions matched by the same event: {matches}")

    def Inspect(self, nb_lep = 4, nb_fj = 2):
        """Print ClassifyChannelKey output for all small event combinations."""
        L = [Lepton(f, q) for f in ("Electron", "Muon") for q in (1, -1)]
        J = [object()]  # a "jet" — only len() matters
        headers = ["Possible Combinations", "Selected Output"]
        rows = []
        for nlp in range(nb_lep + 1):
            for leps in cwr(L, nlp):
                for nj in range(nb_fj + 1):
                    jets = nj * J
                    tag = f"{' '.join([l.ClassName()[0].lower() + ('+' if l.Charge > 0 else '-') for l in leps])}".replace("m", "mu")
                    seperator = "" if tag == "" else ", "
                    tag += f"{seperator}{nj}J" if nj > 0 else ""
                    rows.append([tag, self.ClassifyChannelKey(leps, jets)])
        print(tabulate(rows, headers, tablefmt="github", colalign=("left",)*2))
        print("Check every selected output and compare it with your expected output.")

    @classmethod
    def Merge(cls, selectors, splitByFlavour=False):
        """Return a new EventSelector with the ac_counts of `selectors` summed."""
        # reuse the parsed config: config_path is None when the caller passed a dict
        merged = cls(selectors[0].config_path if selectors[0].config_path else {"Channels": selectors[0].config_file})
        for sel in selectors:
            for k, v in sel.ac_counts.items():
                merged.ac_counts[k] = merged.ac_counts.get(k, 0) + v
        return merged

    def WriteEventSelectionSummary(self, root_file, treeName):
        root_file.cd()
        stages = list(self.ac_counts.keys())
        h = ROOT.TH1D(f"cutflow_{treeName}", f"analysis channels ({treeName})",
                      len(stages), 0.5, len(stages) + 0.5)
        for i, (stage, count) in enumerate(self.ac_counts.items(), start=1):
            h.SetBinContent(i, count)
            h.GetXaxis().SetBinLabel(i, stage)
        h.Write()

    def ReadEventSelectionSummary(self, root_file, treeName):
        f = ROOT.TFile.Open(str(root_file), "READ")
        if not f or f.IsZombie():
            raise OSError(f"Could not open ROOT file: {root_file}")
        h = f.Get(f"cutflow_{treeName}")
        if not h:
            f.Close()
            raise KeyError(f"Histogram cutflow_{treeName} not found in {root_file}")
        self.ac_counts = {
            h.GetXaxis().GetBinLabel(i): h.GetBinContent(i)
            for i in range(1, h.GetNbinsX() + 1)
        }
        f.Close()
        return self

    def EventSelectionSummaryMarkdown(self):
        total_events = self.ac_counts["initial"]
        rows = [
            [stage, f"{count:,.0f}", f"{count / total_events:.2%}" if total_events else "—"]
            for stage, count in self.ac_counts.items()
        ]
        return tabulate(rows, headers=["Analysis Channel/Region", "Events", "Fraction"], tablefmt="github")

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
        print(tabulate(df, headers='keys', tablefmt="simple", showindex=False, colalign=("left",) * len(df.columns)))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Event variables usable in a card's 'requirement:' expressions\n"
               "(these are NOT command-line arguments, do not pass them on the command line):\n\n"
               + tabulate([[k, d["description"]] for k, d in EventSelector._event_vars([], []).items()], headers=["variable", "meaning"], tablefmt="plain"))
    parser.add_argument("config_file", help=(
        "YAML card with channel/region definitions, for example:\n"
        "--- config_file.yml ----------------\n"
        "|Channels:\n"
        "|  2OSL:\n"
        "|    Regions:\n"
        "|      leplep:\n"
        "|        requirement: (nL == 2) & (QL == 0)\n"
        "|  2SSL:\n"
        "|    Regions:\n"
        "|      leplep:\n"
        "|        requirement: (nL == 2) & (QL == 2)\n"
        "--- config_file.yml ----------------\n"))
    parser.add_argument("--nb_lep", type=int, default=4, help="Maximum number of leptons used in classification inspection")
    parser.add_argument("--nb_fj", type=int, default=2, help="Maximum number of fatjets used in classification inspection")
    parser.add_argument("--check", action="store_true", help="Print classification output for all small event combinations")
    args = parser.parse_args()

    sel = EventSelector(args.config_file)
    print(f"Loaded {len(sel.ac_keys)} channels/regions from {args.config_file}")

    if args.check:
        sel.Inspect()

