# event_selection.py
import ast
from itertools import combinations_with_replacement as cwr
from pathlib import Path

import ROOT
import pandas as pd
import yaml
from tabulate import tabulate


_VARIABLES = {"nL", "ne", "nmu", "QL", "nJ"}


def _value(node, variables):
    if isinstance(node, ast.Name):
        if node.id not in _VARIABLES:
            raise ValueError(f"unknown variable '{node.id}'")
        return variables.get(node.id, 0)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_value(node.operand, variables)
    raise ValueError("requirements support variables, numbers, comparisons, and '&'/'|' only")


def _matches(node, variables):
    if isinstance(node, ast.BoolOp):
        values = [_matches(value, variables) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        left, right = _matches(node.left, variables), _matches(node.right, variables)
        return left and right if isinstance(node.op, ast.BitAnd) else left or right
    if isinstance(node, ast.Compare):
        left = _value(node.left, variables)
        for operator, comparator in zip(node.ops, node.comparators):
            right = _value(comparator, variables)
            if isinstance(operator, ast.Eq):
                matches = left == right
            elif isinstance(operator, ast.NotEq):
                matches = left != right
            elif isinstance(operator, ast.Lt):
                matches = left < right
            elif isinstance(operator, ast.LtE):
                matches = left <= right
            elif isinstance(operator, ast.Gt):
                matches = left > right
            elif isinstance(operator, ast.GtE):
                matches = left >= right
            else:
                raise ValueError("requirements support ==, !=, <, <=, >, and >= only")
            if not matches:
                return False
            left = right
        return True
    return bool(_value(node, variables))


class EventSelector:
    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "defaults" / "event_selection.yml"
        self._channel_config = self._load_channel_config(self.config_path)
        self.ac_keys, self.ac_dict = self.GetChannelKeys()
        self.ac_counts = dict.fromkeys(["initial", *self.ac_keys, "dropped"], 0)

    @staticmethod
    def _load_channel_config(config_path):
        with open(config_path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        return config.get("Channels", {})

    @classmethod
    def Merge(cls, selectors):
        """Return a selector with the counts of `selectors` summed."""
        merged = cls(selectors[0].config_path if selectors else None)
        for selector in selectors:
            for key, value in selector.ac_counts.items():
                merged.ac_counts[key] = merged.ac_counts.get(key, 0) + value
        return merged

    def GetChannelKeys(self):
        ac_dict, self._channel_rules = {}, []
        for channel, definition in self._channel_config.items():
            regions = definition.get("Regions", {})
            ac_dict[channel] = list(regions)
            for region, region_definition in regions.items():
                requirement = region_definition["requirement"]
                try:
                    tree = ast.parse(requirement, mode="eval").body
                except SyntaxError as error:
                    raise ValueError(f"invalid requirement for {channel}_{region}") from error
                self._channel_rules.append((channel, region, tree))
        ac_keys = [f"{channel}_{region}" for channel, regions in ac_dict.items() for region in regions]
        return ac_keys, ac_dict

    def _variables(self, leptons, fatjets):
        charges = [lep.Charge for lep in leptons]
        return {
            "nL": len(leptons),
            "ne": sum(lep.ClassName() == "Electron" for lep in leptons),
            "nmu": sum(lep.ClassName() == "Muon" for lep in leptons),
            "QL": abs(sum(charges)),
            "nJ": len(fatjets),
        }

    def _matching_channels(self, goodLeptons, goodFatJets):
        variables = self._variables(goodLeptons, goodFatJets)
        return [f"{channel}_{region}" for channel, region, tree in self._channel_rules if _matches(tree, variables)]

    def ClassifyChannelKey(self, goodLeptons, goodFatJets):
        matches = self._matching_channels(goodLeptons, goodFatJets)
        return matches[0] if matches else None

    def Select(self, goodLeptons, goodFatJets, valid_keys=None):
        """Classify the event, update the cutflow, and return its channel key."""
        valid_keys = self.ac_keys if valid_keys is None else valid_keys
        self.ac_counts["initial"] += 1
        ac_key = self.ClassifyChannelKey(goodLeptons, goodFatJets)
        if ac_key is None or ac_key not in valid_keys:
            self.ac_counts["dropped"] += 1
            return None
        self.ac_counts[ac_key] += 1
        return ac_key

    @classmethod
    def Inspect(cls, config_path):
        selector = cls(config_path)
        leptons = [type("Lepton", (), {"ClassName": lambda self, name=name: name, "Charge": charge})()
                   for name in ("Electron", "Muon") for charge in (1, -1)]
        overlaps = []
        for n_leptons in range(5):
            for event_leptons in cwr(leptons, n_leptons):
                for n_fatjets in range(4):
                    matches = selector._matching_channels(event_leptons, [object()] * n_fatjets)
                    if len(matches) > 1:
                        overlaps.append(matches)
        if overlaps:
            print("Invalid card: overlapping regions found:")
            for matches in overlaps[:10]:
                print(f"  {', '.join(matches)}")
            return False
        print(f"Valid card: {len(selector.ac_keys)} exclusive channels")
        return True

    def WriteEventSelectionSummary(self, root_file, treeName):
        root_file.cd()
        stages = list(self.ac_counts)
        histogram = ROOT.TH1D(f"cutflow_{treeName}", f"analysis channels ({treeName})", len(stages), 0.5, len(stages) + 0.5)
        for index, (stage, count) in enumerate(self.ac_counts.items(), start=1):
            histogram.SetBinContent(index, count)
            histogram.GetXaxis().SetBinLabel(index, stage)
        histogram.Write()

    def PrintEventSelectionSummary(self, treeName, event_weight=None, lum=None):
        total_events = self.ac_counts["initial"]
        if total_events == 0:
            return
        df = pd.DataFrame(self.ac_counts.items(), columns=[" Analysis Channel/Region", "Events"])
        if event_weight is not None and lum is not None:
            df[f"Yield ({int(lum)} fb^-1)"] = df["Events"] * event_weight
            df["Cross Section (fb)"] = df[f"Yield ({int(lum)} fb^-1)"] / lum
        df["Fraction"] = (df["Events"] / total_events).map(lambda value: f"{value * 100:.2f}%")
        print(f"\n*** Analysis channels yields for {treeName} ***")
        print(tabulate(df, headers="keys", tablefmt="simple", showindex=False, colalign=("left",) * len(df.columns)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect an event-selection YAML card.")
    parser.add_argument("config_path", help="Path to event_selection.yml")
    parser.add_argument("--inspect", action="store_true", help="Check that configured regions are exclusive")
    args = parser.parse_args()
    if args.inspect and not EventSelector.Inspect(args.config_path):
        raise SystemExit(1)
