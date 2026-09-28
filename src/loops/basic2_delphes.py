#!/usr/bin/env python3
"""Fixed-schema flat-tree loop with object-level selection but no channel split.

The loop retains every processed event, stores at most three selected leptons
and two selected fat jets, and computes pairwise observables for populated
objects. It applies the common object cuts, but does not route events into
analysis-channel trees. See README.md for its contrast with other loops.
"""

import os
from itertools import combinations

import ROOT
import numpy as np
from tqdm import tqdm

from ..core.delphes_utilis import build_chain, load_delphes
from ..core.object_selection import ObjectSelector
from ..core.event_selection import EventSelector
from .loop_utilis import check_loop_args, run_loop_cli


MAX_LEPTONS = 3
MAX_FATJETS = 2
PAIR_OBJECTS = [*(f"Lepton{i}" for i in range(MAX_LEPTONS)),
                *(f"FatJet{i}" for i in range(MAX_FATJETS))]


def _branch_names():
    names = [f"{field}_Lepton{i}" for i in range(MAX_LEPTONS)
             for field in ("PT", "Eta", "Phi")]
    names += [f"{field}_FatJet{i}" for i in range(MAX_FATJETS)
              for field in ("PT", "Eta", "Phi", "Mass")]
    names += [f"{field}_{a}_{b}" for a, b in combinations(PAIR_OBJECTS, 2)
              for field in ("DeltaR", "DeltaPhi", "DeltaEta", "M")]
    names += ["MET", "HT", "LT", "ST", "Meff", "weight", "gen_weight"]
    return names


def loop_tree(**loop_args):
    chain = build_chain(loop_args["inputRootFile"])
    reader = ROOT.ExRootTreeReader(chain)
    args = check_loop_args(loop_args, reader.GetEntries())

    electron = reader.UseBranch("Electron")
    muon = reader.UseBranch("Muon")
    fatjet = reader.UseBranch("FatJet")
    jet = reader.UseBranch("Jet")
    met_branch = reader.UseBranch("MissingET")
    scalar_ht = reader.UseBranch("ScalarHT")
    weight_branch = reader.UseBranch("Weight")

    tree = ROOT.TTree(args["treeName"], args["treeName"])
    tree.SetDirectory(0)
    buffers = {name: np.zeros(1, dtype=np.float64) for name in _branch_names()}
    for name, buffer in buffers.items():
        tree.Branch(name, buffer, f"{name}/D")

    selector = ObjectSelector()
    event_selector = EventSelector()  # Provided for the shared make_dataset result contract.
    entries = range(args["start_entry"], args["end_entry"])
    for entry in tqdm(entries) if args["show_progress"] else entries:
        reader.ReadEntry(entry)
        selected = selector.Select(muon, electron, fatjet, jet, event_weight=args["eventWeight"])
        event_selector.Select(selected["goodLeptons"], selected["goodFatJets"])
        leptons = selected["goodLeptons"][:MAX_LEPTONS]
        fatjets = selected["goodFatJets"][:MAX_FATJETS]
        p4s = {}

        for buffer in buffers.values():
            buffer[0] = np.nan
        buffers["weight"][0] = 0.0

        for index, lep in enumerate(leptons):
            name = f"Lepton{index}"
            p4s[name] = lep.P4()
            buffers[f"PT_{name}"][0] = lep.PT
            buffers[f"Eta_{name}"][0] = lep.Eta
            buffers[f"Phi_{name}"][0] = lep.Phi

        for index, fj in enumerate(fatjets):
            name = f"FatJet{index}"
            p4s[name] = fj.P4()
            buffers[f"PT_{name}"][0] = fj.PT
            buffers[f"Eta_{name}"][0] = fj.Eta
            buffers[f"Phi_{name}"][0] = fj.Phi
            buffers[f"Mass_{name}"][0] = fj.SoftDroppedP4[0].M()

        for first_name, second_name in combinations(PAIR_OBJECTS, 2):
            if first_name not in p4s or second_name not in p4s:
                continue
            first, second = p4s[first_name], p4s[second_name]
            suffix = f"{first_name}_{second_name}"
            buffers[f"DeltaR_{suffix}"][0] = first.DeltaR(second)
            buffers[f"DeltaPhi_{suffix}"][0] = first.DeltaPhi(second)
            buffers[f"DeltaEta_{suffix}"][0] = first.Eta() - second.Eta()
            buffers[f"M_{suffix}"][0] = (first + second).M()

        met = met_branch.At(0).MET
        ht = scalar_ht.At(0).HT
        lt = sum(obj.PT for obj in selected["goodLeptons"])
        buffers["MET"][0] = met
        buffers["HT"][0] = ht
        buffers["LT"][0] = lt
        buffers["ST"][0] = ht + lt
        buffers["Meff"][0] = ht + lt + met
        buffers["gen_weight"][0] = weight_branch.At(0).Weight
        buffers["weight"][0] = args["eventWeight"]
        tree.Fill()

    root_paths = {}
    trees = {"inclusive_events": tree}
    if args["output_dir"] is not None:
        output_path = os.path.join(args["output_dir"], args["output_file_name"])
        output = ROOT.TFile.Open(output_path, "RECREATE")
        tree.SetDirectory(output)
        tree.Write()
        output.Close()
        root_paths["inclusive_events"] = output_path

    return {
        "trees": trees,
        "root_paths": root_paths,
        "ObjectSelector": selector,
        "EventSelector": event_selector,
    }


def main():
    load_delphes()
    run_loop_cli(loop_tree)
    return 0


if __name__ == "__main__":
    main()
