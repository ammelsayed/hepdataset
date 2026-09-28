#!/usr/bin/env python3
"""Compact fixed-schema loop for exactly-one-lepton, fat-jet events.

Reads one Delphes file, applies the shared object selection, and keeps events
with exactly one selected lepton and at least one selected fat jet. It writes
separate Muon/Electron fields, up to two FatJets, lepton--FatJet observables,
and a per-event analysis weight. It is intentionally narrower than basic2,
basic3, and the configurable adaptive/explicit loops.
"""

import os
import ROOT
import numpy as np
from tqdm import tqdm

from ..core.delphes_utilis import build_chain, load_delphes
from ..core.kinematics import DeltaR, DeltaPhi, DeltaEta
from ..core.object_selection import ObjectSelector
from ..core.event_selection import EventSelector
from .loop_utilis import check_loop_args, run_loop_cli


def loop_tree(**loop_args):
    input_file = loop_args["inputRootFile"]
    chain = build_chain(input_file)
    reader = ROOT.ExRootTreeReader(chain)
    args = check_loop_args(loop_args, reader.GetEntries())

    electron = reader.UseBranch("Electron")
    muon = reader.UseBranch("Muon")
    fatjet = reader.UseBranch("FatJet")
    jet = reader.UseBranch("Jet")

    tree = ROOT.TTree(args["treeName"], args["treeName"])
    tree.SetDirectory(0)
    names = [
        "PT_Muon0", "Eta_Muon0", "Phi_Muon0",
        "PT_Electron0", "Eta_Electron0", "Phi_Electron0",
        "PT_FatJet0", "Eta_FatJet0", "Phi_FatJet0", "M_FatJet0",
        "PT_FatJet1", "Eta_FatJet1", "Phi_FatJet1", "M_FatJet1",
        "DeltaR_Lepton_FatJet0", "DeltaPhi_Lepton_FatJet0", "DeltaEta_Lepton_FatJet0",
        "DeltaR_Lepton_FatJet1", "DeltaPhi_Lepton_FatJet1", "DeltaEta_Lepton_FatJet1",
        "M_Lepton_FatJet0", "M_Lepton_FatJet1", "weight",
    ]
    buffers = {name: np.zeros(1, dtype=np.float64) for name in names}
    for name, buffer in buffers.items():
        tree.Branch(name, buffer, f"{name}/D")

    selector = ObjectSelector()
    event_selector = EventSelector()
    entries = range(args["start_entry"], args["end_entry"])
    for entry in tqdm(entries) if args["show_progress"] else entries:
        reader.ReadEntry(entry)
        selected = selector.Select(muon, electron, fatjet, jet, event_weight=args["eventWeight"])
        leptons = selected["goodLeptons"]
        fatjets = selected["goodFatJets"]
        if len(leptons) != 1 or not fatjets:
            continue

        for buffer in buffers.values():
            buffer[0] = np.nan
        buffers["weight"][0] = 0.0

        lepton = leptons[0]
        prefix = "Muon" if lepton.ClassName() == "Muon" else "Electron"
        for field, value in (("PT", lepton.PT), ("Eta", lepton.Eta), ("Phi", lepton.Phi)):
            buffers[f"{field}_{prefix}0"][0] = value

        for index, fj in enumerate(fatjets[:2]):
            tag = f"FatJet{index}"
            buffers[f"PT_{tag}"][0] = fj.PT
            buffers[f"Eta_{tag}"][0] = fj.Eta
            buffers[f"Phi_{tag}"][0] = fj.Phi
            buffers[f"M_{tag}"][0] = fj.SoftDroppedP4[0].M()
            buffers[f"DeltaR_Lepton_{tag}"][0] = DeltaR(lepton, fj)
            buffers[f"DeltaPhi_Lepton_{tag}"][0] = DeltaPhi(lepton, fj)
            buffers[f"DeltaEta_Lepton_{tag}"][0] = DeltaEta(lepton, fj)
            buffers[f"M_Lepton_{tag}"][0] = (lepton.P4() + fj.P4()).M()

        buffers["weight"][0] = args["eventWeight"]
        tree.Fill()

    trees = {"inclusive_events": tree}
    root_paths = {}
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
