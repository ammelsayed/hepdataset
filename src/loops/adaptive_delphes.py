#!/usr/bin/env python3
import os
import ROOT
import numpy as np
from tqdm import tqdm
from mt2 import mt2
from ..core.delphes_utilis   import build_chain
from ..core.object_selection import ObjectSelector
from ..core.event_selection  import EventSelector
from ..core.branches_reader  import BranchesHandler
from .loop_utilis            import check_loop_args
from ..core.kinematics       import EventShapes, Centrality, MtW

def loop_tree(**loop_args):
    inputRootFile = loop_args["inputRootFile"]
    collect_summary = loop_args.pop("collect_summary", False)

    # Read the input file
    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)

    # Fill defaults, validate entries, compute weight, prepare output dir, print summary
    loop_args = check_loop_args(loop_args, TreeReader.GetEntries())
    start_entry, end_entry = loop_args["start_entry"], loop_args["end_entry"]
    eventWeight = loop_args["eventWeight"]
    treeName = loop_args["treeName"]
    show_progress = loop_args["show_progress"]
    debug_loop = loop_args["debug_loop"]
    output_dir = loop_args["output_dir"]
    output_file_name = loop_args["output_file_name"]
    luminosity = loop_args["luminosity"]
    
    # Branches to read
    Electron_branch  = TreeReader.UseBranch("Electron")
    Muon_branch      = TreeReader.UseBranch("Muon")
    FatJet_branch    = TreeReader.UseBranch("FatJet")
    Jet_branch       = TreeReader.UseBranch("Jet")
    MissingET_branch = TreeReader.UseBranch("MissingET")
    ScalarHT_branch  = TreeReader.UseBranch("ScalarHT")
    Weight_branch    = TreeReader.UseBranch("Weight")

    # get the branch names
    BR = BranchesHandler(branches_config_path)
    if not BR.is_valid():
        BR.print_validation()
        raise ValueError(f"Invalid branches configuration: {branches_config_path}")
    float_branch_names = BR.get_float_branch_names()
    int_branch_names = BR.get_int_branch_names()

    # Analysis channel bookkeeping
    eventSel = EventSelector(loop_args["event_selection_config_path"])
    ac_keys, ac_dict = eventSel.ac_keys, eventSel.ac_dict

    # Initialize object selector
    objSel = ObjectSelector()

    # book the trees and create the branch buffers
    trees, buffers = {}, {}
    for ac_key in ac_keys:
        tree = ROOT.TTree(treeName, treeName)
        tree.SetDirectory(0)

        # Create a fresh buffer dictionary for this tree
        b = {}
        for branch_name in int_branch_names:
            b[branch_name] = np.zeros(1, dtype=np.int32)
            tree.Branch(branch_name, b[branch_name], f"{branch_name}/I")  # /I for Integer
        for branch_name in float_branch_names:
            b[branch_name] = np.zeros(1, dtype=np.float64)
            tree.Branch(branch_name, b[branch_name], f"{branch_name}/D")  # /D for Double (64-bit float)
        buffers[ac_key] = b      

        trees[ac_key] = tree
    
    # Event loop
    rng = range(start_entry, end_entry)
    for entry in (tqdm(rng) if show_progress else rng):
        
        TreeReader.ReadEntry(entry)

        # Object selection
        selected_objects = objSel.Select(Muon_branch, Electron_branch, FatJet_branch, Jet_branch, event_weight = eventWeight)
        goodFatJets = selected_objects["goodFatJets"]
        goodLeptons = selected_objects["goodLeptons"]
        goodJets = selected_objects["goodJets"]
        goodBJets = selected_objects["goodBJets"]
        goodTauJets = selected_objects["goodTauJets"]

        # Identify the analysis channel key
        ac_key = eventSel.Select(goodLeptons, goodFatJets, valid_keys = trees.keys())

        # Skip events that don't match any analysis channel
        if ac_key is None:
            continue
        
        # Reset branches to np.nan (weight defaults to 0.0 for skipped events)
        b = buffers[ac_key]
        for branch_name in int_branch_names:
            b[branch_name][0] = -1
        for branch_name in float_branch_names:
            b[branch_name][0] = np.nan
        b["weight"][0] = 0.0
        b["gen_weight"][0] = 0.0

        # ================================================================
        # Fill the branches
        # ================================================================

        # A dictonary to hold the selected objects' four-momenta for easier access
        P4s = {}
        Charges = {}
        Flavours = {}

        # Count number of objects before quality selections AND store them locally
        nPreQS_Muon = Muon_branch.GetEntries()
        nPreQS_Electron = Electron_branch.GetEntries()
        nPreQS_Lepton = nPreQS_Muon + nPreQS_Electron
        nPreQS_FatJet = FatJet_branch.GetEntries()
        nPreQS_Jet = Jet_branch.GetEntries()
        b["nPreQS_Muon"][0] = nPreQS_Muon
        b["nPreQS_Electron"][0] = nPreQS_Electron
        b["nPreQS_Lepton"][0] = nPreQS_Lepton
        b["nPreQS_FatJet"][0] = nPreQS_FatJet
        b["nPreQS_Jet"][0] = nPreQS_Jet
       
        # ----------------------------------------------------------------
        # Fill leptons kinematics
        # ---------------------------------------------------------------

        muonMass = 0.1057
        electronMass = 0.511E-3

        for lep_idx, lepton in enumerate(goodLeptons[:BR.get_obj_count("Lepton")]):
            mainClassNames = ["Lepton"] # branches including these class names will be filled
            lepClassName = lepton.ClassName()
            if lepClassName in BR.get_obj_repr("Lepton"):
                mainClassNames.append(lepClassName)

            # Store charge and flavour for N-body combinations
            if lepClassName == 'Muon':
                Charges[f"Lepton{lep_idx}"] = int(lepton.Charge)
                Flavours[f"Lepton{lep_idx}"] = 2
                Charges[f"Muon{lep_idx}"] = int(lepton.Charge)
                Flavours[f"Muon{lep_idx}"] = 2
            elif lepClassName == 'Electron':
                Charges[f"Lepton{lep_idx}"] = int(lepton.Charge)
                Flavours[f"Lepton{lep_idx}"] = 1
                Charges[f"Electron{lep_idx}"] = int(lepton.Charge)
                Flavours[f"Electron{lep_idx}"] = 1

            for lepType in mainClassNames:
                inst = f"{lepType}{lep_idx}"

                # Correct P4 object with mass
                if lepClassName == 'Muon':
                    p4 = ROOT.TLorentzVector()
                    p4.SetPtEtaPhiM(lepton.PT, lepton.Eta, lepton.Phi, muonMass)
                    P4s[inst] = p4
                elif lepClassName == 'Electron':
                    p4 = ROOT.TLorentzVector()
                    p4.SetPtEtaPhiM(lepton.PT, lepton.Eta, lepton.Phi, electronMass)
                    P4s[inst] = p4
                else: continue 

                # Fill kinematics asked for
                for k in BR.get_obj_kinematics("Lepton"):
                    branch_name = f"{k}_{inst}"
                    if k == "ElectronTag":
                        b[branch_name][0] = 1 if lepClassName == "Electron" else 0
                    elif k == "MuonTag":
                        b[branch_name][0] = 1 if lepClassName == "Muon" else 0
                    else:
                        try:
                            b[branch_name][0] = getattr(lepton, k) # example: lep.PT 
                        except AttributeError:
                            try:
                                b[branch_name][0] = getattr(p4, k)() # example: lep.P4().Px() 
                            except AttributeError as e:
                                continue # leave as NaN (already set at reset time)

        # ----------------------------------------------------------------
        # Fill Large-R (FatJets) kinematics
        # ---------------------------------------------------------------

        for fj_idx, fatjet in enumerate(goodFatJets[:BR.get_obj_count("FatJet")]):
            classname = fatjet.ClassName()
            p4_fj = fatjet.P4()
            P4s[f"FatJet{fj_idx}"] = p4_fj

            # Log softdropped/trimmed/pruned large-R jet four-momenta
            # Required only if softdropped/trimmed/prune fatjets are configured as a representaion of large-R jets.
            if hasattr(fatjet, 'SoftDroppedP4') and ("SoftDroppedFatJet" in BR.get_obj_repr("FatJet")):
                p4_sd = fatjet.SoftDroppedP4[0]
                P4s[f"SoftDroppedFatJet{fj_idx}"] = p4_sd
            if hasattr(fatjet, 'TrimmedP4') and ("TrimmedFatJet" in BR.get_obj_repr("FatJet")):
                p4_tr = fatjet.TrimmedP4[0]            
                P4s[f"TrimmedFatJet{fj_idx}"] = p4_tr
            if hasattr(fatjet, 'PrunedP4') and ("PrunedFatJet" in BR.get_obj_repr("FatJet")):
                p4_pr = fatjet.PrunedP4[0]
                P4s[f"PrunedFatJet{fj_idx}"] = p4_pr

            # Fill kinematics asked for
            for k in BR.get_obj_kinematics("FatJet"):                
                if k in ["Tau1", "Tau2", "Tau3", "Tau4", "Tau5", "Tau21", "Tau31", "Tau32", "Tau41", "Tau42", "Tau43", "Tau51", "Tau52", "Tau53", "Tau54"]: continue # n-subjetness variables are handeled seperatly
                branch_name = f"{k}_FatJet{fj_idx}"
                try:
                    b[branch_name][0] = getattr(fatjet, k)  
                except AttributeError:
                    try:
                        b[branch_name][0] = getattr(p4_fj, k)()  
                    except AttributeError as e:
                        continue

                # grooming variants do not have direct methods so we just use one layer of reading 
                if hasattr(fatjet, 'SoftDroppedP4') and ("SoftDroppedFatJet" in BR.get_obj_repr("FatJet")):
                    branch_name = f"{k}_SoftDroppedFatJet{fj_idx}"
                    try:
                        b[branch_name][0] = getattr(p4_sd, k)()  
                    except AttributeError as e:
                        continue
                if hasattr(fatjet, 'TrimmedP4') and ("TrimmedFatJet" in BR.get_obj_repr("FatJet")):
                    branch_name = f"{k}_TrimmedFatJet{fj_idx}"
                    try:
                        b[branch_name][0] = getattr(p4_tr, k)()  
                    except AttributeError as e:
                        continue
                if hasattr(fatjet, 'PrunedP4') and ("PrunedFatJet" in BR.get_obj_repr("FatJet")):
                    branch_name = f"{k}_PrunedFatJet{fj_idx}"
                    try:
                        b[branch_name][0] = getattr(p4_pr, k)()  
                    except AttributeError as e:
                        continue

            # Fill n-subjetness substructure variables
            # Those are only filled for FatJet, t groomed variants do not carry these methods
            ntaus_available = min(len(fatjet.Tau), 5)
            requested_taus = BR.get_obj_kinematics("FatJet")
            tau_values = {}

            for t_idx in range(1, ntaus_available + 1):
                tau_name = f"Tau{t_idx}"
                if tau_name in requested_taus:
                    tau_values[t_idx] = fatjet.Tau[t_idx - 1]
                    b[f"{tau_name}_FatJet{fj_idx}"][0] = tau_values[t_idx]

            for k in requested_taus:
                if k.startswith("Tau") and len(k) == 5 and k[3].isdigit() and k[4].isdigit():
                    i, j = int(k[3]), int(k[4])
                    if i <= ntaus_available and j <= ntaus_available and i > j:
                        ti = tau_values.get(i)
                        if ti is None:
                            ti = fatjet.Tau[i - 1]
                            tau_values[i] = ti
                        tj = tau_values.get(j)
                        if tj is None:
                            tj = fatjet.Tau[j - 1]
                            tau_values[j] = tj
                        b[f"Tau{i}{j}_FatJet{fj_idx}"][0] = ti / tj if tj > 0 else 1.0

        # ----------------------------------------------------------------
        # Fill Small-R Jets kinematics
        # ----------------------------------------------------------------

        for goodJetsList, prefix in zip([goodJets, goodBJets, goodTauJets], ["Jet", "BJet", "TauJet"]):
            if prefix not in BR.get_obj_repr("Jet"): continue
            for jet_idx, jet in enumerate(goodJetsList[:BR.get_obj_count("Jet")]):
                classname = jet.ClassName()
                inst = f"{prefix}{jet_idx}"
                p4 = jet.P4()
                P4s[inst] = p4
                for k in BR.get_obj_kinematics("Jet"):
                    branch_name = f"{k}_{inst}"
                    try:
                        b[branch_name][0] = getattr(jet, k)  
                    except AttributeError:
                        try:
                            b[branch_name][0] = getattr(p4, k)()  
                        except AttributeError as e:
                            continue

        # ----------------------------------------------------------------
        # Fill scalars and event shapes
        # ----------------------------------------------------------------

        # Fill the met kinematics
        met = MissingET_branch.At(0)
        p4_met = met.P4()
        P4s["MET"] = p4_met
        for k in BR.get_obj_kinematics("MET"):
            branch_name = f"{k}_MET"
            try:
                b[branch_name][0] = getattr(met, k)  
            except AttributeError:
                try:
                    b[branch_name][0] = getattr(p4_met, k)()  
                except AttributeError as e:
                    continue

        # Fill other global scalars        
        ht = ScalarHT_branch.At(0).HT
        if "HT" in BR.global_scalars: b["HT"][0] = ht
        
        lt = sum([lep.PT for lep in goodLeptons])
        st = lt + ht
        meff = st + met.MET
        sig_met = ( met.MET / np.sqrt(ht) ) if ht > 0 else np.nan
        if "LT" in BR.global_scalars: b["LT"][0] = lt
        if "ST" in BR.global_scalars: b["ST"][0] = st
        if "Significance_MET" in BR.global_scalars: b["Significance_MET"][0] = sig_met
        if "Meff" in BR.global_scalars: b["Meff"][0] = meff

        sumPT_jets = sum([jet.PT for jet in goodJets])
        sumPT_fjs = sum([fj.PT for fj in goodFatJets])
        sumPT_sdfjs = sum([fj.SoftDroppedP4[0].Pt() for fj in goodFatJets])

        if "ScalarSumPT_Jets" in BR.global_scalars: b["ScalarSumPT_Jets"][0] = sumPT_jets
        if "ScalarSumPT_FatJets" in BR.global_scalars: b["ScalarSumPT_FatJets"][0] = sumPT_fjs
        if "ScalarSumPT_SoftDroppedFatJets" in BR.global_scalars: b["ScalarSumPT_SoftDroppedFatJets"][0] = sumPT_sdfjs
        if "ScalarSumPT_Hadronic" in BR.global_scalars: b["ScalarSumPT_Hadronic"][0] = sumPT_jets + sumPT_sdfjs # should equal HT
        
        # Fill event shapes
        if BR.event_shapes:
            vis_p4s = [lep.P4() for lep in goodLeptons] + [fj.P4() for fj in goodFatJets] + [jet.P4() for jet in goodJets] + [met.P4()]
            px_arr = np.array([p.Px() for p in vis_p4s], dtype=np.float64)
            py_arr = np.array([p.Py() for p in vis_p4s], dtype=np.float64)
            pz_arr = np.array([p.Pz() for p in vis_p4s], dtype=np.float64)
            e_arr  = np.array([p.Energy() for p in vis_p4s], dtype=np.float64)
        
            if "Sphericity" in BR.event_shapes or "Aplanarity" in BR.event_shapes or "Circularity" in BR.event_shapes:
                S, A, C = EventShapes(px_arr, py_arr, pz_arr)
                if "Sphericity" in BR.event_shapes: b["Sphericity"][0] = S
                if "Aplanarity" in BR.event_shapes: b["Aplanarity"][0] = A
                if "Circularity" in BR.event_shapes: b["Circularity"][0] = C
            if "Centrality" in BR.event_shapes: b["Centrality"][0] = Centrality(px_arr, py_arr, pz_arr, e_arr)

        # ----------------------------------------------------------------
        # Fill N-body relations
        # ----------------------------------------------------------------

        if BR.multiObjects_Nmax >= 2:
            for N in range(2, BR.multiObjects_Nmax + 1, 1):
                for p_names in BR.get_nbody_combinations(N):
                    p4_list = [P4s.get(n) for n in p_names]
                    if all(p4_list):
                        sufx = '_'.join(p_names)

                        # Without start, sum() defaults to  0 (integer), 
                        # which would fail because you can't add 0 + TLorentzVector()
                        total_p4 = sum(p4_list[1:], p4_list[0])

                        # fill basic kinematics first 
                        for k in BR.get_nbody_kinematics(N):
                            if k not in BR.multiObject_2body_kinematics:
                                branch_name = f"{k}_{sufx}"
                                try:
                                    b[branch_name][0] = getattr(total_p4, k)()  
                                except AttributeError as e:
                                    continue

                        # by default 2body kinematics should be included for any 2-body objects
                        # those are just ["DeltaR", "DeltaPhi", "DeltaEta", "MtW", "isOSSF", "isOSOF", "isSSOF", "isSSSF"]
                        # we fill them manullay
                        if N == 2:
                            p1 = p4_list[0]; p2 = p4_list[1]
                            if "DeltaR" in BR.multiObject_2body_kinematics:
                                b[f"DeltaR_{sufx}"][0] = p1.DeltaR(p2)
                            if "DeltaPhi" in BR.multiObject_2body_kinematics:
                                b[f"DeltaPhi_{sufx}"][0] = p1.DeltaPhi(p2)
                            if "DeltaEta" in BR.multiObject_2body_kinematics:
                                b[f"DeltaEta_{sufx}"][0] = p1.Eta() - p2.Eta()
                            if "MtW" in BR.multiObject_2body_kinematics and "MET" in p_names:
                                b[f"MtW_{sufx}"][0] = MtW(p1.Pt(), p1.Phi(), p2.Pt(), p2.Phi())
                            
                            # Fill OS/SS and SF/OF tags for lepton pairs
                            p1_name = p_names[0]; p2_name = p_names[1]
                            if (p1_name in Flavours) and (p2_name in Flavours):
                                q1 = Charges[p1_name]; q2 = Charges[p2_name]
                                f1 = Flavours[p1_name]; f2 = Flavours[p2_name]
                                
                                os_flag = (q1 * q2 == -1)
                                ss_flag = (q1 * q2 == 1)
                                sf_flag = (f1 == f2)
                                of_flag = (f1 != f2)
                                
                                if "isOSSF" in BR.multiObject_2body_kinematics:
                                    b[f"isOSSF_{sufx}"][0] = 1 if (os_flag and sf_flag) else 0
                                if "isOSOF" in BR.multiObject_2body_kinematics:
                                    b[f"isOSOF_{sufx}"][0] = 1 if (os_flag and of_flag) else 0
                                if "isSSOF" in BR.multiObject_2body_kinematics:
                                    b[f"isSSOF_{sufx}"][0] = 1 if (ss_flag and of_flag) else 0
                                if "isSSSF" in BR.multiObject_2body_kinematics:
                                    b[f"isSSSF_{sufx}"][0] = 1 if (ss_flag and sf_flag) else 0

                        # MT2 uses the configured pair of visible objects and
                        # the event's missing transverse momentum.
                        if BR.multiObjects_include_mt2 and N == 2:
                            supported = set(BR.get_mt2_obj_instances())
                            p1 = p4_list[0]; p2 = p4_list[1]
                            p1_name = p_names[0]; p2_name = p_names[1]
                            if p1_name in supported and p2_name in supported:
                                b[f"MT2_{sufx}"][0] = mt2(
                                    p1.M(), p1.Px(), p1.Py(),
                                    p2.M(), p2.Px(), p2.Py(),
                                    p4_met.Px(), p4_met.Py(),
                                    0.0, 0.0   # neutrino masses
                                )
                        
                        # only fill the event shapes if they are asked for
                        if BR.multiObjects_include_trival_kinematics:
                            tk = set(BR.multiObject_trival_kinematics)
                            px_arr = np.array([p.Px() for p in p4_list], dtype=np.float64)
                            py_arr = np.array([p.Py() for p in p4_list], dtype=np.float64)
                            pz_arr = np.array([p.Pz() for p in p4_list], dtype=np.float64)
                            e_arr  = np.array([p.Energy() for p in p4_list], dtype=np.float64)

                            if "Sphericity" in tk or "Aplanarity" in tk or "Circularity" in tk:
                                S, A, C = EventShapes(px_arr, py_arr, pz_arr)
                                if "Sphericity" in tk: b[f"Sphericity_{sufx}"][0] = S
                                if "Aplanarity" in tk: b[f"Aplanarity_{sufx}"][0] = A
                                if "Circularity" in tk: b[f"Circularity_{sufx}"][0] = C
                            if "Centrality" in tk: b[f"Centrality_{sufx}"][0] = Centrality(px_arr, py_arr, pz_arr, e_arr)
                            if "ScalarSumPT" in tk: b[f"ScalarSumPT_{sufx}"][0] = sum([p.Pt() for p in p4_list])


        # fill weight branches 
        w_gen = Weight_branch.At(0).Weight
        b["gen_weight"][0] = w_gen
        b["weight"][0] = eventWeight

        # Fill the trees
        trees[ac_key].Fill()

    # Print object selection cutflow & analysis channels yeilds
    if show_progress:
        objSel.PrintObjectSelectionSummary(lum=luminosity, event_weight=eventWeight)
        eventSel.PrintEventSelectionSummary(treeName, event_weight = eventWeight, lum = luminosity)
    

    # if output_dir is given, then write the trees into root files.
    root_paths = {}
    if output_dir is not None:
        # Write the trees into .root files at:
        # <output_dir>/<analysis_channel_name>/<analysis_region_name>/<output_root_file_name>.root
        for ac, ac_rgs in ac_dict.items():
            for ac_r in ac_rgs:
                ac_key = f"{ac}_{ac_r}"
                path = os.path.join(output_dir, ac, ac_r, output_file_name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                f_out = ROOT.TFile.Open(path, "RECREATE")
                trees[ac_key].SetDirectory(f_out)
                trees[ac_key].Write()
                # trees[ac_key].Delete()
                f_out.Close()
                root_paths[ac_key] = path

        # Write object-selection histograms + cutflow into a dedicated root
        if write_metadata:
            out_objsel = os.path.join(output_dir, "ObjectSelection")
            os.makedirs(out_objsel, exist_ok=True)
            f_objsel = ROOT.TFile.Open(os.path.join(out_objsel, output_file_name), "RECREATE")
            objSel.WriteObjectSelectionHistograms(f_objsel)
            objSel.WriteObjectSelectionSummary(f_objsel)
            f_objsel.Close()
            if show_progress:
                objSel.DrawObjectSelectionHistograms(output_dir = out_objsel)

            out_evtsel = os.path.join(output_dir, "EventSelection")
            os.makedirs(out_evtsel, exist_ok=True)
            f_evtsel = ROOT.TFile.Open(os.path.join(out_evtsel, output_file_name), "RECREATE")
            eventSel.WriteEventSelectionSummary(f_evtsel, treeName)
            f_evtsel.Close()
        
    # Only return TTree objects if explicitly requested (for standalone execution)
    # When running in parallel workers, return_trees is False to prevent PyROOT pickling crashes!
    return {
        "trees" : trees if return_trees else None,
        "root_paths" : root_paths,
        "ObjectSelector": objSel,
        "EventSelector": eventSel
    }


def main():
    from ..core.delphes_utilis import load_delphes
    from .loop_utilis          import run_loop_cli
    load_delphes()
    return run_loop_cli(loop_tree)

if __name__ == "__main__":

    main()
