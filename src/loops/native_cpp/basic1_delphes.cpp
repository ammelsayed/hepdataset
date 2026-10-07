#include "object_selection.h"
#include "event_selection.h"

// Compact selected-event tree: exactly one lepton and at least one fat jet.
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <TChain.h>
#include <TFile.h>
#include <TSystem.h>
#include <TTree.h>
#include "external/ExRootAnalysis/ExRootTreeReader.h"

int loop_tree(const std::string& input_root_file, const std::string& output_dir,
              const std::string& output_file_name, const std::string& tree_name,
              Long64_t start_entry, Long64_t end_entry, double event_weight, bool event_weight_set,
              double cross_section, double luminosity, bool overwrite, bool show_progress) {
    TChain chain("Delphes");
    chain.Add(input_root_file.c_str());
    ExRootTreeReader reader(&chain);
    const Long64_t entries = reader.GetEntries();
    if (end_entry < 0 || end_entry > entries) end_entry = entries;
    if (start_entry < 0 || start_entry > entries || end_entry < start_entry) {
        std::cerr << "Invalid entry range\n";
        return 2;
    }
    if (!event_weight_set) {
        if ((cross_section >= 0.0) != (luminosity >= 0.0)) {
            std::cerr << "cross section and luminosity must be provided together\n";
            return 2;
        }
        if (cross_section >= 0.0 && luminosity >= 0.0) {
            if (end_entry == start_entry) {
                std::cerr << "Cannot compute an event weight for an empty entry range\n";
                return 2;
            }
            event_weight = cross_section * luminosity / (end_entry - start_entry);
        } else {
            event_weight = 1.0;
        }
    }

    const std::filesystem::path output_path = std::filesystem::path(output_dir) / output_file_name;
    if (output_path.extension() != ".root") {
        std::cerr << "output_file_name must end with .root\n";
        return 2;
    }
    if (std::filesystem::exists(output_path) && !overwrite) {
        std::cerr << "Output file exists; pass --overwrite to replace it\n";
        return 2;
    }
    std::filesystem::create_directories(output_path.parent_path());

    TClonesArray* electrons = reader.UseBranch("Electron");
    TClonesArray* muons = reader.UseBranch("Muon");
    TClonesArray* fat_jets = reader.UseBranch("FatJet");
    TClonesArray* jets = reader.UseBranch("Jet");

    const std::vector<std::string> branch_names{
        "PT_Muon0", "Eta_Muon0", "Phi_Muon0",
        "PT_Electron0", "Eta_Electron0", "Phi_Electron0",
        "PT_FatJet0", "Eta_FatJet0", "Phi_FatJet0", "M_FatJet0",
        "PT_FatJet1", "Eta_FatJet1", "Phi_FatJet1", "M_FatJet1",
        "DeltaR_Lepton_FatJet0", "DeltaPhi_Lepton_FatJet0", "DeltaEta_Lepton_FatJet0",
        "DeltaR_Lepton_FatJet1", "DeltaPhi_Lepton_FatJet1", "DeltaEta_Lepton_FatJet1",
        "M_Lepton_FatJet0", "M_Lepton_FatJet1", "weight"
    };

    TFile output(output_path.c_str(), "RECREATE");
    TTree tree(tree_name.c_str(), tree_name.c_str());
    tree.SetDirectory(&output);
    std::map<std::string, double> values;
    for (const auto& name : branch_names) {
        values[name] = std::numeric_limits<double>::quiet_NaN();
        tree.Branch(name.c_str(), &values[name], (name + "/D").c_str());
    }

    ObjectSelector object_selector;
    EventSelector event_selector;
    const std::vector<std::string> selected_channels{"1L_lepJ", "1L_lepJJ"};
    for (Long64_t entry = start_entry; entry < end_entry; ++entry) {
        reader.ReadEntry(entry);
        const auto selected = object_selector.Select(muons, electrons, fat_jets, jets, event_weight);
        const auto& leptons = selected.leptons;
        const auto& good_fat_jets = selected.fat_jets;
        if (event_selector.Select(leptons, good_fat_jets, selected_channels).empty()) continue;

        for (const auto& name : branch_names) values[name] = std::numeric_limits<double>::quiet_NaN();
        values["weight"] = 0.0;
        const auto& lepton = leptons.front();
        const std::string lepton_type = lepton.is_electron ? "Electron" : "Muon";
        values["PT_" + lepton_type + "0"] = lepton.pt;
        values["Eta_" + lepton_type + "0"] = lepton.eta;
        values["Phi_" + lepton_type + "0"] = lepton.phi;

        for (std::size_t i = 0; i < good_fat_jets.size() && i < 2; ++i) {
            const auto* fat_jet = good_fat_jets[i];
            const std::string tag = "FatJet" + std::to_string(i);
            values["PT_" + tag] = fat_jet->PT;
            values["Eta_" + tag] = fat_jet->Eta;
            values["Phi_" + tag] = fat_jet->Phi;
            values["M_" + tag] = fat_jet->SoftDroppedP4[0].M();
            values["DeltaR_Lepton_" + tag] = lepton.p4.DeltaR(fat_jet->P4());
            values["DeltaPhi_Lepton_" + tag] = lepton.p4.DeltaPhi(fat_jet->P4());
            values["DeltaEta_Lepton_" + tag] = lepton.eta - fat_jet->Eta;
            values["M_Lepton_" + tag] = (fat_jet->P4() + lepton.p4).M();
        }
        values["weight"] = event_weight;
        tree.Fill();
        if (show_progress && (entry + 1) % 500 == 0)
            std::cout << "Processed " << (entry + 1 - start_entry) << "/" << (end_entry - start_entry) << " entries\r" << std::flush;
    }

    output.Write();
    output.Close();
    std::cout << "Wrote " << tree.GetEntries() << " selected entries to " << output_path << '\n';
    return 0;
}

int main(int argc, char** argv) {
    std::string input_root_file, output_dir = ".", output_file_name = "events.root", tree_name = "Delphes";
    Long64_t start_entry = 0, end_entry = -1;
    double event_weight = -1.0, cross_section = -1.0, luminosity = -1.0;
    bool overwrite = false, show_progress = false, event_weight_set = false;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--input-root-file") input_root_file = argv[++i];
        else if (option == "--output-dir") output_dir = argv[++i];
        else if (option == "--output-file-name") output_file_name = argv[++i];
        else if (option == "--tree-name") tree_name = argv[++i];
        else if (option == "--start-entry") start_entry = std::stoll(argv[++i]);
        else if (option == "--end-entry") end_entry = std::stoll(argv[++i]);
        else if (option == "--event-weight") { event_weight = std::stod(argv[++i]); event_weight_set = true; }
        else if (option == "--cross-section") cross_section = std::stod(argv[++i]);
        else if (option == "--luminosity") luminosity = std::stod(argv[++i]);
        else if (option == "--overwrite") overwrite = true;
        else if (option == "--show-progress") show_progress = true;
        else if (option == "--help") {
            std::cout << "Usage: basic1_delphes --input-root-file FILE [--output-dir DIR] [--start-entry N] [--end-entry N]\n";
            return 0;
        }
    }
    if (input_root_file.empty()) {
        std::cerr << "--input-root-file is required\n";
        return 2;
    }
    if (gSystem->Load("libDelphes") < 0) {
        std::cerr << "Could not load libDelphes; set LD_LIBRARY_PATH to the Delphes library directory\n";
        return 2;
    }
    return loop_tree(input_root_file, output_dir, output_file_name, tree_name,
                     start_entry, end_entry, event_weight, event_weight_set,
                     cross_section, luminosity, overwrite, show_progress);
}
