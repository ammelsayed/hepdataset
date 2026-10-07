#include "object_selection.h"

// Flat tree for every input event; unlike basic3, output is not split by channel.

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
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
    TClonesArray* missing_et = reader.UseBranch("MissingET");
    TClonesArray* scalar_ht = reader.UseBranch("ScalarHT");
    TClonesArray* weights = reader.UseBranch("Weight");

    std::vector<std::string> branch_names;
    for (const auto& kinematic : {"PT", "Eta", "Phi"})
        for (int i = 0; i < 3; ++i) branch_names.push_back(std::string(kinematic) + "_Lepton" + std::to_string(i));
    for (const auto& kinematic : {"PT", "Eta", "Phi", "Mass"})
        for (int i = 0; i < 2; ++i) branch_names.push_back(std::string(kinematic) + "_FatJet" + std::to_string(i));

    std::vector<std::string> objects{"Lepton0", "Lepton1", "Lepton2", "FatJet0", "FatJet1"};
    const std::vector<std::string> pair_kinematics{"DeltaR", "DeltaPhi", "DeltaEta", "M"};
    for (const auto& kinematic : pair_kinematics)
        for (std::size_t i = 0; i < objects.size(); ++i)
            for (std::size_t j = i + 1; j < objects.size(); ++j)
                branch_names.push_back(kinematic + "_" + objects[i] + "_" + objects[j]);
    for (const auto& name : {"MET", "HT", "LT", "ST", "Meff", "weight", "gen_weight"})
        branch_names.emplace_back(name);

    TFile output(output_path.c_str(), "RECREATE");
    TTree tree(tree_name.c_str(), tree_name.c_str());
    tree.SetDirectory(&output);
    std::map<std::string, double> values;
    for (const auto& name : branch_names) {
        values[name] = std::numeric_limits<double>::quiet_NaN();
        tree.Branch(name.c_str(), &values[name], (name + "/D").c_str());
    }

    ObjectSelector object_selector;
    for (Long64_t entry = start_entry; entry < end_entry; ++entry) {
        reader.ReadEntry(entry);
        const auto selected = object_selector.Select(muons, electrons, fat_jets, jets, event_weight);
        const auto& leptons = selected.leptons;
        const auto& good_fat_jets = selected.fat_jets;
        for (const auto& name : branch_names) values[name] = std::numeric_limits<double>::quiet_NaN();
        values["weight"] = 0.0;

        std::map<std::string, TLorentzVector> p4s;
        for (std::size_t i = 0; i < leptons.size() && i < 3; ++i) {
            const auto& lepton = leptons[i];
            const std::string name = "Lepton" + std::to_string(i);
            p4s[name] = lepton.p4;
            values["PT_" + name] = lepton.pt;
            values["Eta_" + name] = lepton.eta;
            values["Phi_" + name] = lepton.phi;
        }
        for (std::size_t i = 0; i < good_fat_jets.size() && i < 2; ++i) {
            const auto* fat_jet = good_fat_jets[i];
            const std::string name = "FatJet" + std::to_string(i);
            p4s[name] = fat_jet->P4();
            values["PT_" + name] = fat_jet->PT;
            values["Eta_" + name] = fat_jet->Eta;
            values["Phi_" + name] = fat_jet->Phi;
            values["Mass_" + name] = fat_jet->SoftDroppedP4[0].M();
        }

        for (std::size_t i = 0; i < objects.size(); ++i) {
            for (std::size_t j = i + 1; j < objects.size(); ++j) {
                const auto first = p4s.find(objects[i]);
                const auto second = p4s.find(objects[j]);
                if (first == p4s.end() || second == p4s.end()) continue;
                const std::string suffix = objects[i] + "_" + objects[j];
                values["DeltaR_" + suffix] = first->second.DeltaR(second->second);
                values["DeltaPhi_" + suffix] = first->second.DeltaPhi(second->second);
                values["DeltaEta_" + suffix] = first->second.Eta() - second->second.Eta();
                values["M_" + suffix] = (first->second + second->second).M();
            }
        }

        const double met = static_cast<MissingET*>(missing_et->At(0))->MET;
        const double ht = static_cast<ScalarHT*>(scalar_ht->At(0))->HT;
        double lt = 0.0;
        for (const auto& lepton : leptons) lt += lepton.pt;
        values["MET"] = met;
        values["HT"] = ht;
        values["LT"] = lt;
        values["ST"] = ht + lt;
        values["Meff"] = ht + lt + met;
        values["gen_weight"] = static_cast<Weight*>(weights->At(0))->Weight;
        values["weight"] = event_weight;
        tree.Fill();
        if (show_progress && (entry + 1) % 500 == 0)
            std::cout << "Processed " << (entry + 1 - start_entry) << "/" << (end_entry - start_entry) << " entries\r" << std::flush;
    }

    output.Write();
    output.Close();
    std::cout << "Wrote " << tree.GetEntries() << " entries to " << output_path << '\n';
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
            std::cout << "Usage: basic2_delphes --input-root-file FILE [--output-dir DIR] [--start-entry N] [--end-entry N]\n";
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
