#include "object_selection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

ObjectSelector::ObjectSelector()
    : cutflow{{"lepton", {{"initial", 0}}}, {"fatjet", {{"initial", 0}}}} {
    auto book = [this](const char* name, const char* title, int bins, double low, double high) {
        hist[name] = new TH1D(name, title, bins, low, high);
        hist[name]->Sumw2();
        hist[name]->SetDirectory(nullptr);
    };

    book("h_nL", "; Lepton Multiplicity; Weighted Events", 6, -0.5, 5.5);
    book("h_nL_raw", "; Lepton Multiplicity; Weighted Events", 6, -0.5, 5.5);
    book("h_nL_baseline", "; Lepton Multiplicity; Events", 6, -0.5, 5.5);
    book("h_nL_pass_isolation", "; Lepton Multiplicity; Events", 6, -0.5, 5.5);
    book("h_nL_pass_em_olr", "; Lepton Multiplicity; Events", 6, -0.5, 5.5);
    book("h_nL_pass_fj_olr", "; Lepton Multiplicity; Events", 6, -0.5, 5.5);
    book("h_nL_pass_pTCut", "; Lepton Multiplicity; Events", 6, -0.5, 5.5);
    book("h_nJ", " FatJet Multiplicity; Weighted Events", 11, -0.5, 10.5);
    book("h_nJ_raw", "; FatJet Multiplicity; Weighted Events", 11, -0.5, 10.5);
    book("h_nJ_baseline", "; FatJet Multiplicity; Events", 11, -0.5, 10.5);
    book("h_nJ_pass_massCut", "; FatJet Multiplicity; Events", 11, -0.5, 10.5);
    book("h_nJ_pass_tau21Cut", "; FatJet Multiplicity; Events", 11, -0.5, 10.5);
    book("h_nJ_pass_pTCut", "; FatJet Multiplicity; Events", 11, -0.5, 10.5);
    book("h_nj", "; Jet Multiplicity; Weighted Events", 16, -0.5, 15.5);
    book("h_nj_raw", "; Jet Multiplicity; Weighted Events", 16, -0.5, 15.5);
}

SelectedObjects ObjectSelector::Select(
    TClonesArray* muons,
    TClonesArray* electrons,
    TClonesArray* fat_jets,
    TClonesArray* jets,
    double event_weight) {
    ++cutflow["lepton"]["initial"];
    ++cutflow["fatjet"]["initial"];
    hist["h_nL_raw"]->Fill(muons->GetEntries() + electrons->GetEntries(), event_weight);
    hist["h_nJ_raw"]->Fill(fat_jets->GetEntries(), event_weight);
    hist["h_nj_raw"]->Fill(jets->GetEntries(), event_weight);

    struct LeptonInput {
        SelectedLepton object;
        double isolation;
    };
    std::vector<LeptonInput> leptons;
    for (int i = 0; i < muons->GetEntries(); ++i) {
        auto* muon = static_cast<Muon*>(muons->At(i));
        if (std::fabs(muon->Eta) <= 2.7 && muon->PT > 10.0)
            leptons.push_back({{false, muon->Charge, muon->PT, muon->Eta, muon->Phi, muon->P4()}, muon->IsolationVar});
    }
    for (int i = 0; i < electrons->GetEntries(); ++i) {
        auto* electron = static_cast<Electron*>(electrons->At(i));
        if (std::fabs(electron->Eta) <= 2.7 && electron->PT > 10.0)
            leptons.push_back({{true, electron->Charge, electron->PT, electron->Eta, electron->Phi, electron->P4()}, electron->IsolationVar});
    }
    auto sort_leptons = [](auto& objects) {
        std::stable_sort(objects.begin(), objects.end(), [](const auto& a, const auto& b) {
            return a.object.pt > b.object.pt;
        });
    };
    sort_leptons(leptons);
    hist["h_nL_baseline"]->Fill(leptons.size(), event_weight);
    if (!leptons.empty()) ++cutflow["lepton"]["baseline"];

    std::vector<LeptonInput> isolated;
    for (const auto& lepton : leptons) {
        const double isolation_cut = lepton.object.is_electron ? 0.20 : 0.15;
        if (lepton.isolation < isolation_cut) isolated.push_back(lepton);
    }
    sort_leptons(isolated);
    hist["h_nL_pass_isolation"]->Fill(isolated.size(), event_weight);
    if (!isolated.empty()) ++cutflow["lepton"]["pass_isolation"];

    std::vector<bool> remove_lepton(isolated.size(), false);
    for (std::size_t i = 0; i < isolated.size(); ++i) {
        for (std::size_t j = i + 1; j < isolated.size(); ++j) {
            const auto& first = isolated[i].object;
            const auto& second = isolated[j].object;
            if (first.is_electron != second.is_electron && first.p4.DeltaR(second.p4) < 0.1)
                remove_lepton[first.pt > second.pt ? j : i] = true;
        }
    }
    std::vector<LeptonInput> em_clean;
    for (std::size_t i = 0; i < isolated.size(); ++i)
        if (!remove_lepton[i]) em_clean.push_back(isolated[i]);
    sort_leptons(em_clean);
    hist["h_nL_pass_em_olr"]->Fill(em_clean.size(), event_weight);
    if (!em_clean.empty()) ++cutflow["lepton"]["pass_em_olr"];

    std::vector<Jet*> selected_fatjets;
    for (int i = 0; i < fat_jets->GetEntries(); ++i) {
        auto* jet = static_cast<Jet*>(fat_jets->At(i));
        if (std::fabs(jet->Eta) <= 2.5 && jet->PT > 30.0) selected_fatjets.push_back(jet);
    }
    auto sort_jets = [](std::vector<Jet*>& objects) {
        std::stable_sort(objects.begin(), objects.end(), [](const Jet* a, const Jet* b) { return a->PT > b->PT; });
    };
    sort_jets(selected_fatjets);
    hist["h_nJ_baseline"]->Fill(selected_fatjets.size(), event_weight);
    if (!selected_fatjets.empty()) ++cutflow["fatjet"]["baseline"];

    selected_fatjets.erase(std::remove_if(selected_fatjets.begin(), selected_fatjets.end(), [](const Jet* jet) {
        const double mass = jet->SoftDroppedP4[0].M();
        return mass < 60.0 || mass > 145.0;
    }), selected_fatjets.end());
    hist["h_nJ_pass_massCut"]->Fill(selected_fatjets.size(), event_weight);
    if (!selected_fatjets.empty()) ++cutflow["fatjet"]["pass_massCut"];

    selected_fatjets.erase(std::remove_if(selected_fatjets.begin(), selected_fatjets.end(), [](const Jet* jet) {
        const double tau21 = jet->Tau[0] > 0.0 ? jet->Tau[1] / jet->Tau[0] : 1.0;
        return tau21 >= 0.7;
    }), selected_fatjets.end());
    hist["h_nJ_pass_tau21Cut"]->Fill(selected_fatjets.size(), event_weight);
    if (!selected_fatjets.empty()) ++cutflow["fatjet"]["pass_tau21Cut"];

    std::vector<bool> remove_overlap(em_clean.size(), false);
    for (std::size_t i = 0; i < em_clean.size(); ++i)
        for (const auto* jet : selected_fatjets)
            if (em_clean[i].object.p4.DeltaR(jet->P4()) < 1.0) remove_overlap[i] = true;
    std::vector<LeptonInput> overlap_clean;
    for (std::size_t i = 0; i < em_clean.size(); ++i)
        if (!remove_overlap[i]) overlap_clean.push_back(em_clean[i]);
    sort_leptons(overlap_clean);
    hist["h_nL_pass_fj_olr"]->Fill(overlap_clean.size(), event_weight);
    if (!overlap_clean.empty()) ++cutflow["lepton"]["pass_fj_olr"];

    std::vector<SelectedLepton> good_leptons;
    const std::array<double, 3> muon_pt_cuts{25.0, 20.0, 10.0};
    const std::array<double, 3> electron_pt_cuts{27.0, 20.0, 10.0};
    for (const auto& lepton : overlap_clean) {
        const int rank = std::min(static_cast<int>(good_leptons.size()), 2);
        const auto& pt_cuts = lepton.object.is_electron ? electron_pt_cuts : muon_pt_cuts;
        if (lepton.object.pt > pt_cuts[rank]) good_leptons.push_back(lepton.object);
    }
    hist["h_nL_pass_pTCut"]->Fill(good_leptons.size(), event_weight);
    if (!good_leptons.empty()) ++cutflow["lepton"]["pass_pTCut"];

    std::vector<Jet*> good_fatjets;
    const std::array<double, 3> fatjet_pt_cuts{200.0, 150.0, 30.0};
    for (auto* jet : selected_fatjets) {
        const int rank = std::min(static_cast<int>(good_fatjets.size()), 2);
        if (jet->PT > fatjet_pt_cuts[rank]) good_fatjets.push_back(jet);
    }
    hist["h_nJ_pass_pTCut"]->Fill(good_fatjets.size(), event_weight);
    if (!good_fatjets.empty()) ++cutflow["fatjet"]["pass_pTCut"];

    return {good_leptons, good_fatjets, {}, {}, {}};
}

void ObjectSelector::WriteObjectSelectionHistograms(TFile* root_file) const {
    root_file->cd();
    for (const auto& [name, histogram] : hist) histogram->Write();
}

void ObjectSelector::WriteObjectSelectionSummary(TFile* root_file) const {
    root_file->cd();
    const std::array<std::pair<const char*, std::vector<const char*>>, 2> stages{{
        {"lepton", {"initial", "baseline", "pass_isolation", "pass_em_olr", "pass_fj_olr", "pass_pTCut"}},
        {"fatjet", {"initial", "baseline", "pass_massCut", "pass_tau21Cut", "pass_pTCut"}}
    }};
    for (const auto& [object, names] : stages) {
        std::vector<const char*> present;
        for (const char* name : names)
            if (cutflow.at(object).count(name)) present.push_back(name);
        TH1D cutflow_hist((std::string("cutflow_") + object).c_str(),
                          (std::string(object) + " selection cutflow").c_str(),
                          present.size(), 0.5, present.size() + 0.5);
        for (std::size_t i = 0; i < present.size(); ++i) {
            cutflow_hist.SetBinContent(i + 1, cutflow.at(object).at(present[i]));
            cutflow_hist.GetXaxis()->SetBinLabel(i + 1, present[i]);
        }
        cutflow_hist.Write();
    }
}
