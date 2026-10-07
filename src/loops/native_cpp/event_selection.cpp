#include "event_selection.h"

#include <algorithm>
#include <cstdlib>

#include <TH1D.h>

EventSelector::EventSelector(bool split_by_flavour)
    : splitByFlavour(split_by_flavour) {
    GetChannelKeys();
    ac_counts.emplace_back("initial", 0);
    for (const auto& key : ac_keys) ac_counts.emplace_back(key, 0);
    ac_counts.emplace_back("dropped", 0);
}

EventSelector EventSelector::Merge(const std::vector<EventSelector*>& selectors) {
    EventSelector merged(selectors.empty() ? false : selectors.front()->splitByFlavour);
    for (const auto* selector : selectors) {
        for (auto& [key, count] : merged.ac_counts) {
            const auto found = std::find_if(selector->ac_counts.begin(), selector->ac_counts.end(),
                [&key](const auto& item) { return item.first == key; });
            if (found != selector->ac_counts.end()) count += found->second;
        }
    }
    return merged;
}

std::string EventSelector::LeptonFlavour(const SelectedLepton& lepton) const {
    if (!splitByFlavour) return "lep";
    return lepton.is_electron ? "e" : "mu";
}

void EventSelector::GetChannelKeys() {
    if (splitByFlavour) {
        ac_dict = {
            {"0L", {"Nothing", "J", "JJ", "JJJ"}},
            {"1L", {"e", "mu", "eJ", "muJ", "eJJ", "muJJ"}},
            {"2OSL", {"ee", "emu", "mumu", "eeJ", "emuJ", "mumuJ"}},
            {"2SSL", {"ee", "emu", "mumu", "eeJ", "emuJ", "mumuJ"}},
            {"3L", {"eee", "eemu", "emumu", "mumumu"}}
        };
    } else {
        ac_dict = {
            {"0L", {"Nothing", "J", "JJ", "JJJ"}},
            {"1L", {"lep", "lepJ", "lepJJ"}},
            {"2OSL", {"leplep", "leplepJ"}},
            {"2SSL", {"leplep", "leplepJ"}},
            {"3L", {"lepleplep"}}
        };
    }

    ac_keys.clear();
    for (const auto& [region, categories] : ac_dict)
        for (const auto& category : categories)
            ac_keys.push_back(region + "_" + category);
}

std::string EventSelector::ClassifyChannelKey(
    const std::vector<SelectedLepton>& good_leptons,
    const std::vector<Jet*>& good_fatjets) const {
    const auto n_leptons = good_leptons.size();
    const auto n_fatjets = good_fatjets.size();

    if (n_leptons == 0) {
        if (n_fatjets == 0) return "0L_Nothing";
        if (n_fatjets == 1) return "0L_J";
        if (n_fatjets == 2) return "0L_JJ";
        return "0L_JJJ";
    }

    if (n_leptons == 1) {
        const auto flavour = LeptonFlavour(good_leptons[0]);
        if (n_fatjets == 0) return "1L_" + flavour;
        if (n_fatjets == 1) return "1L_" + flavour + "J";
        return "1L_" + flavour + "JJ";
    }

    if (n_leptons == 2) {
        const auto flavour1 = LeptonFlavour(good_leptons[0]);
        const auto flavour2 = LeptonFlavour(good_leptons[1]);
        const auto flavours = std::min(flavour1, flavour2) + std::max(flavour1, flavour2);
        const int charge_sum = std::abs(good_leptons[0].charge + good_leptons[1].charge);
        const std::string region = charge_sum == 0 ? "2OSL" : charge_sum == 2 ? "2SSL" : "";
        if (region.empty()) return "";
        return region + "_" + flavours + (n_fatjets == 0 ? "" : "J");
    }

    std::vector<std::string> flavours;
    for (std::size_t i = 0; i < 3; ++i) flavours.push_back(LeptonFlavour(good_leptons[i]));
    std::sort(flavours.begin(), flavours.end());
    std::string channel = "3L_";
    for (const auto& flavour : flavours) channel += flavour;
    return channel;
}

std::string EventSelector::Select(
    const std::vector<SelectedLepton>& good_leptons,
    const std::vector<Jet*>& good_fatjets,
    const std::vector<std::string>& valid_keys) {
    ++ac_counts.front().second;
    const auto key = ClassifyChannelKey(good_leptons, good_fatjets);
    const auto& allowed = valid_keys.empty() ? ac_keys : valid_keys;
    if (key.empty() || std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
        ++ac_counts.back().second;
        return "";
    }

    const auto found = std::find_if(ac_counts.begin(), ac_counts.end(), [&key](const auto& item) {
        return item.first == key;
    });
    ++found->second;
    return key;
}

void EventSelector::WriteEventSelectionSummary(TFile* root_file, const std::string& tree_name) const {
    root_file->cd();
    TH1D cutflow(("cutflow_" + tree_name).c_str(), ("analysis channels (" + tree_name + ")").c_str(),
                 ac_counts.size(), 0.5, ac_counts.size() + 0.5);
    for (std::size_t i = 0; i < ac_counts.size(); ++i) {
        cutflow.SetBinContent(i + 1, ac_counts[i].second);
        cutflow.GetXaxis()->SetBinLabel(i + 1, ac_counts[i].first.c_str());
    }
    cutflow.Write();
}
