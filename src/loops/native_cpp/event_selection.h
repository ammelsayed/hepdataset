#ifndef HEPDATASET_EVENT_SELECTION_H
#define HEPDATASET_EVENT_SELECTION_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <TFile.h>

#include "object_selection.h"

class EventSelector {
public:
    explicit EventSelector(bool split_by_flavour = false);

    static EventSelector Merge(const std::vector<EventSelector*>& selectors);

    std::string LeptonFlavour(const SelectedLepton& lepton) const;
    void GetChannelKeys();
    std::string ClassifyChannelKey(
        const std::vector<SelectedLepton>& good_leptons,
        const std::vector<Jet*>& good_fatjets
    ) const;
    std::string Select(
        const std::vector<SelectedLepton>& good_leptons,
        const std::vector<Jet*>& good_fatjets,
        const std::vector<std::string>& valid_keys = {}
    );
    void WriteEventSelectionSummary(TFile* root_file, const std::string& tree_name) const;

    bool splitByFlavour;
    std::vector<std::string> ac_keys;
    std::map<std::string, std::vector<std::string>> ac_dict;
    std::vector<std::pair<std::string, std::uint64_t>> ac_counts;
};

#endif
