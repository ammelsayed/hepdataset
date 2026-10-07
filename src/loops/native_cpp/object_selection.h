#ifndef HEPDATASET_OBJECT_SELECTION_H
#define HEPDATASET_OBJECT_SELECTION_H

#include <map>
#include <string>
#include <vector>

#include <TClonesArray.h>
#include <TH1D.h>
#include <TLorentzVector.h>
#include <TFile.h>

#include "classes/DelphesClasses.h"

struct SelectedLepton {
    bool is_electron;
    int charge;
    double pt;
    double eta;
    double phi;
    TLorentzVector p4;
};

struct SelectedObjects {
    std::vector<SelectedLepton> leptons;
    std::vector<Jet*> fat_jets;
    std::vector<Jet*> jets;
    std::vector<Jet*> b_jets;
    std::vector<Jet*> tau_jets;
};

class ObjectSelector {
public:
    ObjectSelector();

    SelectedObjects Select(
        TClonesArray* muons,
        TClonesArray* electrons,
        TClonesArray* fat_jets,
        TClonesArray* jets,
        double event_weight = 1.0
    );

    void WriteObjectSelectionHistograms(TFile* root_file) const;
    void WriteObjectSelectionSummary(TFile* root_file) const;

    std::map<std::string, std::map<std::string, unsigned long long>> cutflow;
    std::map<std::string, TH1D*> hist;
};

#endif
