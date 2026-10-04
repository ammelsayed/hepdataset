// Native C++ implementation of HEPDataset's configurable adaptive Delphes loop.
// Event reading, object selection, event-channel selection, kinematic filling,
// and ROOT output all execute in C++; Python is not embedded or called.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "TChain.h"
#include "TClonesArray.h"
#include "TFile.h"
#include "TH1D.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "TObject.h"
#include "TROOT.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "vendor/mt2_bisect.h"

namespace fs = std::filesystem;
namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;

struct ObjectSpec {
  std::string name;
  int count = 0;
  std::vector<std::string> representations;
  std::vector<std::string> kinematics;
};

struct Manifest {
  std::map<std::string, ObjectSpec> objects;
  std::vector<std::string> globalScalars, eventShapes, comboSet, mt2ObjectTypes;
  std::vector<std::string> mt2ObjectInstances;
  std::vector<std::string> basicKinematics, twoBodyKinematics, trivalKinematics;
  std::vector<std::string> intBranches, floatBranches;
  std::map<int, std::vector<std::string>> nbodyKinematics;
  std::map<int, std::vector<std::vector<std::string>>> combos;
  int nmax = 2;
  bool includeSameRepresentations = false;
  bool includeTrival = false;
  bool includeMt2 = false;
};

std::vector<std::string> split(const std::string& value, char delimiter) {
  std::vector<std::string> out;
  if (value.empty()) return out;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, delimiter)) if (!item.empty()) out.push_back(item);
  return out;
}

std::vector<std::string> fields(const std::string& line) { return split(line, '\t'); }

Manifest readManifest(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open native branch manifest: " + path);
  Manifest m;
  std::string line;
  bool header = false;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    auto f = fields(line);
    if (f.empty()) continue;
    if (f[0] == "HEPDATASET_NATIVE_CONFIG") {
      if (f.size() != 2 || f[1] != "1") throw std::runtime_error("Unsupported native manifest version");
      header = true;
    } else if (f[0] == "OBJECT" && f.size() == 5) {
      ObjectSpec spec;
      spec.name = f[1]; spec.count = std::stoi(f[2]);
      spec.representations = split(f[3], ','); spec.kinematics = split(f[4], ',');
      m.objects[spec.name] = std::move(spec);
    } else if (f[0] == "GLOBAL_SCALARS" && f.size() == 2) m.globalScalars = split(f[1], ',');
    else if (f[0] == "EVENT_SHAPES" && f.size() == 2) m.eventShapes = split(f[1], ',');
    else if (f[0] == "MULTI_OBJECTS" && f.size() == 5) {
      m.nmax = std::stoi(f[1]); m.includeSameRepresentations = std::stoi(f[2]) != 0;
      m.includeTrival = std::stoi(f[3]) != 0; m.includeMt2 = std::stoi(f[4]) != 0;
    } else if (f[0] == "COMBO_SET" && f.size() == 2) m.comboSet = split(f[1], ',');
    else if (f[0] == "MT2_OBJECT_TYPES" && f.size() == 2) m.mt2ObjectTypes = split(f[1], ',');
    else if (f[0] == "MT2_OBJECT_INSTANCES" && f.size() == 2) m.mt2ObjectInstances = split(f[1], ',');
    else if (f[0] == "BASIC_KINEMATICS" && f.size() == 2) m.basicKinematics = split(f[1], ',');
    else if (f[0] == "TWO_BODY_KINEMATICS" && f.size() == 2) m.twoBodyKinematics = split(f[1], ',');
    else if (f[0] == "TRIVAL_KINEMATICS" && f.size() == 2) m.trivalKinematics = split(f[1], ',');
    else if (f[0] == "INT_BRANCHES" && f.size() == 2) m.intBranches = split(f[1], ',');
    else if (f[0] == "FLOAT_BRANCHES" && f.size() == 2) m.floatBranches = split(f[1], ',');
    else if (f[0] == "NBODY_KINEMATICS" && f.size() == 3) m.nbodyKinematics[std::stoi(f[1])] = split(f[2], ',');
    else if (f[0] == "COMBO" && f.size() == 3) m.combos[std::stoi(f[1])].push_back(split(f[2], ','));
    else throw std::runtime_error("Malformed native manifest line: " + line);
  }
  if (!header) throw std::runtime_error("Missing HEPDATASET_NATIVE_CONFIG header");
  if (m.intBranches.empty() || m.floatBranches.empty()) throw std::runtime_error("Manifest has no output branches");
  return m;
}

bool has(const std::vector<std::string>& values, const std::string& item) {
  return std::find(values.begin(), values.end(), item) != values.end();
}

double deltaPhi(double a, double b) {
  double d = a - b;
  while (d > kPi) d -= 2.0 * kPi;
  while (d < -kPi) d += 2.0 * kPi;
  return d;
}

double mtW(double pt1, double phi1, double pt2, double phi2) {
  return std::sqrt(std::max(0.0, 2.0 * pt1 * pt2 * (1.0 - std::cos(deltaPhi(phi1, phi2)))));
}

double tauValue(const Jet* jet, int index) { return index >= 1 && index <= 5 ? jet->Tau[index - 1] : kNaN; }

double fourVectorValue(const TLorentzVector& p, const std::string& k) {
  if (k == "X") return p.X(); if (k == "Y") return p.Y(); if (k == "Z") return p.Z();
  if (k == "Px") return p.Px(); if (k == "Py") return p.Py(); if (k == "Pz") return p.Pz();
  if (k == "Pt" || k == "Perp") return p.Pt(); if (k == "P") return p.P();
  if (k == "E" || k == "Energy") return p.Energy(); if (k == "Theta") return p.Theta();
  if (k == "CosTheta") return p.CosTheta(); if (k == "Phi") return p.Phi();
  if (k == "Rho") return p.Rho(); if (k == "Perp2") return p.Perp2();
  if (k == "Et") return p.Et(); if (k == "Et2") return p.Et2();
  if (k == "Mag") return p.Mag(); if (k == "Mag2") return p.Mag2();
  if (k == "M") return p.M(); if (k == "M2") return p.M2();
  if (k == "Mt") return p.Mt(); if (k == "Mt2") return p.Mt2();
  if (k == "Beta") return p.Beta(); if (k == "Gamma") return p.Gamma();
  if (k == "Plus") return p.Plus(); if (k == "Minus") return p.Minus();
  if (k == "Rapidity") return p.Rapidity(); if (k == "Eta" || k == "PseudoRapidity") return p.Eta();
  return kNaN;
}

double objectValue(TObject* object, const TLorentzVector& p4, const std::string& k,
                   const std::string& objectType) {
  const std::string cls = object ? object->ClassName() : "";
  if (k == "ElectronTag") return cls == "Electron" ? 1.0 : 0.0;
  if (k == "MuonTag") return cls == "Muon" ? 1.0 : 0.0;
  if (auto* e = dynamic_cast<Electron*>(object)) {
    if (k == "PT") return e->PT; if (k == "Eta") return e->Eta; if (k == "Phi") return e->Phi;
    if (k == "T") return e->T; if (k == "Charge") return e->Charge;
    if (k == "IsolationVar") return e->IsolationVar; if (k == "IsolationVarRhoCorr") return e->IsolationVarRhoCorr;
    if (k == "SumPtCharged") return e->SumPtCharged; if (k == "SumPtNeutral") return e->SumPtNeutral;
    if (k == "SumPtChargedPU") return e->SumPtChargedPU; if (k == "SumPt") return e->SumPt;
    if (k == "D0") return e->D0; if (k == "DZ") return e->DZ;
    if (k == "ErrorD0") return e->ErrorD0; if (k == "ErrorDZ") return e->ErrorDZ;
  }
  if (auto* mu = dynamic_cast<Muon*>(object)) {
    if (k == "PT") return mu->PT; if (k == "Eta") return mu->Eta; if (k == "Phi") return mu->Phi;
    if (k == "T") return mu->T; if (k == "Charge") return mu->Charge;
    if (k == "IsolationVar") return mu->IsolationVar; if (k == "IsolationVarRhoCorr") return mu->IsolationVarRhoCorr;
    if (k == "SumPtCharged") return mu->SumPtCharged; if (k == "SumPtNeutral") return mu->SumPtNeutral;
    if (k == "SumPtChargedPU") return mu->SumPtChargedPU; if (k == "SumPt") return mu->SumPt;
    if (k == "D0") return mu->D0; if (k == "DZ") return mu->DZ;
    if (k == "ErrorD0") return mu->ErrorD0; if (k == "ErrorDZ") return mu->ErrorDZ;
  }
  if (auto* j = dynamic_cast<Jet*>(object)) {
    if (k == "PT") return j->PT; if (k == "Eta") return j->Eta; if (k == "Phi") return j->Phi;
    if (k == "T") return j->T; if (k == "Mass") return j->Mass; if (k == "DeltaEta") return j->DeltaEta;
    if (k == "DeltaPhi") return j->DeltaPhi; if (k == "Flavor") return j->Flavor;
    if (k == "FlavorAlgo") return j->FlavorAlgo; if (k == "FlavorPhys") return j->FlavorPhys;
    if (k == "TauFlavor") return j->TauFlavor; if (k == "BTag") return j->BTag;
    if (k == "BTagAlgo") return j->BTagAlgo; if (k == "BTagPhys") return j->BTagPhys;
    if (k == "TauTag") return j->TauTag; if (k == "TauWeight") return j->TauWeight;
    if (k == "Charge") return j->Charge; if (k == "EhadOverEem") return j->EhadOverEem;
    if (k == "NCharged") return j->NCharged; if (k == "NNeutrals") return j->NNeutrals;
    if (k == "NeutralEnergyFraction") return j->NeutralEnergyFraction;
    if (k == "ChargedEnergyFraction") return j->ChargedEnergyFraction;
    if (k == "Beta") return j->Beta; if (k == "BetaStar") return j->BetaStar;
    if (k == "MeanSqDeltaR") return j->MeanSqDeltaR; if (k == "NSubJetsTrimmed") return j->NSubJetsTrimmed;
    if (k == "NSubJetsPruned") return j->NSubJetsPruned; if (k == "NSubJetsSoftDropped") return j->NSubJetsSoftDropped;
    if (k == "ExclYmerge12") return j->ExclYmerge12; if (k == "ExclYmerge23") return j->ExclYmerge23;
    if (k == "ExclYmerge34") return j->ExclYmerge34; if (k == "ExclYmerge45") return j->ExclYmerge45;
    if (k == "ExclYmerge56") return j->ExclYmerge56;
    if (k == "Tau1") return tauValue(j, 1); if (k == "Tau2") return tauValue(j, 2);
    if (k == "Tau3") return tauValue(j, 3); if (k == "Tau4") return tauValue(j, 4);
    if (k == "Tau5") return tauValue(j, 5);
    const std::map<std::string, std::pair<int, int>> ratios = {
      {"Tau21", {2,1}}, {"Tau31", {3,1}}, {"Tau32", {3,2}}, {"Tau41", {4,1}},
      {"Tau42", {4,2}}, {"Tau43", {4,3}}, {"Tau51", {5,1}}, {"Tau52", {5,2}},
      {"Tau53", {5,3}}, {"Tau54", {5,4}}
    };
    auto ratio = ratios.find(k);
    if (ratio != ratios.end()) {
      const double den = tauValue(j, ratio->second.second);
      return den > 0 ? tauValue(j, ratio->second.first) / den : 1.0;
    }
  }
  if (auto* met = dynamic_cast<MissingET*>(object)) {
    if (k == "MET") return met->MET; if (k == "Eta") return met->Eta; if (k == "Phi") return met->Phi;
  }
  if (auto* ht = dynamic_cast<ScalarHT*>(object)) if (k == "HT") return ht->HT;
  (void)objectType;
  return fourVectorValue(p4, k);
}

struct ObjectCounts {
  std::map<std::string, std::map<std::string, Long64_t>> stages;
  std::map<std::string, std::vector<std::string>> stageOrder;
  std::map<std::string, std::unique_ptr<TH1D>> hist;

  ObjectCounts() {
    stages["lepton"] = {{"initial",0},{"baseline",0},{"pass_isolation",0},{"pass_em_olr",0},{"pass_fj_olr",0},{"pass_pTCut",0}};
    stages["fatjet"] = {{"initial",0},{"baseline",0},{"pass_massCut",0},{"pass_tau21Cut",0},{"pass_pTCut",0}};
    stageOrder["lepton"] = {"initial","baseline","pass_isolation","pass_em_olr","pass_fj_olr","pass_pTCut"};
    stageOrder["fatjet"] = {"initial","baseline","pass_massCut","pass_tau21Cut","pass_pTCut"};
    const std::vector<std::tuple<std::string,std::string,int,double,double>> definitions = {
      {"h_nL","; Lepton Multiplicity; Weighted Events",6,-0.5,5.5},
      {"h_nL_raw","; Lepton Multiplicity; Weighted Events",6,-0.5,5.5},
      {"h_nL_baseline","; Lepton Multiplicity; Events",6,-0.5,5.5},
      {"h_nL_pass_isolation","; Lepton Multiplicity; Events",6,-0.5,5.5},
      {"h_nL_pass_em_olr","; Lepton Multiplicity; Events",6,-0.5,5.5},
      {"h_nL_pass_fj_olr","; Lepton Multiplicity; Events",6,-0.5,5.5},
      {"h_nL_pass_pTCut","; Lepton Multiplicity; Events",6,-0.5,5.5},
      {"h_nJ","; FatJet Multiplicity; Weighted Events",11,-0.5,10.5},
      {"h_nJ_raw","; FatJet Multiplicity; Weighted Events",11,-0.5,10.5},
      {"h_nJ_baseline","; FatJet Multiplicity; Events",11,-0.5,10.5},
      {"h_nJ_pass_massCut","; FatJet Multiplicity; Events",11,-0.5,10.5},
      {"h_nJ_pass_tau21Cut","; FatJet Multiplicity; Events",11,-0.5,10.5},
      {"h_nJ_pass_pTCut","; FatJet Multiplicity; Events",11,-0.5,10.5},
      {"h_nj","; Jet Multiplicity; Weighted Events",16,-0.5,15.5},
      {"h_nj_raw","; Jet Multiplicity; Weighted Events",16,-0.5,15.5}
    };
    for (const auto& d : definitions) {
      auto h = std::make_unique<TH1D>(std::get<0>(d).c_str(), std::get<1>(d).c_str(), std::get<2>(d), std::get<3>(d), std::get<4>(d));
      h->Sumw2(); h->SetDirectory(nullptr); hist.emplace(std::get<0>(d), std::move(h));
    }
  }
};

struct EventCounts {
  std::vector<std::string> keys;
  std::map<std::string, Long64_t> counts;
  bool splitByFlavour = false;
  explicit EventCounts(bool splitFlavour) : splitByFlavour(splitFlavour) {
    const std::map<std::string, std::vector<std::string>> regions = splitByFlavour ?
      std::map<std::string,std::vector<std::string>>{{"0L",{"Nothing","J","JJ","JJJ"}},
       {"1L",{"e","mu","eJ","muJ","eJJ","muJJ"}},
       {"2OSL",{"ee","emu","mumu","eeJ","emuJ","mumuJ"}},
       {"2SSL",{"ee","emu","mumu","eeJ","emuJ","mumuJ"}},
       {"3L",{"eee","eemu","emumu","mumumu"}}} :
      std::map<std::string,std::vector<std::string>>{{"0L",{"Nothing","J","JJ","JJJ"}},
       {"1L",{"lep","lepJ","lepJJ"}}, {"2OSL",{"leplep","leplepJ"}},
       {"2SSL",{"leplep","leplepJ"}}, {"3L",{"lepleplep"}}};
    for (const auto& [ac, rs] : regions) for (const auto& r : rs) keys.push_back(ac + "_" + r);
    counts["initial"] = 0;
    for (const auto& k : keys) counts[k] = 0;
    counts["dropped"] = 0;
  }
  std::string flavour(const TObject* o) const {
    if (!splitByFlavour) return "lep";
    return std::string(o && std::string(o->ClassName()) == "Electron" ? "e" : "mu");
  }
  std::string classify(const std::vector<TObject*>& leptons, const std::vector<Jet*>& fatjets) const {
    const int nl = static_cast<int>(leptons.size()), nj = static_cast<int>(fatjets.size());
    if (nl == 0) return "0L_" + std::string(nj == 0 ? "Nothing" : nj == 1 ? "J" : nj == 2 ? "JJ" : "JJJ");
    if (nl == 1) {
      const std::string f = flavour(leptons[0]);
      return "1L_" + f + (nj == 0 ? "" : nj == 1 ? "J" : "JJ");
    }
    if (nl == 2) {
      auto* a = dynamic_cast<Electron*>(leptons[0]);
      auto* b = dynamic_cast<Electron*>(leptons[1]);
      const int qa = a ? a->Charge : dynamic_cast<Muon*>(leptons[0])->Charge;
      const int qb = b ? b->Charge : dynamic_cast<Muon*>(leptons[1])->Charge;
      const int q = std::abs(qa + qb);
      const std::string ac = q == 0 ? "2OSL" : q == 2 ? "2SSL" : "";
      if (ac.empty()) return "";
      std::string fs = "leplep";
      if (splitByFlavour) {
        std::vector<std::string> flavours{flavour(leptons[0]), flavour(leptons[1])};
        std::sort(flavours.begin(), flavours.end());
        fs = flavours[0] + flavours[1];
      }
      return ac + "_" + fs + (nj == 0 ? "" : "J");
    }
    std::string fs = "lepleplep";
    if (splitByFlavour) {
      std::vector<std::string> flavours{flavour(leptons[0]), flavour(leptons[1]), flavour(leptons[2])};
      std::sort(flavours.begin(), flavours.end());
      fs = flavours[0] + flavours[1] + flavours[2];
    }
    return "3L_" + fs;
  }
  std::string select(const std::vector<TObject*>& leptons, const std::vector<Jet*>& fatjets) {
    counts["initial"]++;
    const std::string key = classify(leptons, fatjets);
    if (key.empty() || !counts.count(key)) { counts["dropped"]++; return ""; }
    counts[key]++;
    return key;
  }
};

struct Buffers {
  std::map<std::string, int> ints;
  std::map<std::string, double> floats;
  void book(TTree* tree, const Manifest& m) {
    for (const auto& name : m.intBranches) { ints[name] = -1; tree->Branch(name.c_str(), &ints[name], (name + "/I").c_str()); }
    for (const auto& name : m.floatBranches) { floats[name] = kNaN; tree->Branch(name.c_str(), &floats[name], (name + "/D").c_str()); }
  }
  void reset(const Manifest& m) {
    for (const auto& name : m.intBranches) ints[name] = -1;
    for (const auto& name : m.floatBranches) floats[name] = kNaN;
    if (floats.count("weight")) floats["weight"] = 0.0;
    if (floats.count("gen_weight")) floats["gen_weight"] = 0.0;
  }
  void set(const std::string& name, double value) {
    if (floats.count(name)) floats[name] = value;
    else if (ints.count(name) && std::isfinite(value)) ints[name] = static_cast<int>(value);
  }
  void setInt(const std::string& name, int value) {
    if (ints.count(name)) ints[name] = value;
  }
};

struct Args {
  std::string input, manifest, outputDir = ".", outputName = "events.root", treeName = "Delphes";
  Long64_t start = 0, end = -1;
  double eventWeight = 1.0, luminosity = 400.0;
  bool splitByFlavour = false, showProgress = false, overwrite = false, writeMetadata = true, countEntries = false;
};

void printUsage() {
  std::cout << "Usage: native_adaptive_delphes --input-root-file FILE --branch-manifest FILE [options]\n"
            << "       native_adaptive_delphes --count-entries FILE\n"
            << "  --output-dir DIR          Output directory (default: .)\n"
            << "  --output-file-name FILE   Output ROOT basename (default: events.root)\n"
            << "  --tree-name NAME          Output tree name (default: Delphes)\n"
            << "  --start-entry N           First zero-based entry, inclusive (default: 0)\n"
            << "  --end-entry N             Last boundary, exclusive (default: all entries)\n"
            << "  --event-weight X          Analysis event-weight multiplier (default: 1)\n"
            << "  --luminosity FB-1         Luminosity for selector summaries (default: 400)\n"
            << "  --split-by-flavour        Split channels by electron/muon flavour\n"
            << "  --show-progress           Print progress and summaries\n"
            << "  --overwrite               Permit an existing output directory\n"
            << "  --no-write-metadata       Do not write ObjectSelection/EventSelection ROOT files\n";
}

Args parseArgs(int argc, char** argv) {
  Args a;
  auto value = [&](int& i, const std::string& flag) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error("Missing value after " + flag);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--help" || flag == "-h") { printUsage(); std::exit(0); }
    else if (flag == "--count-entries") { a.countEntries = true; a.input = value(i, flag); }
    else if (flag.rfind("--", 0) != 0 && a.input.empty()) a.input = flag;
    else if (flag == "--input-root-file") a.input = value(i, flag);
    else if (flag == "--branch-manifest") a.manifest = value(i, flag);
    else if (flag == "--output-dir") a.outputDir = value(i, flag);
    else if (flag == "--output-file-name") a.outputName = value(i, flag);
    else if (flag == "--tree-name") a.treeName = value(i, flag);
    else if (flag == "--start-entry") a.start = std::stoll(value(i, flag));
    else if (flag == "--end-entry") a.end = std::stoll(value(i, flag));
    else if (flag == "--event-weight") a.eventWeight = std::stod(value(i, flag));
    else if (flag == "--luminosity") a.luminosity = std::stod(value(i, flag));
    else if (flag == "--split-by-flavour") a.splitByFlavour = true;
    else if (flag == "--show-progress") a.showProgress = true;
    else if (flag == "--overwrite") a.overwrite = true;
    else if (flag == "--no-write-metadata") a.writeMetadata = false;
    else throw std::runtime_error("Unknown argument: " + flag);
  }
  if (a.input.empty() || (!a.countEntries && a.manifest.empty())) throw std::runtime_error("--input-root-file and --branch-manifest are required");
  if (a.start < 0 || (a.end >= 0 && a.end < a.start)) throw std::runtime_error("Invalid entry range");
  if (a.outputName.size() < 5 || a.outputName.substr(a.outputName.size()-5) != ".root") throw std::runtime_error("--output-file-name must end in .root");
  return a;
}

std::vector<TObject*> eventLeptons(TClonesArray* muons, TClonesArray* electrons) {
  std::vector<TObject*> out;
  for (int i = 0; i < muons->GetEntriesFast(); ++i) out.push_back(muons->At(i));
  for (int i = 0; i < electrons->GetEntriesFast(); ++i) out.push_back(electrons->At(i));
  std::stable_sort(out.begin(), out.end(), [](TObject* a, TObject* b) {
    auto pt = [](TObject* o) { if (auto* e = dynamic_cast<Electron*>(o)) return e->PT; return dynamic_cast<Muon*>(o)->PT; };
    return pt(a) > pt(b);
  });
  return out;
}

double leptonPt(TObject* o) { if (auto* e = dynamic_cast<Electron*>(o)) return e->PT; return dynamic_cast<Muon*>(o)->PT; }
double leptonEta(TObject* o) { if (auto* e = dynamic_cast<Electron*>(o)) return e->Eta; return dynamic_cast<Muon*>(o)->Eta; }
double leptonPhi(TObject* o) { if (auto* e = dynamic_cast<Electron*>(o)) return e->Phi; return dynamic_cast<Muon*>(o)->Phi; }
double leptonIsolation(TObject* o) { if (auto* e = dynamic_cast<Electron*>(o)) return e->IsolationVar; return dynamic_cast<Muon*>(o)->IsolationVar; }
int leptonCharge(TObject* o) { if (auto* e = dynamic_cast<Electron*>(o)) return e->Charge; return dynamic_cast<Muon*>(o)->Charge; }
TLorentzVector selectionP4(TObject* o) { if (auto* e = dynamic_cast<Electron*>(o)) return e->P4(); return dynamic_cast<Muon*>(o)->P4(); }

struct SelectedObjects {
  std::vector<TObject*> leptons;
  std::vector<Jet*> fatjets, jets, bjets, taujets;
};

SelectedObjects selectObjects(TClonesArray* muons, TClonesArray* electrons, TClonesArray* fatjetBranch,
                              TClonesArray* jetBranch, double weight, ObjectCounts& out) {
  ++out.stages["lepton"]["initial"];
  ++out.stages["fatjet"]["initial"];
  out.hist.at("h_nL_raw")->Fill(muons->GetEntriesFast() + electrons->GetEntriesFast(), weight);
  out.hist.at("h_nJ_raw")->Fill(fatjetBranch->GetEntriesFast(), weight);
  out.hist.at("h_nj_raw")->Fill(jetBranch->GetEntriesFast(), weight);

  auto leptons = eventLeptons(muons, electrons);
  std::vector<TObject*> stage;
  for (auto* l : leptons) if (std::fabs(leptonEta(l)) <= 2.7 && leptonPt(l) > 10.0) stage.push_back(l);
  auto fillCount = [](TH1D* h, std::size_t n, double w) { h->Fill(static_cast<double>(n), w); };
  fillCount(out.hist.at("h_nL_baseline").get(), stage.size(), weight);
  if (!stage.empty()) ++out.stages["lepton"]["baseline"];

  leptons.clear();
  for (auto* l : stage) {
    const std::string cls = l->ClassName();
    if ((cls == "Muon" && leptonIsolation(l) < 0.15) || (cls == "Electron" && leptonIsolation(l) < 0.20)) leptons.push_back(l);
  }
  std::stable_sort(leptons.begin(), leptons.end(), [](TObject* a, TObject* b) { return leptonPt(a) > leptonPt(b); });
  fillCount(out.hist.at("h_nL_pass_isolation").get(), leptons.size(), weight);
  if (!leptons.empty()) ++out.stages["lepton"]["pass_isolation"];

  std::set<std::size_t> removeLeptons;
  for (std::size_t i = 0; i < leptons.size(); ++i) for (std::size_t j = i + 1; j < leptons.size(); ++j) {
    if (std::string(leptons[i]->ClassName()) == std::string(leptons[j]->ClassName())) continue;
    if (selectionP4(leptons[i]).DeltaR(selectionP4(leptons[j])) < 0.1) removeLeptons.insert(leptonPt(leptons[i]) > leptonPt(leptons[j]) ? j : i);
  }
  stage.clear();
  for (std::size_t i = 0; i < leptons.size(); ++i) if (!removeLeptons.count(i)) stage.push_back(leptons[i]);
  leptons = stage;
  fillCount(out.hist.at("h_nL_pass_em_olr").get(), leptons.size(), weight);
  if (!leptons.empty()) ++out.stages["lepton"]["pass_em_olr"];

  std::vector<Jet*> fatjets;
  for (int i = 0; i < fatjetBranch->GetEntriesFast(); ++i) {
    auto* j = dynamic_cast<Jet*>(fatjetBranch->At(i));
    if (j && std::fabs(j->Eta) <= 2.5 && j->PT > 30.0) fatjets.push_back(j);
  }
  std::stable_sort(fatjets.begin(), fatjets.end(), [](Jet* a, Jet* b) { return a->PT > b->PT; });
  fillCount(out.hist.at("h_nJ_baseline").get(), fatjets.size(), weight);
  if (!fatjets.empty()) ++out.stages["fatjet"]["baseline"];

  std::vector<Jet*> stageJets;
  for (auto* j : fatjets) { const double mass = j->SoftDroppedP4[0].M(); if (60.0 <= mass && mass <= 145.0) stageJets.push_back(j); }
  fatjets = stageJets;
  std::stable_sort(fatjets.begin(), fatjets.end(), [](Jet* a, Jet* b) { return a->PT > b->PT; });
  fillCount(out.hist.at("h_nJ_pass_massCut").get(), fatjets.size(), weight);
  if (!fatjets.empty()) ++out.stages["fatjet"]["pass_massCut"];

  stageJets.clear();
  for (auto* j : fatjets) { const double tau21 = j->Tau[0] > 0.0 ? j->Tau[1] / j->Tau[0] : 1.0; if (tau21 < 0.7) stageJets.push_back(j); }
  fatjets = stageJets;
  std::stable_sort(fatjets.begin(), fatjets.end(), [](Jet* a, Jet* b) { return a->PT > b->PT; });
  fillCount(out.hist.at("h_nJ_pass_tau21Cut").get(), fatjets.size(), weight);
  if (!fatjets.empty()) ++out.stages["fatjet"]["pass_tau21Cut"];

  std::set<std::size_t> removeForFatjet;
  for (std::size_t i = 0; i < leptons.size(); ++i) for (auto* fj : fatjets)
    if (selectionP4(leptons[i]).DeltaR(fj->P4()) < 1.0) { removeForFatjet.insert(i); break; }
  stage.clear();
  for (std::size_t i = 0; i < leptons.size(); ++i) if (!removeForFatjet.count(i)) stage.push_back(leptons[i]);
  leptons = stage;
  std::stable_sort(leptons.begin(), leptons.end(), [](TObject* a, TObject* b) { return leptonPt(a) > leptonPt(b); });
  fillCount(out.hist.at("h_nL_pass_fj_olr").get(), leptons.size(), weight);
  if (!leptons.empty()) ++out.stages["lepton"]["pass_fj_olr"];

  stage.clear();
  for (auto* l : leptons) {
    const int rank = static_cast<int>(stage.size());
    const std::string cls = l->ClassName();
    const double minPt = rank == 0 ? (cls == "Muon" ? 25.0 : 27.0) : rank == 1 ? 20.0 : 10.0;
    if (leptonPt(l) > minPt) stage.push_back(l);
  }
  leptons = stage;
  fillCount(out.hist.at("h_nL_pass_pTCut").get(), leptons.size(), weight);
  if (!leptons.empty()) ++out.stages["lepton"]["pass_pTCut"];

  stageJets.clear();
  for (auto* j : fatjets) {
    const int rank = static_cast<int>(stageJets.size());
    const double minPt = rank == 0 ? 200.0 : rank == 1 ? 150.0 : 30.0;
    if (j->PT > minPt) stageJets.push_back(j);
  }
  fatjets = stageJets;
  fillCount(out.hist.at("h_nJ_pass_pTCut").get(), fatjets.size(), weight);
  if (!fatjets.empty()) ++out.stages["fatjet"]["pass_pTCut"];

  SelectedObjects result;
  result.leptons = std::move(leptons); result.fatjets = std::move(fatjets);
  return result;
}

void fillObjectCounts(const Manifest& m, Buffers& b, TClonesArray* muons, TClonesArray* electrons,
                      TClonesArray* fatjets, TClonesArray* jets) {
  const int nMuon = muons->GetEntriesFast(), nElectron = electrons->GetEntriesFast();
  b.setInt("nPreQS_Muon", nMuon); b.setInt("nPreQS_Electron", nElectron);
  b.setInt("nPreQS_Lepton", nMuon + nElectron); b.setInt("nPreQS_FatJet", fatjets->GetEntriesFast());
  b.setInt("nPreQS_Jet", jets->GetEntriesFast());
  (void)m;
}

void fillLepton(const Manifest& m, Buffers& b, std::map<std::string, TLorentzVector>& p4s,
                std::map<std::string, int>& charges, std::map<std::string, int>& flavours,
                const std::vector<TObject*>& leptons) {
  auto it = m.objects.find("Lepton"); if (it == m.objects.end()) return;
  const auto& spec = it->second;
  const int n = std::min<int>(leptons.size(), spec.count);
  for (int i = 0; i < n; ++i) {
    TObject* l = leptons[i]; const std::string cls = l->ClassName();
    const double mass = cls == "Muon" ? 0.1057 : 0.000511;
    TLorentzVector p; p.SetPtEtaPhiM(leptonPt(l), leptonEta(l), leptonPhi(l), mass);
    const int charge = leptonCharge(l), flavour = cls == "Muon" ? 2 : 1;
    std::vector<std::string> reps{"Lepton"}; if (has(spec.representations, cls)) reps.push_back(cls);
    for (const auto& rep : reps) {
      const std::string inst = rep + std::to_string(i); p4s[inst] = p; charges[inst] = charge; flavours[inst] = flavour;
      for (const auto& k : spec.kinematics) b.set(k + "_" + inst, objectValue(l, p, k, "Lepton"));
    }
  }
}

void fillFatJets(const Manifest& m, Buffers& b, std::map<std::string, TLorentzVector>& p4s,
                 const std::vector<Jet*>& jets) {
  auto it = m.objects.find("FatJet"); if (it == m.objects.end()) return;
  const auto& spec = it->second; const int n = std::min<int>(jets.size(), spec.count);
  for (int i = 0; i < n; ++i) {
    Jet* j = jets[i]; const int slot = i;
    const TLorentzVector p = j->P4(); p4s["FatJet" + std::to_string(slot)] = p;
    if (has(spec.representations, "SoftDroppedFatJet")) p4s["SoftDroppedFatJet" + std::to_string(slot)] = j->SoftDroppedP4[0];
    if (has(spec.representations, "TrimmedFatJet")) p4s["TrimmedFatJet" + std::to_string(slot)] = j->TrimmedP4[0];
    if (has(spec.representations, "PrunedFatJet")) p4s["PrunedFatJet" + std::to_string(slot)] = j->PrunedP4[0];
    for (const auto& k : spec.kinematics) {
      b.set(k + "_FatJet" + std::to_string(slot), objectValue(j, p, k, "FatJet"));
      if (has(spec.representations, "SoftDroppedFatJet")) b.set(k + "_SoftDroppedFatJet" + std::to_string(slot), fourVectorValue(j->SoftDroppedP4[0], k));
      if (has(spec.representations, "TrimmedFatJet")) b.set(k + "_TrimmedFatJet" + std::to_string(slot), fourVectorValue(j->TrimmedP4[0], k));
      if (has(spec.representations, "PrunedFatJet")) b.set(k + "_PrunedFatJet" + std::to_string(slot), fourVectorValue(j->PrunedP4[0], k));
    }
  }
}

void fillSmallJets(const Manifest& m, Buffers& b, std::map<std::string, TLorentzVector>& p4s,
                   const SelectedObjects& selected) {
  auto it = m.objects.find("Jet"); if (it == m.objects.end()) return;
  const auto& spec = it->second;
  const std::vector<std::pair<std::string,const std::vector<Jet*>*>> lists = {{"Jet",&selected.jets},{"BJet",&selected.bjets},{"TauJet",&selected.taujets}};
  for (const auto& [rep, jets] : lists) {
    if (!has(spec.representations, rep)) continue;
    for (int i = 0; i < std::min<int>(jets->size(), spec.count); ++i) {
      Jet* j = (*jets)[i]; const std::string inst = rep + std::to_string(i); const TLorentzVector p = j->P4();
      p4s[inst] = p; for (const auto& k : spec.kinematics) b.set(k + "_" + inst, objectValue(j, p, k, "Jet"));
    }
  }
}

void fillMetAndHT(const Manifest& m, Buffers& b, std::map<std::string, TLorentzVector>& p4s,
                  TClonesArray* metBranch, TClonesArray* scalarHtBranch) {
  if (metBranch->GetEntriesFast() > 0) {
    auto* met = dynamic_cast<MissingET*>(metBranch->At(0)); const TLorentzVector p = met->P4(); p4s["MET"] = p;
    auto it = m.objects.find("MET"); if (it != m.objects.end()) for (const auto& k : it->second.kinematics) b.set(k + "_MET", objectValue(met, p, k, "MET"));
  }
  if (scalarHtBranch->GetEntriesFast() > 0) {
    auto* ht = dynamic_cast<ScalarHT*>(scalarHtBranch->At(0)); const TLorentzVector p(ht->HT, 0.0, 0.0, ht->HT);
    auto it = m.objects.find("ScalarHT"); if (it != m.objects.end()) for (const auto& k : it->second.kinematics) b.set(k + "_ScalarHT", objectValue(ht, p, k, "ScalarHT"));
  }
}

void fillGlobalScalars(const Manifest& m, Buffers& b, const SelectedObjects& o,
                       MissingET* met, ScalarHT* scalarHt) {
  const double ht = scalarHt ? scalarHt->HT : kNaN;
  double lt=0.0, sumJet=0.0, sumFat=0.0, sumSoft=0.0;
  for (auto* l : o.leptons) lt += leptonPt(l);
  for (auto* j : o.jets) sumJet += j->PT;
  for (auto* j : o.fatjets) { sumFat += j->PT; sumSoft += j->SoftDroppedP4[0].Pt(); }
  const double metpt = met ? met->MET : kNaN, st = lt + ht;
  for (const auto& name : m.globalScalars) {
    if (name == "HT") b.set(name, ht); else if (name == "LT") b.set(name, lt);
    else if (name == "ST") b.set(name, st); else if (name == "Meff") b.set(name, st + metpt);
    else if (name == "Significance_MET") b.set(name, ht > 0.0 ? metpt / std::sqrt(ht) : kNaN);
    else if (name == "ScalarSumPT_Jets") b.set(name, sumJet);
    else if (name == "ScalarSumPT_FatJets") b.set(name, sumFat);
    else if (name == "ScalarSumPT_SoftDroppedFatJets") b.set(name, sumSoft);
    else if (name == "ScalarSumPT_Hadronic") b.set(name, sumJet + sumSoft);
  }
}

void eventShapes(const std::vector<TLorentzVector>& p4s, double& sph, double& apl, double& circ, double& centrality) {
  double xx=0, yy=0, zz=0, xy=0, xz=0, yz=0, denom=0, pxx=0, pyy=0, pxy=0, ptden=0, sumpt=0, sumE=0;
  for (const auto& p : p4s) {
    const double x=p.Px(), y=p.Py(), z=p.Pz(), pt2=x*x+y*y, p2=pt2+z*z;
    xx+=x*x; yy+=y*y; zz+=z*z; xy+=x*y; xz+=x*z; yz+=y*z; denom+=p2;
    pxx+=x*x; pyy+=y*y; pxy+=x*y; ptden+=pt2; sumpt+=std::sqrt(pt2); sumE+=p.Energy();
  }
  sph=apl=circ=centrality=kNaN;
  if (denom > 0) {
    TMatrixDSym matrix(3); matrix(0,0)=xx/denom; matrix(1,1)=yy/denom; matrix(2,2)=zz/denom;
    matrix(0,1)=matrix(1,0)=xy/denom; matrix(0,2)=matrix(2,0)=xz/denom; matrix(1,2)=matrix(2,1)=yz/denom;
    TMatrixDSymEigen eig(matrix); auto vals=eig.GetEigenValues(); std::array<double,3> e{vals[0],vals[1],vals[2]}; std::sort(e.begin(),e.end(),std::greater<double>());
    sph=1.5*(e[1]+e[2]); apl=1.5*e[2];
  }
  if (ptden > 0) { const double tr=(pxx+pyy)/ptden, det=(pxx*pyy-pxy*pxy)/(ptden*ptden); const double disc=std::sqrt(std::max(0.0,tr*tr-4*det)); const double low=(tr-disc)/2, high=(tr+disc)/2; if (high+low>0) circ=2*low/(high+low); }
  if (sumE > 0) centrality=sumpt/sumE;
}

void fillEventShapes(const Manifest& m, Buffers& b, const SelectedObjects& o, MissingET* met) {
  if (m.eventShapes.empty()) return;
  std::vector<TLorentzVector> vis;
  for (auto* l : o.leptons) vis.push_back(selectionP4(l));
  for (auto* j : o.fatjets) vis.push_back(j->P4());
  for (auto* j : o.jets) vis.push_back(j->P4());
  if (met) vis.push_back(met->P4());
  double sph,apl,circ,cent; eventShapes(vis,sph,apl,circ,cent);
  if (has(m.eventShapes,"Sphericity")) b.set("Sphericity",sph);
  if (has(m.eventShapes,"Aplanarity")) b.set("Aplanarity",apl);
  if (has(m.eventShapes,"Circularity")) b.set("Circularity",circ);
  if (has(m.eventShapes,"Centrality")) b.set("Centrality",cent);
}

void fillNBody(const Manifest& m, Buffers& b, const std::map<std::string,TLorentzVector>& p4s,
               const std::map<std::string,int>& charges, const std::map<std::string,int>& flavours,
               MissingET* met) {
  for (const auto& [n, combos] : m.combos) for (const auto& combo : combos) {
    bool complete=true; std::vector<TLorentzVector> legs;
    for (const auto& name : combo) { auto it=p4s.find(name); if (it==p4s.end()) { complete=false; break; } legs.push_back(it->second); }
    if (!complete) continue;
    const std::string suffix = [&](){ std::string s; for (const auto& p : combo) { if (!s.empty()) s += "_"; s += p; } return s; }();
    TLorentzVector total=legs.front(); for (std::size_t i=1;i<legs.size();++i) total+=legs[i];
    for (const auto& k : m.nbodyKinematics.at(n)) {
      const std::string branch=k+"_"+suffix;
      if (k=="DeltaR" || k=="DeltaPhi" || k=="DeltaEta" || k=="MtW" || k.rfind("is",0)==0 || k=="MT2") continue;
      const double v = (k=="ScalarSumPT" || k=="VectorSumPT" || k=="Sphericity" || k=="Aplanarity" || k=="Circularity" || k=="Centrality") ? kNaN : fourVectorValue(total,k);
      b.set(branch,v);
    }
    if (n==2) {
      const auto& a=legs[0]; const auto& c=legs[1];
      if (has(m.twoBodyKinematics,"DeltaR")) b.set("DeltaR_"+suffix,a.DeltaR(c));
      if (has(m.twoBodyKinematics,"DeltaPhi")) b.set("DeltaPhi_"+suffix,a.DeltaPhi(c));
      if (has(m.twoBodyKinematics,"DeltaEta")) b.set("DeltaEta_"+suffix,a.Eta()-c.Eta());
      if (has(m.twoBodyKinematics,"MtW") && has(combo,"MET")) b.set("MtW_"+suffix,mtW(a.Pt(),a.Phi(),c.Pt(),c.Phi()));
      if (combo[0].find("Lepton")==0 && combo[1].find("Lepton")==0 && charges.count(combo[0]) && charges.count(combo[1])) {
        const int qprod=charges.at(combo[0])*charges.at(combo[1]); const bool same=flavours.at(combo[0])==flavours.at(combo[1]);
        if (has(m.twoBodyKinematics,"isOSSF")) b.set("isOSSF_"+suffix,qprod<0 && same ? 1:0);
        if (has(m.twoBodyKinematics,"isOSOF")) b.set("isOSOF_"+suffix,qprod<0 && !same ? 1:0);
        if (has(m.twoBodyKinematics,"isSSOF")) b.set("isSSOF_"+suffix,qprod>0 && !same ? 1:0);
        if (has(m.twoBodyKinematics,"isSSSF")) b.set("isSSSF_"+suffix,qprod>0 && same ? 1:0);
      }
      if (m.includeMt2 && met && std::all_of(combo.begin(),combo.end(),[&](const std::string& name){ return has(m.mt2ObjectInstances,name); })) {
        const double value=mt2_bisect_impl<double>(a.M(),a.Px(),a.Py(),c.M(),c.Px(),c.Py(),met->P4().Px(),met->P4().Py(),0.0,0.0);
        b.set("MT2_"+suffix,value);
      }
    }
    if (m.includeTrival) {
      double sph,apl,circ,cent; eventShapes(legs,sph,apl,circ,cent);
      double sumpt=0, sx=0, sy=0; for (const auto& p:legs) { sumpt+=p.Pt(); sx+=p.Px(); sy+=p.Py(); }
      if (has(m.trivalKinematics,"Sphericity")) b.set("Sphericity_"+suffix,sph);
      if (has(m.trivalKinematics,"Aplanarity")) b.set("Aplanarity_"+suffix,apl);
      if (has(m.trivalKinematics,"Circularity")) b.set("Circularity_"+suffix,circ);
      if (has(m.trivalKinematics,"Centrality")) b.set("Centrality_"+suffix,cent);
      if (has(m.trivalKinematics,"ScalarSumPT")) b.set("ScalarSumPT_"+suffix,sumpt);
      if (has(m.trivalKinematics,"VectorSumPT")) b.set("VectorSumPT_"+suffix,std::sqrt(sx*sx+sy*sy));
    }
  }
}

void writeObjectMetadata(const std::string& outputDir, const std::string& name, const ObjectCounts& counts) {
  fs::create_directories(outputDir); const std::string path=(fs::path(outputDir)/name).string();
  TFile f(path.c_str(),"RECREATE");
  for (const auto& [n,h] : counts.hist) h->Write();
  for (const auto& [obj,stages] : counts.stages) {
    TH1D h(("cutflow_"+obj).c_str(),(obj+" selection cutflow").c_str(),stages.size(),0.5,stages.size()+0.5);
    std::vector<std::string> populated;
    for (const auto& stage : counts.stageOrder.at(obj))
      if (stage == "initial" || stages.at(stage) > 0) populated.push_back(stage);
    h.SetBins(populated.size(), 0.5, populated.size() + 0.5);
    int i=1; for (const auto& stage : populated) { h.SetBinContent(i,stages.at(stage)); h.GetXaxis()->SetBinLabel(i,stage.c_str()); ++i; }
    h.Write();
  }
  f.Close();
}
void writeEventMetadata(const std::string& outputDir, const std::string& name, const std::string& treeName, const EventCounts& counts) {
  fs::create_directories(outputDir); const std::string path=(fs::path(outputDir)/name).string();
  TFile f(path.c_str(),"RECREATE");
  std::vector<std::string> stages{"initial"}; stages.insert(stages.end(),counts.keys.begin(),counts.keys.end()); stages.push_back("dropped");
  TH1D h(("cutflow_"+treeName).c_str(),("analysis channels ("+treeName+")").c_str(),stages.size(),0.5,stages.size()+0.5);
  for (std::size_t i=0;i<stages.size();++i) { h.SetBinContent(i+1,counts.counts.at(stages[i])); h.GetXaxis()->SetBinLabel(i+1,stages[i].c_str()); }
  h.Write(); f.Close();
}

int run(const Args& a) {
  const Manifest m=readManifest(a.manifest);
  if (a.outputDir.empty()) throw std::runtime_error("--output-dir cannot be empty");
  if (fs::exists(a.outputDir) && !a.overwrite) throw std::runtime_error("Output directory exists; pass --overwrite to allow writing there: "+a.outputDir);
  fs::create_directories(a.outputDir);

  TChain chain("Delphes");
  if (chain.Add(a.input.c_str()) == 0) throw std::runtime_error("No ROOT files matched input: "+a.input);
  ExRootTreeReader reader(&chain);
  const Long64_t total=reader.GetEntries();
  const Long64_t start=a.start, end=a.end<0 ? total : std::min(a.end,total);
  if (start>total || end<start) throw std::runtime_error("Entry range outside input tree");
  if (a.showProgress) std::cout<<"Input entries: "<<total<<"; processing ["<<start<<", "<<end<<") with event weight "<<a.eventWeight<<"\n";

  TClonesArray* branchElectron=reader.UseBranch("Electron");
  TClonesArray* branchMuon=reader.UseBranch("Muon");
  TClonesArray* branchFatJet=reader.UseBranch("FatJet");
  TClonesArray* branchJet=reader.UseBranch("Jet");
  TClonesArray* branchMet=reader.UseBranch("MissingET");
  TClonesArray* branchHt=reader.UseBranch("ScalarHT");
  TClonesArray* branchWeight=reader.UseBranch("Weight");

  EventCounts eventCounts(a.splitByFlavour); ObjectCounts objectCounts;
  std::map<std::string,std::unique_ptr<TTree>> trees;
  std::map<std::string,std::unique_ptr<Buffers>> buffers;
  for (const auto& key:eventCounts.keys) {
    auto tree=std::make_unique<TTree>(a.treeName.c_str(),a.treeName.c_str()); tree->SetDirectory(nullptr);
    auto b=std::make_unique<Buffers>(); b->book(tree.get(),m); buffers[key]=std::move(b); trees[key]=std::move(tree);
  }

  for (Long64_t entry=start; entry<end; ++entry) {
    reader.ReadEntry(entry);
    SelectedObjects selected=selectObjects(branchMuon,branchElectron,branchFatJet,branchJet,a.eventWeight,objectCounts);
    const std::string acKey=eventCounts.select(selected.leptons,selected.fatjets);
    if (acKey.empty()) continue;
    Buffers& b=*buffers.at(acKey); b.reset(m);
    fillObjectCounts(m,b,branchMuon,branchElectron,branchFatJet,branchJet);
    std::map<std::string,TLorentzVector> p4s; std::map<std::string,int> charges,flavours;
    fillLepton(m,b,p4s,charges,flavours,selected.leptons);
    fillFatJets(m,b,p4s,selected.fatjets);
    fillSmallJets(m,b,p4s,selected);
    fillMetAndHT(m,b,p4s,branchMet,branchHt);
    auto* met=branchMet->GetEntriesFast()>0 ? dynamic_cast<MissingET*>(branchMet->At(0)) : nullptr;
    auto* ht=branchHt->GetEntriesFast()>0 ? dynamic_cast<ScalarHT*>(branchHt->At(0)) : nullptr;
    fillGlobalScalars(m,b,selected,met,ht);
    fillEventShapes(m,b,selected,met);
    fillNBody(m,b,p4s,charges,flavours,met);
    b.set("weight",a.eventWeight);
    if (branchWeight->GetEntriesFast()>0) b.set("gen_weight",dynamic_cast<Weight*>(branchWeight->At(0))->Weight);
    trees.at(acKey)->Fill();
    if (a.showProgress && (entry+1)%50000==0) std::cout<<"Processed "<<(entry+1)<<" / "<<end<<" entries\n";
  }

  std::map<std::string,std::string> channelRegions;
  for (const auto& key:eventCounts.keys) {
    auto pos=key.find('_');
    std::string ac=key.substr(0,pos), region=key.substr(pos+1);
    const fs::path path=fs::path(a.outputDir)/ac/region/a.outputName;
    fs::create_directories(path.parent_path());
    TFile f(path.string().c_str(),"RECREATE"); f.cd(); trees.at(key)->Write(); trees.at(key)->SetDirectory(nullptr); f.Close();
    channelRegions[key]=path.string();
  }
  if (a.writeMetadata) {
    writeObjectMetadata((fs::path(a.outputDir)/"ObjectSelection").string(),a.outputName,objectCounts);
    writeEventMetadata((fs::path(a.outputDir)/"EventSelection").string(),a.outputName,a.treeName,eventCounts);
  }
  if (a.showProgress) {
    std::cout<<"Processed range ["<<start<<", "<<end<<"); output written under "<<a.outputDir<<"\n";
    std::cout<<"Event-channel counts:\n";
    for (const auto& [key,value]:eventCounts.counts) std::cout<<"  "<<key<<": "<<value<<"\n";
  }
  return 0;
}
} // namespace

int main(int argc, char** argv) {
  try {
    const Args args=parseArgs(argc,argv);
    if (args.countEntries) {
      TChain chain("Delphes");
      if (chain.Add(args.input.c_str()) == 0) throw std::runtime_error("No ROOT files matched input: " + args.input);
      std::cout << "TOTAL_ENTRIES=" << chain.GetEntries() << "\n";
      return 0;
    }
    return run(args);
  }
  catch (const std::exception& e) { std::cerr<<"native_adaptive_delphes: "<<e.what()<<"\n"; return 2; }
}
