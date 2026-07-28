/*
 * Delphes event extraction to ROOT TTree
 * Reads Muons and FatJets from Delphes file and saves to new tree
 */

#include <iostream>
#include "TChain.h"
#include "TClonesArray.h"
#include "TTree.h"
#include "TFile.h"


R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"

using namespace std;

TTree* extractDelphesData(const char *inputFile)
{

  // Create chain of root trees
  TChain chain("Delphes");
  chain.Add(inputFile);

  // Create object of class ExRootTreeReader
  ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
  Long64_t numberOfEntries = treeReader->GetEntries();

  cout << "Number of entries: " << numberOfEntries << endl;

  // Get pointers to branches used in this analysis
  TClonesArray *branchMuon = treeReader->UseBranch("Muon");
  TClonesArray *branchFatJet = treeReader->UseBranch("FatJet");

  // Create output tree
  TTree *outputTree = new TTree("test", "Extracted Delphes Data");

  // Branch variables
  Int_t Number_Muon;
  vector<Float_t> *Pt_Muon = new vector<Float_t>();
  vector<Float_t> *Eta_Muon = new vector<Float_t>();
  vector<Float_t> *Phi_Muon = new vector<Float_t>();

  Int_t Number_FatJet;
  vector<Float_t> *Pt_FatJet = new vector<Float_t>();
  vector<Float_t> *Eta_FatJet = new vector<Float_t>();
  vector<Float_t> *Phi_FatJet = new vector<Float_t>();

  // Create branches in output tree
  outputTree->Branch("Number_Muon", &Number_Muon);
  outputTree->Branch("Pt_Muon", "vector<float>", &Pt_Muon);
  outputTree->Branch("Eta_Muon", "vector<float>", &Eta_Muon);
  outputTree->Branch("Phi_Muon", "vector<float>", &Phi_Muon);

  outputTree->Branch("Number_FatJet", &Number_FatJet);
  outputTree->Branch("Pt_FatJet", "vector<float>", &Pt_FatJet);
  outputTree->Branch("Eta_FatJet", "vector<float>", &Eta_FatJet);
  outputTree->Branch("Phi_FatJet", "vector<float>", &Phi_FatJet);

  // Loop over all events
  for(Int_t entry = 0; entry < numberOfEntries; ++entry)
  {
    // Load selected branches with data from specified event
    treeReader->ReadEntry(entry);

    if((entry + 1) % 100000 == 0)
    {
      cout << "Processed " << (entry + 1) << " events..." << endl;
    }

    // Clear vectors for this event
    Pt_Muon->clear();
    Eta_Muon->clear();
    Phi_Muon->clear();

    Pt_FatJet->clear();
    Eta_FatJet->clear();
    Phi_FatJet->clear();

    // Process Muons
    Number_Muon = branchMuon->GetEntries();
    for(Int_t i = 0; i < branchMuon->GetEntries(); ++i)
    {
      Muon *muon = (Muon*) branchMuon->At(i);
      Pt_Muon->push_back(muon->PT);
      Eta_Muon->push_back(muon->Eta);
      Phi_Muon->push_back(muon->Phi);
    }

    // Process FatJets
    Number_FatJet = branchFatJet->GetEntries();
    for(Int_t i = 0; i < branchFatJet->GetEntries(); ++i)
    {
      Jet *fatjet = (Jet*) branchFatJet->At(i);
      Pt_FatJet->push_back(fatjet->PT);
      Eta_FatJet->push_back(fatjet->Eta);
      Phi_FatJet->push_back(fatjet->Phi);
    }

    // Fill the tree with data from this event
    outputTree->Fill();
  }

  cout << "Event extraction complete. Tree has " << outputTree->GetEntries() << " entries." << endl;

  delete treeReader;

  return outputTree;
}