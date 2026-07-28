#!/usr/bin/env python3
"""
Main script to extract Delphes data using C++ and analyze with histograms
"""

import ROOT
import sys

ROOT.gROOT.SetBatch(True)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <delphes_root_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Load ROOT libraries
    ROOT.gSystem.Load("libDelphes")
    
    # Load the C++ file
    print("Loading C++ extraction code...")
    ROOT.gROOT.ProcessLine('.L extract_delphes_Version2.cpp+')
    
    # Call the C++ function to extract data
    print(f"Extracting data from {input_file}...")
    tree = ROOT.extractDelphesData(input_file)

    print("\nTree structure (Print):")
    tree.Print()

    print("\nList of branches with their types:")
    branches = tree.GetListOfBranches()
    for b in branches:
        print(f"  {b.GetName()}  ->  {b.GetClassName()}")
    
    # Create histograms from the extracted tree
    print("\nCreating histograms...")
    
    # Muon histograms
    hist_muon_pt = ROOT.TH1D("muon_pt", "Muon P_{T}", 100, 0.0, 100.0)
    hist_muon_eta = ROOT.TH1D("muon_eta", "Muon #eta", 50, -2.5, 2.5)
    hist_muon_phi = ROOT.TH1D("muon_phi", "Muon #phi", 50, -3.14, 3.14)
    hist_muon_count = ROOT.TH1D("muon_count", "Number of Muons per Event", 10, 0, 10)
    
    # FatJet histograms
    hist_fatjet_pt = ROOT.TH1D("fatjet_pt", "FatJet P_{T}", 100, 0.0, 500.0)
    hist_fatjet_eta = ROOT.TH1D("fatjet_eta", "FatJet #eta", 50, -2.5, 2.5)
    hist_fatjet_phi = ROOT.TH1D("fatjet_phi", "FatJet #phi", 50, -3.14, 3.14)
    hist_fatjet_count = ROOT.TH1D("fatjet_count", "Number of FatJets per Event", 10, 0, 10)
    
    # Fill histograms using Draw/Project
    print("Filling muon histograms...")
    tree.Project("muon_count", "Number_Muon", "", "goff")
    tree.Project("muon_pt", "Pt_Muon", "", "goff")
    tree.Project("muon_eta", "Eta_Muon", "", "goff")
    tree.Project("muon_phi", "Phi_Muon", "", "goff")
    
    print("Filling fatjet histograms...")
    tree.Project("fatjet_count", "Number_FatJet", "", "goff")
    tree.Project("fatjet_pt", "Pt_FatJet", "", "goff")
    tree.Project("fatjet_eta", "Eta_FatJet", "", "goff")
    tree.Project("fatjet_phi", "Phi_FatJet", "", "goff")
    
    # Save histograms to file
    output_file = ROOT.TFile("histograms.root", "RECREATE")
    hist_muon_pt.Write()
    hist_muon_eta.Write()
    hist_muon_phi.Write()
    hist_muon_count.Write()
    hist_fatjet_pt.Write()
    hist_fatjet_eta.Write()
    hist_fatjet_phi.Write()
    hist_fatjet_count.Write()
    output_file.Close()
    
    print("\nHistograms saved to histograms.root")
    
    # Print statistics
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total events processed: {tree.GetEntries()}")
    print(f"\nMuon Statistics:")
    print(f"  Average muons per event: {hist_muon_count.GetMean():.2f}")
    print(f"  Average muon PT: {hist_muon_pt.GetMean():.2f} GeV")
    print(f"\nFatJet Statistics:")
    print(f"  Average fatjets per event: {hist_fatjet_count.GetMean():.2f}")
    print(f"  Average fatjet PT: {hist_fatjet_pt.GetMean():.2f} GeV")

if __name__ == "__main__":
    main()