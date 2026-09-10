import ROOT

def select_objects(FatJet_branch, Electron_branch, Muon_branch):
    """Select objects based on kinematic criteria.

    Parameters
    ----------
    FatJet_branch : ROOT.TBranch
        Branch containing fat jet objects.
    Electron_branch : ROOT.TBranch
        Branch containing electron objects.
    Muon_branch : ROOT.TBranch
        Branch containing muon objects.

    Returns
    -------
    dict
        Dictionary containing selected objects:
        {
            "goodFatJets": list of selected fat jets,
            "goodElectrons": list of selected electrons,
            "goodMuons": list of selected muons
        }
    """
    goodFatJets, goodWFatJets, goodZFatjets, goodHFatjets, goodTopFatjets, goodEWFatJets = [], [], [], [], [], []
    goodLeptons, goodElectrons, goodMuons = [], [], []
    goodJets, goodBJets, goodCJets, goodSJets, goodTauJets = [], [], [], [], []

    # Collect good fatjets (pT > 300 GeV, |eta| < 2.5, in mass window)
    mW, mH = 80.4, 125.0
    mWindow = 20
    for i in range(FatJet_branch.GetEntries()):
        fatjet = FatJet_branch.At(i)
        MassOk = mW - mWindow <= fatjet.SoftDroppedP4[0].M() <= mH + mWindow
        if MassOk and fatjet.PT > 300 and abs(fatjet.Eta) <= 2.5:
            goodFatJets.append(fatjet)

    # Collect good leptons
    IsoCutMuon     = 0.1
    IsoCutElectron = 0.2
    for i in range(Muon_branch.GetEntries()):
        muon = Muon_branch.At(i)
        if muon.IsolationVar < IsoCutMuon and muon.PT > 25 and abs(muon.Eta) <= 2.5:
            goodLeptons.append(muon)
    for i in range(Electron_branch.GetEntries()):
        electron = Electron_branch.At(i)
        if electron.IsolationVar < IsoCutElectron and electron.PT > 30 and abs(electron.Eta) <= 2.5:
            goodLeptons.append(electron)
    
    goodFatJets.sort(key=lambda fj: fj.PT, reverse=True)
    goodLeptons.sort(key=lambda lep: lep.PT, reverse=True)

    return {
        "goodFatJets": goodFatJets,
        "goodWFatJets": goodWFatJets,
        "goodZFatjets": goodZFatjets,
        "goodHFatjets": goodHFatjets,
        "goodTopFatjets": goodTopFatjets,
        "goodEWFatJets": goodEWFatJets,

        "goodLeptons": goodLeptons,
        "goodElectrons": goodElectrons,
        "goodMuons": goodMuons,
        
        "goodJets": goodJets,
        "goodBJets": goodBJets,
        "goodCJets": goodCJets,
        "goodSJets": goodSJets,
        "goodTauJets": goodTauJets
    }