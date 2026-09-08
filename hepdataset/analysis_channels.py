## =============================
## Define signal regions
## =============================

def get_signal_regions(isLoose = True):
    if isLoose:
        signal_regions = {
            "0L"  : ["JJ"],
            "1L"  : ["lepJ", "lepJJ"],
            "2OS" : ["leplepJ"],
            "2SS" : ["leplepJ"],
            "3L"  : ["lepleplep"],
            # "4L"  : ["leplepleplep"]
        }
    else:
        signal_regions = {
            "0L"  : ["JJ"],
            "1L"  : ["eJ", "muJ", "eJJ", "muJJ"],
            "2OS" : ["eeJ", "emuJ", "mumuJ"],
            "2SS" : ["eeJ", "emuJ", "mumuJ"],
            "3L"  : ["eee", "eemu", "emumu", "mumumu"],
            # "4L"  : ["eeee", "eeemu", "eemumu", "emumumu", "mumumumu"]
        }

    signal_regions_keys = [
        f"{channel}_{region}"  
        for channel, regions in signal_regions.items()
        for region in regions
    ]

    return signal_regions, signal_regions_keys

def lepton_type(idx, leptons, isLoose = True):
    if isLoose:
        return "lep"
    else:
        return "e" if leptons[idx].ClassName().startswith("Electron") else "mu"

def classify_signal_region_key(goodLeptons, goodFatJets, isLoose = True):
    n_leps = len(goodLeptons)
    n_fatjets = len(goodFatJets)

    ## Add this to test code
    if testCode:
        return "test_test1"

    # 0 leptons channel
    if n_leps == 0:
        if n_fatjets >= 2: return "0L_JJ"
        else: return None
    
    # 1 lepton channel
    if n_leps == 1:
        lep_type = lepton_type(0, goodLeptons, isLoose)
        if n_fatjets == 1: return f"1L_{lep_type}J"
        elif n_fatjets >= 2: return f"1L_{lep_type}JJ"
        else: return None

    # 2 leptons channel
    elif n_leps == 2:
        lep1_type = lepton_type(0, goodLeptons, isLoose)
        lep2_type = lepton_type(1, goodLeptons, isLoose)
        lep1_charge = goodLeptons[0].Charge
        lep2_charge = goodLeptons[1].Charge
        total_charge = lep1_charge + lep2_charge
        lep_types = sorted([lep1_type, lep2_type])
        if total_charge == 0: # opposite sign
            if n_fatjets >= 1: return f"2OS_{lep_types[0]}{lep_types[1]}J"
            else: return None
        else: # same sign
            if n_fatjets >= 1: return f"2SS_{lep_types[0]}{lep_types[1]}J"
            else: return None
    
    # 3 leptons channel
    elif n_leps >= 3:
        lep1_type = lepton_type(0, goodLeptons, isLoose)
        lep2_type = lepton_type(1, goodLeptons, isLoose)
        lep3_type = lepton_type(2, goodLeptons, isLoose)
        lep_types = sorted([lep1_type, lep2_type, lep3_type])
        if n_fatjets >= 0: return f"3L_{lep_types[0]}{lep_types[1]}{lep_types[2]}"
        else:return None
    
    # elif n_leps == 4:
    #     lep1_type = lepton_type(0, goodLeptons, isLoose)
    #     lep2_type = lepton_type(1, goodLeptons, isLoose)
    #     lep3_type = lepton_type(2, goodLeptons, isLoose)
    #     lep4_type = lepton_type(3, goodLeptons, isLoose)
    #     lep_types = sorted([lep1_type, lep2_type, lep3_type, lep4_type])
    #     if n_fatjets >= 0: return f"4L_{lep_types[0]}{lep_types[1]}{lep_types[2]}{lep_types[3]}"
    #     else:return None
    
    else:
        return None
