# analysis_channels.py

def PrintAnalysisChannelYields(ac_counts, treeName, event_weight = None, lum = None):
    total_events = ac_counts["initial"]
    if total_events == 0:
        return
    df = pd.DataFrame(ac_counts.items(), columns=[" Analysis Channel/Region", "Events"])
    if (event_weight is not None) and (lum is not None):
        df[f"Yield ({int(lum)} fb^-1)"] = df["Events"] * event_weight
        df[f"Cross Section (fb)"] = df[f"Yield ({int(lum)} fb^-1)"] / lum
    df["Fraction"] = df["Events"] / total_events
    df["Fraction"] = df["Fraction"].map(lambda x: f"{x*100:.2f}%")
    print(f"\n*** Analysis channels yields for {treeName} ***")
    print(tabulate(df, headers='keys', tablefmt="simple", showindex=False, colalign=("left",) * 4))


def lepton_flavour(lep, splitByFlavour=False):
    """
    Return 'lep' if we don't split by flavour, else 'e' / 'mu'.
    """
    return ("e" if lep.ClassName() == "Electron" else "mu") if splitByFlavour else "lep"


def get_analysis_channel_keys(splitByFlavour=False):
    if splitByFlavour:
        ac_dict = {
            "0L" : ["JJ"],
            "1L" : ["eJ", "muJ", "eJJ", "muJJ"],
            "2OSL": ["eeJ", "emuJ", "mumuJ"],
            "2SSL": ["eeJ", "emuJ", "mumuJ"],
            "3L" : ["eee", "eemu", "emumu", "mumumu"],
            # "4L": ["eeee", "eeemu", "eemumu", "emumumu", "mumumumu"],
        }
    else:
        ac_dict = {
            "0L" : ["JJ"],
            "1L" : ["lepJ", "lepJJ"],
            "2OSL": ["leplepJ"],
            "2SSL": ["leplepJ"],
            "3L" : ["lepleplep"],
            # "4L": ["leplepleplep"],
        }

    ac_keys = [f"{ac}_{r}" for ac, acr in ac_dict.items() for r in acr]
    return ac_keys, ac_dict


def classify_analysis_channel(goodLeptons, goodFatJets, splitByFlavour=False):
    n_lep = len(goodLeptons)
    n_fj  = len(goodFatJets)

    # 0 leptons
    if n_lep == 0:
        return "0L_JJ" if n_fj >= 2 else None

    # 1 lepton
    if n_lep == 1:
        f = lepton_flavour(goodLeptons[0], splitByFlavour)
        if n_fj == 1: return f"1L_{f}J"
        elif n_fj >= 2: return f"1L_{f}JJ"
        else: return None

    # 2 leptons
    if n_lep == 2:
        lep1, lep2 = goodLeptons[0], goodLeptons[1]
        q_tot_abs  = abs(lep1.Charge + lep2.Charge)
        fs = "".join(sorted(lepton_flavour(l, splitByFlavour) for l in (lep1, lep2)))

        # opposite sign lepton pair (no requirment on the flavour)
        if q_tot_abs == 0: 
            if n_fj >= 1: return f"2OSL_{fs}J"
            else: return None

        # same sign lepton pair (no requirment on the flavour)
        if q_tot_abs == 2: 
            if n_fj >= 1: return f"2SSL_{fs}J"
            else: return None

    # 3 leptons
    if n_lep >= 3:
        fs = "".join(sorted(lepton_flavour(l, splitByFlavour) for l in goodLeptons[:3]))
        if n_fj >= 0: return f"3L_{fs}"
        else: return None

    # 4 leptons channel
    # if n_lep == 4:
    #     fs = "".join(sorted(lepton_flavour(l, splitByFlavour) for l in goodLeptons[:4]))
    #     if f n_fatjets >= 0: return f"4L_{fs}"
    #     else: return None

    return None


if __name__ == "__main__":

    import argparse
    from itertools import combinations_with_replacement as cwr

    class Lepton:
        def __init__(self, flavour, charge):
            self._flavour = flavour
            self.Charge   = charge
        def ClassName(self):
            return self._flavour

    def _check(n_lep_max = 3, n_fj_max = 2):
        L = [Lepton(f, q) for f in ("Electron", "Muon") for q in (1, -1)]
        J = [object()]  # a "jet" — only len() matters

        samples = [(lp, nj * [J]) for nlp in range(n_lep_max + 1) for lp in cwr(L, nlp) for nj in range(n_fj_max + 1)]

        for split in (False, True):
            print(f"--- splitByFlavour={split} ---")
            for leps, jets in samples:
                tag = f"{[l.ClassName()[0].lower() + ('+' if l.Charge > 0 else '-') for l in leps]}, {len(jets)}j"
                print(f" {tag:25s} -> {classify_analysis_channel(leps, jets, split)}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run channel-classification self-tests")
    args = parser.parse_args()
    if args.check:
        _check()