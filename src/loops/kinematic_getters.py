"""Explicit getter dispatch for the configurable Delphes loop.

The adaptive loop intentionally uses ``getattr`` so any validated property
name can be attempted at runtime. This module provides a finite, inspectable
mapping instead: configured names are resolved through explicit getter tables,
which can be translated directly to switches/calls in C++.
"""

from __future__ import annotations


_VECTOR_GETTERS = {
    "X": lambda p: p.X(), "Y": lambda p: p.Y(), "Z": lambda p: p.Z(),
    "Px": lambda p: p.Px(), "Py": lambda p: p.Py(), "Pz": lambda p: p.Pz(),
    "Pt": lambda p: p.Pt(), "P": lambda p: p.P(), "E": lambda p: p.E(),
    "Energy": lambda p: p.Energy(), "Theta": lambda p: p.Theta(),
    "CosTheta": lambda p: p.CosTheta(), "Phi": lambda p: p.Phi(),
    "Rho": lambda p: p.Rho(), "Perp": lambda p: p.Perp(),
    "Perp2": lambda p: p.Perp2(), "Et": lambda p: p.Et(),
    "Et2": lambda p: p.Et2(), "Mag": lambda p: p.Mag(),
    "Mag2": lambda p: p.Mag2(), "M": lambda p: p.M(),
    "M2": lambda p: p.M2(), "Mt": lambda p: p.Mt(),
    "Mt2": lambda p: p.Mt2(), "Beta": lambda p: p.Beta(),
    "Gamma": lambda p: p.Gamma(), "Plus": lambda p: p.Plus(),
    "Minus": lambda p: p.Minus(), "Rapidity": lambda p: p.Rapidity(),
    "Eta": lambda p: p.Eta(), "PseudoRapidity": lambda p: p.PseudoRapidity(),
}

# The output key is the branch-card kinematic; each body directly names the
# corresponding Delphes member (not a run-time attribute lookup).
_LEPTON_GETTERS = {
    "PT": lambda o: o.PT, "Eta": lambda o: o.Eta, "Phi": lambda o: o.Phi,
    "T": lambda o: o.T, "Charge": lambda o: o.Charge,
    "IsolationVar": lambda o: o.IsolationVar,
    "IsolationVarRhoCorr": lambda o: o.IsolationVarRhoCorr,
    "SumPtCharged": lambda o: o.SumPtCharged,
    "SumPtNeutral": lambda o: o.SumPtNeutral,
    "SumPtChargedPU": lambda o: o.SumPtChargedPU, "SumPt": lambda o: o.SumPt,
    "D0": lambda o: o.D0, "DZ": lambda o: o.DZ,
    "ErrorD0": lambda o: o.ErrorD0, "ErrorDZ": lambda o: o.ErrorDZ,
}
_JET_GETTERS = {
    "PT": lambda o: o.PT, "Eta": lambda o: o.Eta, "Phi": lambda o: o.Phi,
    "T": lambda o: o.T, "Mass": lambda o: o.Mass,
    "DeltaEta": lambda o: o.DeltaEta, "DeltaPhi": lambda o: o.DeltaPhi,
    "Flavor": lambda o: o.Flavor, "FlavorAlgo": lambda o: o.FlavorAlgo,
    "FlavorPhys": lambda o: o.FlavorPhys, "TauFlavor": lambda o: o.TauFlavor,
    "BTag": lambda o: o.BTag, "BTagAlgo": lambda o: o.BTagAlgo,
    "BTagPhys": lambda o: o.BTagPhys, "TauTag": lambda o: o.TauTag,
    "TauWeight": lambda o: o.TauWeight, "Charge": lambda o: o.Charge,
    "EhadOverEem": lambda o: o.EhadOverEem,
    "NCharged": lambda o: o.NCharged, "NNeutrals": lambda o: o.NNeutrals,
    "NeutralEnergyFraction": lambda o: o.NeutralEnergyFraction,
    "Beta": lambda o: o.Beta, "BetaStar": lambda o: o.BetaStar,
    "MeanSqDeltaR": lambda o: o.MeanSqDeltaR,
    "NSubJetsTrimmed": lambda o: o.NSubJetsTrimmed,
    "NSubJetsPruned": lambda o: o.NSubJetsPruned,
    "NSubJetsSoftDropped": lambda o: o.NSubJetsSoftDropped,
    "ExclYmerge12": lambda o: o.ExclYmerge12,
    "ExclYmerge23": lambda o: o.ExclYmerge23,
    "ExclYmerge34": lambda o: o.ExclYmerge34,
    "ExclYmerge45": lambda o: o.ExclYmerge45,
    "ExclYmerge56": lambda o: o.ExclYmerge56,
}
_MET_GETTERS = {
    "MET": lambda o: o.MET, "Eta": lambda o: o.Eta, "Phi": lambda o: o.Phi,
}
_SCALARHT_GETTERS = {"HT": lambda o: o.HT}


def vector_value(vector, name):
    """Return a TLorentzVector value, or ``None`` for an unsupported name."""
    getter = _VECTOR_GETTERS.get(name)
    return getter(vector) if getter is not None else None


def kinematic_value(object_type, obj, vector, name):
    """Return one requested value using explicit direct/vector getter tables.

    ``None`` means the configured name is unavailable for that object. In
    particular, n-subjettiness is handled explicitly for FatJet, and tag bits
    for leptons are derived from the concrete Delphes class.
    """
    if object_type == "Lepton":
        if name == "ElectronTag":
            return int(obj.ClassName() == "Electron")
        if name == "MuonTag":
            return int(obj.ClassName() == "Muon")
        getter = _LEPTON_GETTERS.get(name)
        if getter is not None:
            return getter(obj)
    elif object_type in ("Jet", "FatJet"):
        if object_type == "FatJet" and name.startswith("Tau"):
            suffix = name[3:]
            if len(suffix) == 2 and suffix.isdigit():
                numerator, denominator = int(suffix[0]), int(suffix[1])
                taus = obj.Tau
                if 1 <= denominator < numerator <= len(taus):
                    divisor = taus[denominator - 1]
                    return taus[numerator - 1] / divisor if divisor > 0 else 1.0
            elif suffix.isdigit():
                index = int(suffix)
                taus = obj.Tau
                if 1 <= index <= len(taus):
                    return taus[index - 1]
        getter = _JET_GETTERS.get(name)
        if getter is not None:
            return getter(obj)
    elif object_type == "MET":
        getter = _MET_GETTERS.get(name)
        if getter is not None:
            return getter(obj)
    elif object_type == "ScalarHT":
        getter = _SCALARHT_GETTERS.get(name)
        if getter is not None:
            return getter(obj)
    return vector_value(vector, name)
