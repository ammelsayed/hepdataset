from hepdataset.loops.kinematic_getters import kinematic_value, vector_value


class FakeP4:
    def Pt(self):
        return 42.0

    def M(self):
        return 3.5


class FakeLepton:
    PT = 41.0
    Eta = 1.2
    Charge = -1

    def __init__(self, class_name="Muon"):
        self._class_name = class_name

    def ClassName(self):
        return self._class_name


class FakeFatJet:
    Tau = [0.8, 0.4, 0.2]


def test_direct_lepton_fields_and_class_tags():
    lepton = FakeLepton("Electron")
    assert kinematic_value("Lepton", lepton, FakeP4(), "PT") == 41.0
    assert kinematic_value("Lepton", lepton, FakeP4(), "Charge") == -1
    assert kinematic_value("Lepton", lepton, FakeP4(), "ElectronTag") == 1
    assert kinematic_value("Lepton", lepton, FakeP4(), "MuonTag") == 0


def test_vector_alias_uses_explicit_tlorentzvector_method():
    lepton = FakeLepton()
    assert kinematic_value("Lepton", lepton, FakeP4(), "Pt") == 42.0
    assert vector_value(FakeP4(), "M") == 3.5


def test_missing_or_unimplemented_getter_is_unavailable():
    lepton = FakeLepton()
    assert kinematic_value("Lepton", lepton, FakeP4(), "WTag") is None
    assert vector_value(FakeP4(), "not_a_vector_method") is None


def test_fatjet_tau_values_and_ratios():
    fatjet = FakeFatJet()
    assert kinematic_value("FatJet", fatjet, FakeP4(), "Tau2") == 0.4
    assert kinematic_value("FatJet", fatjet, FakeP4(), "Tau21") == 0.5
