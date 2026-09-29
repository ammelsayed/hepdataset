from hepdataset.core.event_selection import EventSelector


class Lepton:
    def __init__(self, flavour, charge):
        self.flavour = flavour
        self.Charge = charge

    def ClassName(self):
        return self.flavour


def lep(flavour, charge):
    return Lepton(flavour, charge)


def test_channel_keys_are_loaded_from_yaml():
    generic = EventSelector(splitByFlavour=False)
    flavour = EventSelector(splitByFlavour=True)

    assert "0L_Nothing" in generic.ac_keys
    assert "0L_Nothing" in flavour.ac_keys
    assert "2OSL_leplep" in generic.ac_keys
    assert "2OSL_emu" not in generic.ac_keys
    assert "2OSL_emu" in flavour.ac_keys
    assert "2OSL_leplep" not in flavour.ac_keys


def test_two_lepton_os_and_ss_channels_use_configured_flavours():
    electrons_muons = [lep("Electron", 1), lep("Muon", -1)]
    same_sign = [lep("Electron", 1), lep("Muon", 1)]

    generic = EventSelector(splitByFlavour=False)
    flavour = EventSelector(splitByFlavour=True)

    assert generic.ClassifyChannelKey(electrons_muons, []) == "2OSL_leplep"
    assert generic.ClassifyChannelKey(same_sign, []) == "2SSL_leplep"
    assert flavour.ClassifyChannelKey(electrons_muons, []) == "2OSL_emu"
    assert flavour.ClassifyChannelKey(same_sign, []) == "2SSL_emu"


def test_classifier_follows_custom_yaml_requirements(tmp_path):
    config = tmp_path / "event_selection.yml"
    config.write_text(
        """Channels:
  Custom:
    Regions:
      oneJet:
        requirement: (nL == 1) & (nJ == 1)
""",
        encoding="utf-8",
    )

    selector = EventSelector(config_path=config)

    assert selector.ac_keys == ["Custom_oneJet"]
    assert selector.ClassifyChannelKey([lep("Electron", 1)], [object()]) == "Custom_oneJet"
    assert selector.ClassifyChannelKey([], [object()]) is None


def test_more_than_three_leptons_are_dropped_when_yaml_requires_three():
    selector = EventSelector(splitByFlavour=True)
    four_leptons = [
        lep("Electron", 1), lep("Electron", -1),
        lep("Muon", 1), lep("Muon", -1),
    ]

    assert selector.ClassifyChannelKey(four_leptons, []) is None
