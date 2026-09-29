from hepdataset.core.event_selection import EventSelector


class Lepton:
    def __init__(self, flavour, charge):
        self.flavour = flavour
        self.Charge = charge

    def ClassName(self):
        return self.flavour


def lep(flavour, charge):
    return Lepton(flavour, charge)


def test_default_card_uses_inclusive_lepton_channels():
    selector = EventSelector()

    assert "0L_Nothing" in selector.ac_keys
    assert "1L_lep" in selector.ac_keys
    assert "2OSL_leplep" in selector.ac_keys
    assert "1L_e" not in selector.ac_keys


def test_flavour_card_classifies_os_and_ss_channels(tmp_path):
    card = tmp_path / "event_selection.yml"
    card.write_text(
        """Channels:
  2OSL:
    Regions:
      emu:
        requirement: (ne == 1) & (nmu == 1) & (QL == 0)
  2SSL:
    Regions:
      emu:
        requirement: (ne == 1) & (nmu == 1) & (QL == 2)
""",
        encoding="utf-8",
    )
    selector = EventSelector(card)

    assert selector.ClassifyChannelKey([lep("Electron", 1), lep("Muon", -1)], []) == "2OSL_emu"
    assert selector.ClassifyChannelKey([lep("Electron", 1), lep("Muon", 1)], []) == "2SSL_emu"


def test_inspect_rejects_overlapping_regions(tmp_path):
    card = tmp_path / "event_selection.yml"
    card.write_text(
        """Channels:
  1L:
    Regions:
      lep:
        requirement: nL == 1
      electron:
        requirement: ne == 1
""",
        encoding="utf-8",
    )

    assert EventSelector.Inspect(card) is False


def test_merge_preserves_custom_card(tmp_path):
    card = tmp_path / "event_selection.yml"
    card.write_text(
        """Channels:
  Custom:
    Regions:
      oneJet:
        requirement: (nL == 1) & (nJ == 1)
""",
        encoding="utf-8",
    )
    selector = EventSelector(card)
    selector.Select([lep("Electron", 1)], [object()])

    merged = EventSelector.Merge([selector])

    assert merged.config_path == card
    assert merged.ac_keys == ["Custom_oneJet"]
    assert merged.ac_counts["Custom_oneJet"] == 1
