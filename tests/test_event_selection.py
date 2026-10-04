from hepdataset.core.event_selection import EventSelector


class _Lepton:
    def __init__(self, class_name, charge):
        self.class_name = class_name
        self.Charge = charge

    def ClassName(self):
        return self.class_name


def test_event_selector_accepts_flavour_split_mode():
    selector = EventSelector(splitByFlavour=True)

    assert selector.splitByFlavour is True
    assert "1L_e" in selector.ac_keys
    assert "2OSL_ee" in selector.ac_keys
    assert "2SSL_emu" in selector.ac_keys


def test_split_flavour_os_pair_is_not_dropped_by_channel_key_whitespace():
    selector = EventSelector(splitByFlavour=True)

    channel = selector.Select([_Lepton("Electron", -1), _Lepton("Electron", 1)], [])

    assert channel == "2OSL_ee"
    assert selector.ac_counts[channel] == 1
    assert selector.ac_counts["dropped"] == 0
