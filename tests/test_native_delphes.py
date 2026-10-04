from pathlib import Path

from hepdataset.loops.native_delphes import _channel_keys, _native_command


def test_channel_keys_match_python_event_selector_schema():
    ordinary = _channel_keys()
    split = _channel_keys(split_by_flavour=True)

    assert "2OSL_leplep" in ordinary
    assert "2OSL_ee" in split
    assert "1L_muJJ" in split
    assert len(ordinary) == 12
    assert len(split) == 26


def test_native_command_carries_ranges_and_selection_options(tmp_path):
    binary = tmp_path / "native-loop"
    args = {
        "inputRootFile": "events.root",
        "treeName": "SelectedEvents",
        "luminosity": 400.0,
        "show_progress": True,
        "split_by_flavour": True,
    }

    command = _native_command(
        binary, "branches.tsv", args,
        output_dir="out", output_file_name="result.root",
        start=12, end=34, event_weight=0.125, overwrite=True,
    )

    assert command[:2] == [str(binary), "--input-root-file"]
    assert command[command.index("--branch-manifest") + 1] == "branches.tsv"
    assert command[command.index("--start-entry") + 1] == "12"
    assert command[command.index("--end-entry") + 1] == "34"
    assert command[command.index("--event-weight") + 1] == "0.125"
    assert "--show-progress" in command
    assert "--split-by-flavour" in command
    assert "--overwrite" in command
