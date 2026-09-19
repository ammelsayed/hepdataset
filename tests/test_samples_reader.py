"""Tests for the samples card reader: hepdataset.samples_reader (ROOT mocked)."""

import json

import pytest
from yaml import safe_load

from hepdataset.samples_reader import SamplesReader

from conftest import TESTS_DIR


# --------------------------------------------------------------------------
# Constructor validation
# --------------------------------------------------------------------------


class TestConstructor:
    @staticmethod
    def test_missing_file():
        with pytest.raises(FileNotFoundError):
            SamplesReader("/does/not/exist.yml")

    @staticmethod
    def test_directory_path(tmp_path):
        with pytest.raises(IsADirectoryError):
            SamplesReader(str(tmp_path))

    @staticmethod
    def test_unsupported_extension(tmp_path):
        path = tmp_path / "samples.txt"
        path.write_text("x")
        with pytest.raises(ValueError):
            SamplesReader(str(path))

    @staticmethod
    def test_xml_not_supported(tmp_path):
        path = tmp_path / "samples.xml"
        path.write_text("<x/>")
        with pytest.raises(NotImplementedError):
            SamplesReader(str(path))

    @staticmethod
    def test_format_detection(tmp_path):
        m = tmp_path / "s.yml"
        y = tmp_path / "s.yaml"
        j = tmp_path / "s.json"
        m.write_text("{}")
        y.write_text("{}")
        j.write_text("{}")
        assert SamplesReader(str(m)).fmt == "yaml"
        assert SamplesReader(str(y)).fmt == "yaml"
        assert SamplesReader(str(j)).fmt == "json"


# --------------------------------------------------------------------------
# clean_root_files
# --------------------------------------------------------------------------


class TestCleanRootFiles:
    @staticmethod
    def _reader(tmp_path):
        card = tmp_path / "unused.yml"
        card.write_text("{}")
        return SamplesReader(str(card))

    @staticmethod
    def test_keeps_only_existing_root_files(tmp_path):
        good = tmp_path / "good.root"
        good.write_text("")
        reader = TestCleanRootFiles._reader(tmp_path)

        valid = reader.clean_root_files(
            [
                str(tmp_path / "missing.root"),
                str(tmp_path / "not_root.txt"),
                str(good),
                str(good),  # duplicate
            ]
        )
        assert sorted(valid) == [str(good)]

    @staticmethod
    def test_non_root_extension_rejected(tmp_path):
        txt = tmp_path / "sample.root.txt"
        txt.write_text("")
        reader = TestCleanRootFiles._reader(tmp_path)
        assert reader.clean_root_files([str(txt)]) == []


# --------------------------------------------------------------------------
# get_nb_events (backed by the fake ROOT)
# --------------------------------------------------------------------------


class TestNbEvents:
    @staticmethod
    def test_counts_entries(make_root_file, tmp_path):
        path = make_root_file(tmp_path, "a.root", entries=11)
        reader = TestNbEvents._reader(tmp_path)
        assert reader.get_nb_events(str(path)) == 11

    @staticmethod
    def test_missing_file_is_zero(tmp_path):
        reader = TestNbEvents._reader(tmp_path)
        assert reader.get_nb_events(str(tmp_path / "nope.root")) == 0

    @staticmethod
    def _reader(tmp_path):
        card = tmp_path / "unused.yml"
        card.write_text("{}")
        return SamplesReader(str(card))


# --------------------------------------------------------------------------
# read() end-to-end
# --------------------------------------------------------------------------


class TestRead:
    @staticmethod
    def test_fills_defaults_and_drops_bad_processes(make_root_file, tmp_path):
        good1 = make_root_file(tmp_path, "good1.root", entries=2).as_posix()
        good2 = make_root_file(tmp_path, "good2.root", entries=3).as_posix()
        good3 = make_root_file(tmp_path, "good3.root", entries=5).as_posix()

        card = tmp_path / "samples.yml"
        card.write_text(
            f"""
Background:
  complete:
    cross_section: 0.5
    k_factor: 1.2
    files:
      - "{good1}"
      - "{good2}"
  incomplete:
    files:
      - "{good1}"
  no_valid_files:
    files:
      - "/does/not/exist.root"
  missing_files_key:
    cross_section: 0.9
Signal:
  sparse:
    files:
      - "{good3}"
"""
        )

        data = SamplesReader(str(card)).read()

        # 'incomplete' receives default cross section / errors / k-factor
        incomplete = data["Background"]["incomplete"]
        assert incomplete["cross_section"] == 1.0
        assert incomplete["k_factor"] == 1.0
        assert incomplete["cross_section_err_high"] == 0.0
        assert incomplete["cross_section_err_low"] == 0.0
        assert incomplete["nb_events"] == 2

        # valid card stored as given, nb_events summed
        complete = data["Background"]["complete"]
        assert complete["cross_section"] == 0.5
        assert complete["k_factor"] == 1.2
        assert complete["nb_events"] == 5

        # processes with bad/empty files are removed
        assert "no_valid_files" not in data["Background"]
        assert "missing_files_key" not in data["Background"]

        assert data["Signal"]["sparse"]["nb_events"] == 5

    @staticmethod
    def test_example_card_missing_roots_yields_empty_dicts(samples_card, capsys):
        data = SamplesReader(str(samples_card)).read()
        assert data == {"Background": {}, "Signal": {}}

    @staticmethod
    def test_read_json_card(make_root_file, tmp_path):
        good = make_root_file(tmp_path, "j.root", entries=7)
        card = tmp_path / "samples.json"
        card.write_text(
            json.dumps(
                {
                    "Signal": {
                        "proc": {
                            "cross_section": 1.0,
                            "k_factor": 1.0,
                            "files": [str(good)],
                        }
                    }
                }
            )
        )
        data = SamplesReader(str(card)).read()
        assert data["Signal"]["proc"]["nb_events"] == 7


# --------------------------------------------------------------------------
# Export helpers
# --------------------------------------------------------------------------


class TestExports:
    @staticmethod
    def test_to_json_roundtrip(tmp_path):
        data = {"Background": {"p": {"nb_events": 3, "cross_section": 1.0}}}
        out = tmp_path / "out.json"
        SamplesReader.to_json(data, str(out))
        assert json.loads(out.read_text()) == data

    @staticmethod
    def test_to_xml_contains_tags(tmp_path):
        data = {"Signal": {"proc": {"cross_section": 1.0}}}
        out = tmp_path / "out.xml"
        SamplesReader.to_xml(data, str(out))
        text = out.read_text()
        assert "<Signal>" in text
        assert "<proc>" in text
        assert "cross_section" in text


# --------------------------------------------------------------------------
# Table printing (pure output, no ROOT needed)
# --------------------------------------------------------------------------


class TestPrintTable:
    @staticmethod
    def _data():
        return {
            "Background": {
                "tt": {
                    "nb_events": 100,
                    "cross_section": 0.5,
                    "cross_section_err_high": 0.01,
                    "cross_section_err_low": 0.02,
                    "k_factor": 1.0,
                }
            }
        }

    @staticmethod
    def test_plain_table(capsys):
        from hepdataset.samples_reader import print_table

        print_table(TestPrintTable._data(), fmt="plain")
        out = capsys.readouterr().out
        assert "Process" in out
        assert "Background" in out
        assert "tt" in out

    @staticmethod
    def test_latex_table(capsys):
        from hepdataset.samples_reader import print_table

        print_table(TestPrintTable._data(), fmt="latex")
        out = capsys.readouterr().out
        assert "\\begin{table}" in out
        assert "N_{\\mathrm{gen}}" in out