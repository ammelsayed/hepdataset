"""Tests for hepdataset.make_dataset helpers (no ROOT required)."""

import pytest

import hepdataset.make_dataset as md


class TestResolveCategoryName:
    @staticmethod
    @pytest.mark.parametrize(
        "category,expected",
        [
            ("Background", "bkg"),
            ("background", "bkg"),
            ("BACKGROUND", "bkg"),
            ("Signal", "sig"),
            ("signal", "sig"),
            ("Other", "Unknown"),
            ("", "Unknown"),
        ],
    )
    def test_category_mapping(category, expected):
        assert md.resolve_category_name(category) == expected


class TestLoadLoop:
    @staticmethod
    def test_loads_loop_module(monkeypatch):
        sentinel = object()
        imported = []

        class FakeModule:
            loop_tree = sentinel

        def fake_import(name, package=None):
            imported.append((name, package))
            return FakeModule

        monkeypatch.setattr(md.importlib, "import_module", fake_import)

        result = md._load_loop("adaptive_delphes")
        assert result is sentinel
        assert imported == [(".loops.adaptive_delphes", "hepdataset")]

    @staticmethod
    def test_works_with_extension(monkeypatch):
        imported = []

        class FakeModule:
            loop_tree = "tree"

        def fake_import(name, package=None):
            imported.append(name)
            return FakeModule

        monkeypatch.setattr(md.importlib, "import_module", fake_import)
        assert md._load_loop("adaptive_delphes.py") == "tree"
        assert imported == [".loops.adaptive_delphes"]


class TestProcessCrossSection:
    """Cross section resolution across the samples-card schema."""

    @staticmethod
    def test_prefers_inclusive():
        meta = {"inclusive_cross_section": 0.75, "fiducial_cross_section": 0.5}
        assert md.process_cross_section(meta) == 0.75

    @staticmethod
    def test_null_inclusive_falls_back_to_fiducial():
        meta = {"inclusive_cross_section": None, "fiducial_cross_section": 0.5}
        assert md.process_cross_section(meta) == 0.5

    @staticmethod
    def test_absent_inclusive_falls_back_to_fiducial():
        assert md.process_cross_section({"fiducial_cross_section": 0.11}) == 0.11

    @staticmethod
    def test_legacy_key_is_read_with_a_warning():
        with pytest.warns(DeprecationWarning):
            assert md.process_cross_section({"cross_section": 0.3}) == 0.3

    @staticmethod
    def test_missing_cross_section_raises():
        with pytest.raises(KeyError):
            md.process_cross_section({"nb_events": 3})