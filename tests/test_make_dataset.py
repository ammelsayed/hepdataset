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