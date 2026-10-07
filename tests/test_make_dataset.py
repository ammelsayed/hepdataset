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


def test_native_cpp_method_uses_native_runner(monkeypatch, tmp_path):
    from hepdataset import samples_reader
    from hepdataset.loops import native_basic

    class FakeSamplesReader:
        def __init__(self, _path):
            pass

        def read(self):
            return {"Background": {"sample": {
                "files": ["input.root"], "nb_events": 10, "cross_section": 1.0,
            }}}

    calls = []
    monkeypatch.setattr(samples_reader, "SamplesReader", FakeSamplesReader)
    monkeypatch.setattr(native_basic, "run_native_basic", lambda *args, **kwargs: (
        calls.append((args, kwargs)) or {"root_paths": {}, "ObjectSelector": None, "EventSelector": None}
    ))

    md.make_dataset("samples.yml", output_dir=tmp_path, loop_method="basic2_delphes_cpp",
                    max_workers=2, n_chunks=4)

    assert calls[0][0] == ("basic2_delphes",)
    assert calls[0][1]["input_root_file"] == "input.root"
    assert calls[0][1]["max_workers"] == 2
    assert calls[0][1]["n_chunks"] == 4
