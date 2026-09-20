"""Tests for the CLI dispatcher: hepdataset.cli."""

import sys
import types

import pytest

from hepdataset import cli


class _FakeModule:
    def __init__(self, return_value):
        self._return_value = return_value
        self.calls = []

    def main(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._return_value


@pytest.fixture
def fake_module(monkeypatch):
    module = _FakeModule(41)
    seen = []

    def fake_import(name):
        seen.append(name)
        return module

    monkeypatch.setattr(cli, "import_module", fake_import)
    return module, seen


def test_no_args_prints_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hepdataset"])
    assert cli.main() == 0
    assert "Available subcommands" in capsys.readouterr().out


def test_help_flag_prints_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hepdataset", "--help"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "Available subcommands" in out
    assert "adaptive_delphes" in out


def test_unknown_first_arg_routes_to_make(fake_module, monkeypatch):
    module, seen = fake_module
    monkeypatch.setattr(sys, "argv", ["hepdataset", "my_card.yml", "--output-dir", "out"])
    assert cli.main() == 41
    assert seen == ["hepdataset.make_dataset"]
    assert sys.argv == ["hepdataset make", "my_card.yml", "--output-dir", "out"]


def test_recognized_subcommand_dispatches(fake_module, monkeypatch):
    module, seen = fake_module
    monkeypatch.setattr(
        sys, "argv", ["hepdataset", "samples_reader", "--print", "card.yml"]
    )
    assert cli.main() == 41
    assert seen == ["hepdataset.samples_reader"]
    assert sys.argv == ["hepdataset samples_reader", "--print", "card.yml"]


def test_none_return_treated_as_zero(fake_module, monkeypatch):
    module, seen = fake_module
    module._return_value = None
    monkeypatch.setattr(sys, "argv", ["hepdataset", "merge_samples", "output/"])
    assert cli.main() == 0


def test_zero_return_passes_through(fake_module, monkeypatch):
    module, seen = fake_module
    module._return_value = 0
    monkeypatch.setattr(sys, "argv", ["hepdataset", "adaptive_delphes"])
    assert cli.main() == 0