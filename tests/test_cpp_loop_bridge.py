import sys
import types

import pytest

bridge = pytest.importorskip("hepdataset.loops._cpp_loop_bridge")


def test_cpp_bridge_calls_python_loop_tree(monkeypatch):
    module = types.ModuleType("hepdataset.loops.explicit_delphes")
    module.loop_tree = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, module.__name__, module)

    result = bridge.run_loop("explicit_delphes", {"inputRootFile": "sample.root", "start_entry": 3})
    assert result == {"inputRootFile": "sample.root", "start_entry": 3}


def test_cpp_bridge_cli_forwards_args(monkeypatch):
    original_argv = ["outer-program", "--outer-option"]
    monkeypatch.setattr(sys, "argv", original_argv)
    module = types.ModuleType("hepdataset.loops.basic3_delphes")
    module.main = lambda: list(sys.argv)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    result = bridge.run_cli("basic3_delphes", ["input.root", "--show-progress"])
    assert result == ["hepdataset basic3_delphes", "input.root", "--show-progress"]
    assert sys.argv is original_argv


def test_cpp_bridge_cli_restores_argv_when_main_raises(monkeypatch):
    original_argv = ["outer-program", "--outer-option"]
    monkeypatch.setattr(sys, "argv", original_argv)
    module = types.ModuleType("hepdataset.loops.adaptive_delphes")

    def fail():
        raise RuntimeError("loop failed")

    module.main = fail
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(RuntimeError, match="loop failed"):
        bridge.run_cli("adaptive_delphes", ["input.root"])
    assert sys.argv is original_argv
