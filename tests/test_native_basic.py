from pathlib import Path

from hepdataset.loops.native_basic import _runtime_env, merge_native_root_files


def test_runtime_env_uses_root_and_delphes_paths(monkeypatch):
    monkeypatch.setenv("ROOT_CONFIG", "/opt/root/bin/root-config")
    monkeypatch.setenv("DELPHES_HOME", "/opt/delphes")
    env = _runtime_env()
    assert env["PATH"].startswith("/opt/root/bin:")
    assert env["LD_LIBRARY_PATH"].startswith("/opt/delphes:")


def test_native_root_merge_copies_a_single_input(tmp_path):
    source = tmp_path / "chunk.root"
    target = tmp_path / "merged.root"
    source.write_bytes(b"root-file")
    merge_native_root_files([source], target)
    assert target.read_bytes() == b"root-file"
