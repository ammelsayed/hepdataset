import os
import sys
import types
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
TESTS_DIR = Path(__file__).resolve().parent

# Make the `hepdataset` package importable without installing it.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Minimal ROOT test double.
#
# Several modules do `import ROOT` / `from ROOT import TFile` at import time.
# ROOT cannot be installed via pip, so we register a lightweight stand-in in
# sys.modules before any of those modules are imported. The fake exposes just
# enough surface for the code paths we unit-test.
# ---------------------------------------------------------------------------


class _FakeTree:
    def __init__(self, name="Delphes", entries=0):
        self._name = name
        self._entries = entries
        self._class_name = "TTree"

    def GetName(self):
        return self._name

    def GetEntries(self):
        return self._entries

    def SetName(self, name):
        self._name = name

    def CloneTree(self, *args, **kwargs):
        return _FakeTree(self._name, self._entries)

    def Write(self, **kwargs):
        return None

    def __repr__(self):
        return f"_FakeTree({self._name!r}, {self._entries})"


class _FakeKey:
    def __init__(self, name, class_name="TTree"):
        self._name = name
        self._class_name = class_name

    def GetName(self):
        return self._name

    def GetClassName(self):
        return self._class_name

    def __repr__(self):
        return f"_FakeKey({self._name!r})"


class _FakeFile:
    def __init__(self, path, mode="READ"):
        self.path = os.fspath(path)
        self.mode = mode
        self.trees = {}
        self._closed = False

    def cd(self):
        return None

    def Write(self, **kwargs):
        # Simulate ROOT writing an output file to disk.
        if "RECREATE" in (self.mode or ""):
            parent = os.path.dirname(self.path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            open(self.path, "a").close()
        return None

    def Close(self):
        self._closed = True

    def IsZombie(self):
        return False

    def Get(self, name):
        return self.trees.get(name)

    def GetListOfKeys(self):
        return [
            _FakeKey(name, getattr(tree, "_class_name", "TTree"))
            for name, tree in self.trees.items()
        ]

    @classmethod
    def Open(cls, path, *args, **kwargs):
        return _ROOT_CONTROLLER.open(path)

    def __repr__(self):
        return f"_FakeFile({self.path!r})"


class _FakeTChain:
    def __init__(self, *args, **kwargs):
        self._trees = []

    def Add(self, path):
        f = _ROOT_CONTROLLER.open(path)
        for key in f.GetListOfKeys():
            tree = f.Get(key.GetName())
            if tree is not None:
                self._trees.append(tree)

    def AddFile(self, path, tmax=-1, tname=None):
        f = _ROOT_CONTROLLER.open(path)
        for key in f.GetListOfKeys():
            tree = f.Get(key.GetName())
            if tree is not None:
                self._trees.append(tree)
                return
        self._trees.append(_FakeTree(tname or "Delphes", 0))

    def GetEntries(self):
        return sum(t.GetEntries() for t in self._trees)

    def CloneTree(self, *args, **kwargs):
        total = self.GetEntries()
        if _ROOT_CONTROLLER.clone_entries is not None:
            total = _ROOT_CONTROLLER.clone_entries
        return _FakeTree("Delphes", total)


class _RootController:
    """Registry backing the fake ROOT: path -> list of fake trees."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.files = {}
        self.clone_entries = None

    @staticmethod
    def normalize(path):
        return os.path.normcase(os.fspath(path))

    def register(self, path, tree_name="Delphes", entries=0):
        key = self.normalize(path)
        fake = _FakeFile(path)
        fake.trees[tree_name] = _FakeTree(tree_name, entries)
        self.files[key] = fake
        return fake

    def open(self, path, mode="READ"):
        key = self.normalize(path)
        if key in self.files:
            fake = _FakeFile(path, mode)
            fake.trees = {n: _FakeTree(n, t.GetEntries()) for n, t in self.files[key].trees.items()}
            return fake
        return _FakeFile(path, mode)

    def set_entries(self, path, entries):
        key = self.normalize(path)
        for tree in self.files[key].trees.values():
            tree._entries = entries


_ROOT_CONTROLLER = _RootController()


def _install_fake_root():
    module = types.ModuleType("ROOT")
    module.TFile = _FakeFile
    module.TChain = _FakeTChain
    module.SetOwnership = lambda obj, flag: None
    module.gInterpreter = types.SimpleNamespace(
        AddIncludePath=lambda p: None,
        Declare=lambda s: None,
    )
    module.gSystem = types.SimpleNamespace(Load=lambda s: None)
    module.gROOT = types.SimpleNamespace(SetBatch=lambda v: None)
    module.ExRootTreeReader = types.SimpleNamespace
    sys.modules["ROOT"] = module
    return module


_install_fake_root()


def _install_uproot_stub():
    """Fall back to a stub 'uproot' when the real package is absent.

    'uproot' is only used indirectly by samples_reader today; without the
    package installed we still want the tests to be importable.
    """
    try:
        import uproot  # noqa: F401
    except Exception:
        stub = types.ModuleType("uproot")
        stub.open = lambda *a, **k: None
        sys.modules.setdefault("uproot", stub)


_install_uproot_stub()


@pytest.fixture
def root_controller():
    """Access to the fake-ROOT file registry, clean for each test."""
    _ROOT_CONTROLLER.reset()
    yield _ROOT_CONTROLLER
    _ROOT_CONTROLLER.reset()


@pytest.fixture
def make_root_file(root_controller):
    """Create a placeholder .root file on disk and register a fake tree."""

    def _make(root, name, entries=0, tree_name="Delphes"):
        path = root / name
        path.touch()
        root_controller.register(str(path), tree_name=tree_name, entries=entries)
        return path

    return _make


@pytest.fixture
def samples_card(tmp_path):
    return TESTS_DIR / "samples_example1.yml"


@pytest.fixture
def branches_config_example(tmp_path):
    return TESTS_DIR / "branches_config_example1.yml"