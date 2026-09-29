import runpy
import sys
import types

import pytest


def test_inspect_invalid_card_returns_nonzero(tmp_path, monkeypatch):
    card = tmp_path / "event_selection.yml"
    card.write_text(
        """Channels:
  1L:
    Regions:
      lep:
        requirement: nL == 1
      electron:
        requirement: ne == 1
""",
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "ROOT", types.SimpleNamespace(TH1D=object))
    monkeypatch.setattr(sys, "argv", ["event_selection.py", str(card), "--inspect"])

    with pytest.raises(SystemExit) as error:
        runpy.run_path("onepiece/event_selection.py", run_name="__main__")

    assert error.value.code == 1
