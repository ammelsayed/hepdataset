"""Unit tests for the branch-card reader: hepdataset.core.branches_reader."""

import pytest

from hepdataset.core.branches_reader import BranchesHandler

from conftest import TESTS_DIR, REPO_DIR


@pytest.fixture
def default_config():
    return TESTS_DIR / "branches_config_example1.yml"


@pytest.fixture
def packaged_config():
    return REPO_DIR / "src" / "defaults" / "branches_config.yml"


def _config(tmp_path, text):
    path = tmp_path / "config.yml"
    path.write_text(text)
    return str(path)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


class TestLoading:
    @staticmethod
    def test_default_handler_has_no_errors():
        handler = BranchesHandler()
        assert handler.is_valid()
        assert handler.objects == {}
        assert handler.global_scalars == []

    @staticmethod
    def test_loads_example_config(default_config):
        handler = BranchesHandler(str(default_config))
        assert handler.is_valid()
        assert set(handler.objects) == {"Lepton", "FatJet", "Jet", "MET"}

    @staticmethod
    def test_loads_packaged_config(packaged_config):
        handler = BranchesHandler(str(packaged_config))
        assert handler.is_valid()

    @staticmethod
    def test_missing_config_reports_error(tmp_path):
        handler = BranchesHandler(str(tmp_path / "does_not_exist.yml"))
        assert not handler.is_valid()
        assert handler._errors

    @staticmethod
    def test_empty_config_reports_error(tmp_path):
        handler = BranchesHandler(_config(tmp_path, ""))
        assert not handler.is_valid()


# --------------------------------------------------------------------------
# Object accessors
# --------------------------------------------------------------------------


class TestObjectAccessors:
    @staticmethod
    def test_counts_and_representations(default_config):
        handler = BranchesHandler(str(default_config))
        assert handler.get_obj_count("Lepton") == 4
        assert handler.get_obj_repr("Lepton") == ["Lepton"]
        assert handler.get_obj_count("Missing") == 0
        assert handler.get_obj_repr("Missing") == []

    @staticmethod
    def test_object_instances_indexed(default_config):
        handler = BranchesHandler(str(default_config))
        assert handler.get_obj_instances("Lepton") == [
            "Lepton0",
            "Lepton1",
            "Lepton2",
            "Lepton3",
        ]

    @staticmethod
    def test_object_instances_singleton_met(default_config):
        handler = BranchesHandler(str(default_config))
        assert handler.get_obj_instances("MET") == ["MET"]

    @staticmethod
    def test_object_instances_unknown():
        handler = BranchesHandler()
        assert handler.get_obj_instances("Ghost") == []

    @staticmethod
    def test_singleton_branch_names(default_config):
        handler = BranchesHandler(str(default_config))
        names = handler.get_obj_branch_names("MET")

        for kin in handler.get_obj_kinematics("MET"):
            assert f"{kin}_MET" in names

    @staticmethod
    def test_indexed_branch_names(default_config):
        handler = BranchesHandler(str(default_config))
        names = handler.get_obj_branch_names("Lepton")

        assert "Pt_Lepton0" in names
        assert "Eta_Lepton3" in names
        assert not any(n.endswith("_Lepton4") for n in names)

    @staticmethod
    def test_tau_substructure_skipped_for_groomed_jets(default_config):
        handler = BranchesHandler(str(default_config))
        names = handler.get_obj_branch_names("FatJet")

        assert "Tau1_FatJet0" in names
        assert "Pt_SoftDroppedFatJet0" in names
        # SoftDropped FatJets do not carry n-subjettiness kinematics
        tau_names = [n for n in names if "Tau" in n and "SoftDroppedFatJet" in n]
        assert tau_names == []

    @staticmethod
    def test_selection_branch_names(default_config):
        handler = BranchesHandler(str(default_config))
        names = handler.get_int_branch_names()

        assert "nPreQS_Lepton" in names
        assert "nPreES_Jet" in names
        assert "nPostES_FatJet" in names

    @staticmethod
    def test_float_branch_names_contain_weights(default_config):
        handler = BranchesHandler(str(default_config))
        names = handler.get_float_branch_names()

        assert names[-1] == "weight"
        assert "gen_weight" in names
        assert "Sphericity" in names


# --------------------------------------------------------------------------
# N-body combinations
# --------------------------------------------------------------------------


class TestNBody:
    @staticmethod
    def test_2body_combinations_have_same_type_pairs(default_config):
        handler = BranchesHandler(str(default_config))
        # include_same_represenations is True for the example card
        combos = handler.get_nbody_combinations(2)
        combo_strings = ["_".join(c) for c in combos]

        assert ("Lepton0", "Lepton1") in combos
        assert "Lepton0_Lepton1" in combo_strings

    @staticmethod
    def test_2body_combinations_exclude_twins_when_disabled(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    representations: ["Lepton"]
    kinematics: ["Pt"]
  MET:
    count: 1
    representations: ["MET"]
    kinematics: ["MET"]
event-variables:
  global_scalars: []
  event_shapes: []
multi-objects:
  Nmax: 2
  combo_set: ["Lepton", "MET"]
  include_same_represenations: False
  include_mt2: False
  basic_kinematics: ["M"]
  """,
            )
        )
        combos = handler.get_nbody_combinations(2)
        # Lepton + MET only, no Lepton-Lepton pairs
        assert ("Lepton0", "MET") in combos
        assert ("Lepton0", "Lepton1") not in combos

    @staticmethod
    def test_nbody_branch_names_format(default_config):
        handler = BranchesHandler(str(default_config))
        names = handler.get_nbody_branch_names(2)

        assert any(n.startswith("M_Lepton0_FatJet") for n in names)
        assert any(n.startswith("DeltaR_") for n in names)

    @staticmethod
    def test_nbody_kinematics_depend_on_n(default_config):
        handler = BranchesHandler(str(default_config))

        two_body = handler.get_nbody_kinematics(2)
        assert "MtW" in two_body
        assert "MT2" in two_body  # include_mt2 is True in the example card

        three_body = handler.get_nbody_kinematics(3)
        assert "MtW" not in three_body
        assert "MT2" not in three_body

    @staticmethod
    def test_nbody_kinematics_empty_for_n_lt_2(default_config):
        handler = BranchesHandler(str(default_config))
        assert handler.get_nbody_kinematics(1) == []


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class TestValidation:
    @staticmethod
    def test_invalid_object_definition(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    representations: ["Lepton"]
    kinematics: ["Pt"]
  Muon: "not a dict"
event-variables:
  global_scalars: []
  event_shapes: []
multi-objects:
  combo_set: ["Lepton"]
""",
            )
        )
        assert not handler.is_valid()
        assert any("must be a dictionary" in e for e in handler._errors)

    @staticmethod
    def test_missing_count(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    representations: ["Lepton"]
    kinematics: ["Pt"]
""",
            )
        )
        assert not handler.is_valid()
        assert any("missing 'count'" in e for e in handler._errors)

    @staticmethod
    def test_negative_count(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: -1
    representations: ["Lepton"]
    kinematics: ["Pt"]
""",
            )
        )
        assert not handler.is_valid()

    @staticmethod
    def test_non_integer_count(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2.5
    representations: ["Lepton"]
    kinematics: ["Pt"]
""",
            )
        )
        assert not handler.is_valid()

    @staticmethod
    def test_missing_representations(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    kinematics: ["Pt"]
""",
            )
        )
        assert not handler.is_valid()

    @staticmethod
    def test_empty_representations(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    representations: []
    kinematics: ["Pt"]
""",
            )
        )
        assert not handler.is_valid()

    @staticmethod
    def test_non_string_representations(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    representations: [123]
    kinematics: ["Pt"]
""",
            )
        )
        assert not handler.is_valid()

    @staticmethod
    def test_missing_kinematics(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    representations: ["Lepton"]
""",
            )
        )
        assert not handler.is_valid()

    @staticmethod
    def test_empty_kinematics(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    representations: ["Lepton"]
    kinematics: []
""",
            )
        )
        assert not handler.is_valid()

    @staticmethod
    def test_unknown_combo_object(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    representations: ["Lepton"]
    kinematics: ["Pt"]
event-variables:
  global_scalars: []
  event_shapes: []
multi-objects:
  Nmax: 2
  combo_set: ["Ghost"]
""",
            )
        )
        assert not handler.is_valid()
        assert any("'Ghost' in combo_set is not defined" in e for e in handler._errors)

    @staticmethod
    def test_invalid_nmax(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    representations: ["Lepton"]
    kinematics: ["Pt"]
multi-objects:
  Nmax: 1
""",
            )
        )
        assert not handler.is_valid()

    @staticmethod
    def test_non_list_global_scalars(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    representations: ["Lepton"]
    kinematics: ["Pt"]
event-variables:
  global_scalars: "not a list"
  event_shapes: []
""",
            )
        )
        assert not handler.is_valid()

    @staticmethod
    def test_mt2_warning_without_met(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    representations: ["Lepton"]
    kinematics: ["Pt"]
event-variables:
  global_scalars: []
  event_shapes: []
multi-objects:
  Nmax: 2
  combo_set: ["Lepton"]
  include_mt2: True
""",
            )
        )
        assert handler.is_valid()
        assert any("MT2 requires MET" in w for w in handler._warnings)

    @staticmethod
    def test_unknown_object_keys_warn(tmp_path):
        handler = BranchesHandler(
            _config(
                tmp_path,
                """
objects:
  Lepton:
    count: 2
    representations: ["Lepton"]
    kinematics: ["Pt"]
    extra_key: "ignored"
event-variables:
  global_scalars: []
  event_shapes: []
multi-objects:
  Nmax: 2
  combo_set: ["Lepton"]
""",
            )
        )
        assert handler.is_valid()
        assert any("unknown keys" in w for w in handler._warnings)

    @staticmethod
    def test_print_validation_reports_errors(tmp_path, capsys):
        handler = BranchesHandler(_config(tmp_path, "objects: {}\n"))
        handler.print_validation()
        out = capsys.readouterr().out
        assert "VALIDATION ERRORS" in out