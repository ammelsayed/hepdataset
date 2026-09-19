import pytest

import hepdataset


def test_package_importable():
    assert hepdataset.__name__ == "hepdataset"


def test_make_dataset_is_lazy():
    # Accessing the attribute triggers the submodule import; unknown ones raise.
    assert getattr(hepdataset, "make_dataset") is not None
    with pytest.raises(AttributeError):
        hepdataset.not_a_real_attribute


def test_make_dataset_function():
    from hepdataset.make_dataset import make_dataset

    assert callable(make_dataset)


def test_public_modules_importable():
    import hepdataset.cli
    import hepdataset.make_dataset
    import hepdataset.merge_samples
    import hepdataset.samples_reader

    assert hepdataset.cli.SUBCOMMANDS is not None


def test_cli_subcommand_list():
    from hepdataset.cli import SUBCOMMANDS

    expected = {
        "make",
        "samples_reader",
        "merge_samples",
        "basic1_delphes",
        "basic2_delphes",
        "basic3_delphes",
        "adaptive_delphes",
    }
    assert set(SUBCOMMANDS) == expected