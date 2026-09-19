"""Tests for ROOT output merging: hepdataset.merge_samples (ROOT mocked)."""

import hepdataset.merge_samples as ms


class TestTreeHelpers:
    @staticmethod
    def test_get_tree_name(make_root_file, tmp_path):
        path = make_root_file(tmp_path, "a.root", entries=3, tree_name="MyTree")
        assert ms.get_tree_name(str(path)) == "MyTree"

    @staticmethod
    def test_get_tree_name_unknown(tmp_path):
        # File exists physically but is not registered in the fake ROOT.
        path = tmp_path / "b.root"
        path.touch()
        assert ms.get_tree_name(str(path)) is None

    @staticmethod
    def test_get_tree_entries(make_root_file, tmp_path):
        path = make_root_file(tmp_path, "a.root", entries=9)
        assert ms.get_tree_entries(str(path)) == 9

    @staticmethod
    def test_get_tree_entries_unknown(tmp_path):
        path = tmp_path / "b.root"
        path.touch()
        assert ms.get_tree_entries(str(path)) == 0


class TestMergeSamples:
    @staticmethod
    def test_groups_and_merges_processes(make_root_file, tmp_path):
        make_root_file(tmp_path, "bkg_tt_sample0.root", entries=2)
        make_root_file(tmp_path, "bkg_tt_sample1.root", entries=3)
        make_root_file(tmp_path, "sig_ww_sample0.root", entries=5)
        unrelated = tmp_path / "notes.txt"
        unrelated.write_text("not a root file")

        ms.merge_samples(str(tmp_path))

        assert (tmp_path / "bkg_tt.root").exists()
        assert (tmp_path / "sig_ww.root").exists()
        # Sources are removed only after a verified merge
        assert not (tmp_path / "bkg_tt_sample0.root").exists()
        assert not (tmp_path / "bkg_tt_sample1.root").exists()
        assert not (tmp_path / "sig_ww_sample0.root").exists()
        assert unrelated.exists()

    @staticmethod
    def test_keeps_sources_when_write_count_mismatches(make_root_file, tmp_path, root_controller):
        make_root_file(tmp_path, "bkg_tt_sample0.root", entries=2)
        make_root_file(tmp_path, "bkg_tt_sample1.root", entries=3)

        # Force the fake merged tree to claim it wrote fewer events than expected.
        root_controller.clone_entries = 0
        ms.merge_samples(str(tmp_path))

        # Bad output removed, original sources preserved.
        assert not (tmp_path / "bkg_tt.root").exists()
        assert (tmp_path / "bkg_tt_sample0.root").exists()
        assert (tmp_path / "bkg_tt_sample1.root").exists()

    @staticmethod
    def test_short_circuits_without_samples(tmp_path):
        ms.merge_samples(str(tmp_path))
        assert list(tmp_path.iterdir()) == []