"""Tests for the pure-Python/numba kinematics in hepdataset.core.kinematics."""

import math

import pytest

numpy = pytest.importorskip("numpy")
pytest.importorskip("numba")
kin = pytest.importorskip("hepdataset.core.kinematics")

np = numpy


def test_dphi_identity():
    assert kin.dPhi(0.0, 0.0) == 0.0
    assert kin.dPhi(1.0, 1.0) == 0.0


def test_dphi_wraps_into_interval():
    # dPhi(0, 5): delta = -5 is outside [-pi, pi], so +2pi is applied.
    value = kin.dPhi(0.0, 5.0)
    assert value == pytest.approx(2 * math.pi - 5.0)


def test_dphi_uses_shortest_arc():
    # 3.0 and -3.0 differ by 6.0; subtracting 2pi gives ~ -0.283.
    assert kin.dPhi(3.0, -3.0) == pytest.approx(6.0 - 2 * math.pi)


def test_delta_r_basic():
    assert kin.dR(0.0, 0.0, 0.0, 0.0) == 0.0
    assert kin.dR(0.0, 0.0, 1.0, 1.0) == pytest.approx(math.sqrt(2))


def test_deta():
    assert kin.dEta(1.0, 4.0) == pytest.approx(-3.0)


def test_mtw_zero_when_aligned():
    assert kin.MtW(100.0, 0.0, 100.0, 0.0) == 0.0


def test_mtw_standard_form():
    # mT = sqrt(2 pT1 pT2 (1 - cos dphi)), with dphi = pi -> sqrt(1600) = 40
    assert kin.MtW(40.0, 0.0, 10.0, -math.pi) == pytest.approx(40.0)


def test_sphericity_single_forward_particle():
    S, A = kin.SphericityAplanarity(np.array([0.0]), np.array([0.0]), np.array([10.0]))
    assert S == pytest.approx(0.0)
    assert A == pytest.approx(0.0)


def test_sphericity_back_to_back_along_axis():
    S, A = kin.SphericityAplanarity(
        np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([10.0, -10.0])
    )
    assert S == pytest.approx(0.0)
    assert A == pytest.approx(0.0)


def test_sphericity_isotropic_sphere():
    px = np.array([1.0, 0.0, 0.0])
    py = np.array([0.0, 1.0, 0.0])
    pz = np.array([0.0, 0.0, 1.0])
    S, A = kin.SphericityAplanarity(px, py, pz)
    assert S == pytest.approx(1.0)
    assert A == pytest.approx(0.5)


def test_sphericity_empty_momentum_is_sentinel():
    S, A = kin.SphericityAplanarity(
        np.array([0.0]), np.array([0.0]), np.array([0.0])
    )
    assert math.isnan(S)
    assert math.isnan(A)


def test_circularity_back_to_back():
    C = kin.Circularity(np.array([10.0, -10.0]), np.array([0.0, 0.0]))
    assert C == pytest.approx(0.0)


def test_circularity_isotropic_transverse():
    px = np.array([1.0, -1.0, 0.0])
    py = np.array([0.0, 0.0, 1.0])
    assert kin.Circularity(px, py) == pytest.approx(2.0 / 3.0)


def test_centrality_no_energies():
    px = np.array([3.0])
    py = np.array([4.0])
    pz = np.array([0.0])
    # C = sum(pt)/sum(|p|) = 5/5 = 1
    assert kin.Centrality(px, py, pz) == pytest.approx(1.0)


def test_centrality_with_energies():
    px = np.array([10.0])
    py = np.array([0.0])
    pz = np.array([0.0])
    E = np.array([20.0])
    assert kin.Centrality(px, py, pz, E) == pytest.approx(0.5)


def test_event_shapes_combines_outputs():
    px = np.array([1.0, 0.0, 0.0])
    py = np.array([0.0, 1.0, 0.0])
    pz = np.array([0.0, 0.0, 1.0])
    S, A, C = kin.EventShapes(px, py, pz)
    assert S == pytest.approx(1.0)
    assert A == pytest.approx(0.5)
    # Transverse tensor is diagonal 0.5/0.5 -> isotropic -> C = 1
    assert C == pytest.approx(1.0)