import math
import numpy as np
from numba import njit
from typing import Optional, Tuple

SENTINEL = np.nan

def DeltaR(obj1, obj2, use_rapidity=False):
    return obj1.P4().DeltaR(obj2.P4(), useRapidity=use_rapidity)

def DeltaPhi(obj1, obj2):
    return obj1.P4().DeltaPhi(obj2.P4())

def DeltaEta(obj1, obj2):
    return obj1.Eta - obj2.Eta

def Tau21(fj):
    tau1, tau2 = fj.Tau[0], fj.Tau[1]
    return tau2 / tau1 if tau1 > 0 else 1.0

def Tau32(fj):
    tau2, tau3 = fj.Tau[1], fj.Tau[2]
    return tau3 / tau2 if tau2 > 0 else 1.0

@njit(cache=True, fastmath=True)
def dPhi(phi1, phi2):
    delta = phi1 - phi2
    while delta > math.pi:    delta -= 2 * math.pi
    while delta < -math.pi:   delta += 2 * math.pi
    return delta

@njit(cache=True, fastmath=True)
def dEta(eta1, eta2):
    return eta1 - eta2

@njit(cache=True, fastmath=True)
def dR(eta1, phi1, eta2, phi2):
    return (dPhi(phi1, phi2) ** 2 + (eta1 - eta2) ** 2) ** 0.5

@njit(cache=True, fastmath=True)
def MtW(pt1, phi1, pt2, phi2):
    """
    Standard W-type transverse mass:
    mT = sqrt(2 * pT1 * pT2 * (1 − cos Δφ)).
    This is distinct from ROOT's TLorentzVector::Mt(), which returns
    sqrt(E^2 − pz^2) = sqrt(M^2 + pT^2) for the composite system.
    """
    dphi = dPhi(phi1, phi2)
    mt2  = 2.0 * float(pt1) * float(pt2) * (1.0 - np.cos(dphi))
    return float(np.sqrt(max(mt2, 0.0)))

@njit(cache=True, fastmath=True)
def SphericityAplanarity(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> Tuple[float, float]:
    """
    px, py, pz: Arrays of momentum components for all particles in an event.

    Compute Sphericity S and Aplanarity A from the linearised
    momentum tensor S_{ij} = Σ(p_i p_j) / Σ|p|^2  (so the trace = 1).

    Eigenvalues λ1 ≥ λ2 ≥ λ3 (with λ1+λ2+λ3 = 1) give:
      S = (3/2)(λ2 + λ3)  ∈ [0, 1]
      A = (3/2) λ3        ∈ [0, 1/2]

    S ≈ 0 → pencil-like jet event (collimated, one direction dominates)
    S ≈ 1 → perfectly spherical event (isotropic)
    A ≈ 0 → event is planar (lies mostly in a 2D plane)
    A → 0.5 → event is fully 3D isotropic
    """

    # px = np.asarray(px, dtype=np.float64)
    # py = np.asarray(py, dtype=np.float64)
    # pz = np.asarray(pz, dtype=np.float64)

    sum_p2 = (px**2 + py**2 + pz**2).sum()
    if sum_p2 == 0.0:
        return SENTINEL, SENTINEL
    S = np.array([
        [np.sum(px*px), np.sum(px*py), np.sum(px*pz)],
        [np.sum(py*px), np.sum(py*py), np.sum(py*pz)],
        [np.sum(pz*px), np.sum(pz*py), np.sum(pz*pz)],
    ]) / sum_p2
    eig = np.sort(np.linalg.eigvalsh(S))[::-1]   # λ1 ≥ λ2 ≥ λ3
    l1, l2, l3 = eig
    return float(1.5 * (l2 + l3)), float(1.5 * l3)

@njit(cache=True, fastmath=True)
def Circularity(px: np.ndarray, py: np.ndarray) -> float:
    """
    2D circularity C = 2 μ2 / (μ1 + μ2) where μ1 ≥ μ2 are the
    eigenvalues of the transverse-momentum tensor
    M_{ij} = Σ(pT_i pT_j) / Σ pT^2.
    C ∈ [0, 1]; 
    C = 0 for back-to-back events.
    C = 1 for isotropic events.
    """
    # px = np.asarray(px, dtype=np.float64)
    # py = np.asarray(py, dtype=np.float64)
    sum_pt2 = (px**2 + py**2).sum()
    if sum_pt2 == 0.0:
        return SENTINEL
    M = np.array([
        [np.sum(px*px), np.sum(px*py)],
        [np.sum(px*py), np.sum(py*py)],
    ]) / sum_pt2
    eig = np.sort(np.linalg.eigvalsh(M))[::-1]   # μ1 ≥ μ2
    l1, l2 = eig
    return float(2.0 * l2 / (l1 + l2))

@njit(cache=True, fastmath=True)
def EventShapes(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> Tuple[float, float, float]:
    S, A = SphericityAplanarity(px, py, pz)
    C    = Circularity(px, py)
    return S, A, C

@njit(cache=True, fastmath=True)
def Centrality(px: np.ndarray, py: np.ndarray, pz: np.ndarray, E : Optional[np.ndarray] = None) -> float:
    """
    C = Σ pT / Σ E  (or Σ|p| if energies are unavailable).
    """
    # px = np.asarray(px, dtype=np.float64)
    # py = np.asarray(py, dtype=np.float64)
    # pz = np.asarray(pz, dtype=np.float64)
    pt    = np.sqrt(px**2 + py**2)
    denom = (E.sum() if E is not None else np.sqrt(px**2 + py**2 + pz**2).sum())
    return float(pt.sum() / denom) if denom > 0.0 else SENTINEL