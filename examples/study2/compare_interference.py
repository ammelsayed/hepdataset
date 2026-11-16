#!/usr/bin/env python3
"""Compare two signed interference-weight samples, event weight statistics and
angular shapes.

Usage:
    python3 compare_interference.py A.lhe.gz B.lhe.gz --label-a NAME --label-b NAME

Both files are LHE with four charged leptons per event and a signed XWGTUP.
"""
import argparse
import gzip
import math
import re
from collections import defaultdict

import numpy as np

WGT = re.compile(r"<wgt\s+id=['\"]([^'\"]+)['\"]\s*>\s*([-+0-9.eE]+)\s*</wgt>")


def open_any(p, mode):
    return gzip.open(p, mode + "t") if p.endswith(".gz") else open(p, mode)


def read_lhe(path):
    """Return (weights, leptons) where leptons[k] is a (4, 4) float array of
    (E, px, py, pz) for the four charged leptons of event k."""
    weights, leptons = [], []
    cur_w, cur_p, hdr, nup = None, [], False, 0
    with open_any(path, "r") as f:
        for line in f:
            s = line.strip()
            if s.startswith("<event"):
                cur_w, cur_p, hdr = None, [], True
                continue
            if hdr:
                tok = s.split()
                nup = int(tok[0])
                cur_w = float(tok[2])
                hdr = False
                continue
            if s.startswith("</event>"):
                if len(cur_p) >= 4:
                    weights.append(cur_w)
                    leptons.append(np.array(cur_p[:4], dtype=float))
                cur_w, cur_p = None, []
                continue
            if cur_w is not None and len(cur_p) < nup:
                t = s.split()
                if len(t) >= 13 and int(t[1]) == 1 and abs(int(t[0])) in (11, 13):
                    cur_p.append([float(t[i]) for i in (9, 6, 7, 8)])
    return np.array(weights), leptons


def pair_leptons(p):
    """Best OSSF pairing into two Z candidates, minimising the mass residual."""
    idx = range(4)
    best, best_score = None, 1e30
    for (a, b), (c, d) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
        def m2(i, j):
            q = p[i] + p[j]
            return q[0] ** 2 - q[1] ** 2 - q[2] ** 2 - q[3] ** 2
        score = abs(math.sqrt(max(m2(a, b), 0)) - 91.188) + \
                abs(math.sqrt(max(m2(c, d), 0)) - 91.188)
        if score < best_score:
            best_score, best = score, ((a, b), (c, d))
    return best


def cos_theta(p, pair):
    """cos of the lepton angle w.r.t. the Z flight direction, in the Z rest
    frame, for the first lepton of each pair."""
    out = []
    q_all = p.sum(axis=0)
    for (i, j) in pair:
        z = p[i] + p[j]
        mz = math.sqrt(max(z[0] ** 2 - z[1] ** 2 - z[2] ** 2 - z[3] ** 2, 1e-9))
        beta = z[1:] / z[0]
        b2 = float(beta @ beta)
        if b2 > 1e-12:
            gamma = 1.0 / math.sqrt(1 - b2)

            def boost(v):
                vd = float(v[1:] @ beta) / b2
                e = gamma * (v[0] - float(v[1:] @ beta))
                sp = v[1:] + ((gamma - 1) * vd - gamma * v[0]) * beta
                return np.array([e, *sp])
            zr, lr, qr = boost(z), boost(p[i]), boost(q_all)
        else:
            zr, lr, qr = z, p[i], q_all
        n = qr[1:] - zr[1:]
        n = n / max(np.linalg.norm(n), 1e-9)
        l3 = lr[1:] / max(np.linalg.norm(lr[1:]), 1e-9)
        out.append(float(n @ l3))
    return out


def observables(leptons):
    ct1, ct2, dphi = [], [], []
    for p in leptons:
        pair = pair_leptons(p)
        c1, c2 = cos_theta(p, pair)
        ct1.append(c1)
        ct2.append(c2)
        v = p[:, 1:3]
        pt = np.linalg.norm(v, axis=1)
        a, b = np.argsort(pt)[-2:]
        dphi.append(abs(np.arctan2(np.linalg.det(np.stack((v[a], v[b]))),
                                   float(v[a] @ v[b]))))
    return (np.array(ct1), np.array(ct2), np.array(dphi))


def weighted_hist(x, w, lo, hi, nb):
    edges = np.linspace(lo, hi, nb + 1)
    h, _ = np.histogram(x, bins=edges, weights=w)
    h2, _ = np.histogram(x, bins=edges, weights=w ** 2)
    n, _ = np.histogram(x, bins=edges)
    return h, np.sqrt(h2), n, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    wa, la = read_lhe(args.a)
    wb, lb = read_lhe(args.b)
    print("  %-14s events  <w>          rms w        neg%%      max|w|" % "sample")
    for name, w in ((args.label_a, wa), (args.label_b, wb)):
        print("  %-14s %6d  %+12.4e  %10.4e  %6.1f  %10.4e"
              % (name, len(w), w.mean(), w.std(),
                 100.0 * (w < 0).mean(), np.abs(w).max()))
    print()
    print("  %-14s <w>/sigma_ref style closure: mean(w) must be ~0" % "")
    print("  A mean/rms = %+.4f   B mean/rms = %+.4f"
          % (wa.mean() / wa.std(), wb.mean() / wb.std()))
    print()

    oa, ob = observables(la), observables(lb)
    names = ("cos_theta_1", "cos_theta_2", "DeltaPhi_ll")
    ranges = ((-1, 1), (-1, 1), (0, math.pi))
    for name, xa, xb, (lo, hi) in zip(names, oa, ob, ranges):
        ha, ea, na, edges = weighted_hist(xa, wa, lo, hi, 8)
        hb, eb, nb, _ = weighted_hist(xb, wb, lo, hi, 8)
        # normalise each weighted shape by its total |w| so the comparison is
        # of shapes, not of absolute normalisation
        sa = np.abs(ha).sum() or 1.0
        sb = np.abs(hb).sum() or 1.0
        ra, rb = ha / sa, hb / sb
        print("  %s (shape, weighted, normalised by sum|w|)" % name)
        print("    bin        %-14s %-14s diff/sigma" % (args.label_a, args.label_b))
        for k in range(len(ra)):
            sig = math.sqrt((ea[k] / sa) ** 2 + (eb[k] / sb) ** 2) or 1e-30
            print("    [%5.2f,%5.2f]  %+12.4e  %+12.4e  %+7.2f"
                  % (edges[k], edges[k + 1], ra[k], rb[k],
                     (ra[k] - rb[k]) / sig))
        print("    entries %d / %d, weighted sums %+.3e / %+.3e"
              % (na.sum(), nb.sum(), ha.sum(), hb.sum()))
        print()


if __name__ == "__main__":
    main()
