#!/usr/bin/env python3
"""Bake a per-event polarisation weight into XWGTUP of an LHE file.

Usage:
    python3 bake_weight.py unweighted_events.lhe.gz --mode INT
    python3 bake_weight.py in.lhe[.gz] out.lhe[.gz] --mode INT|LL|LT|TT
    python3 bake_weight.py in.lhe --mode INT --ll-id 1001 --lt-id 1002,1003

With no output file given, the result is written next to the input as
weighted_events.lhe.gz.

The input must already carry polarisation weights per event, either three columns
from MG5 `reweight` (`change process p p > z{0} z{0}, z > l+ l-` ...) or four from
MadSpin (`keep_weight_for_polarization_vector [0, T]`, in which case LT is the sum
of the 0T and T0 columns).

Output: the same events with XWGTUP replaced by

    INT : w_incl - (w_LL + w_LT + w_TT)      (signed, may be negative)
    LL  : w_LL        LT : w_LT        TT : w_TT

and XSECUP / XERRUP / XMAXUP in <init> updated.  IDWTUP is left untouched
(MG5 writes -4, i.e. weighted events, negative weights allowed).
"""
import argparse
import gzip
import math
import os
import re
import sys

WT_DEF = re.compile(r"<weight\s+id=['\"]([^'\"]+)['\"]\s*>(.*?)</weight>", re.S)
WGT = re.compile(r"<wgt\s+id=['\"]([^'\"]+)['\"]\s*>\s*([-+0-9.eE]+)\s*</wgt>")
LABEL = re.compile(r"z[({]([^)}]*)[)}]")


def label_to_pol(txt):
    t = txt.strip().replace(" ", "").lower()
    if t in ("0", "l"):
        return "L"
    if t in ("t", "-1,1", "1,-1", "+,-"):
        return "T"
    if t in ("+", "r"):
        return "T+"
    if t in ("-", "l-"):
        return "T-"
    return None


def classify(text):
    """Return 'LL' / 'LT' / 'TT' for a weight-definition text, else None."""
    pols = [label_to_pol(m) for m in LABEL.findall(text)]
    pols = [p for p in pols if p]
    if len(pols) < 2:
        return None
    a, b = pols[0], pols[1]
    if a == "L" and b == "L":
        return "LL"
    if (a == "L" and b in ("T", "T+", "T-")) or (b == "L" and a in ("T", "T+", "T-")):
        return "LT"
    if a != "L" and b != "L":
        return "TT"
    return None


def open_any(path, mode):
    return gzip.open(path, mode + "t") if path.endswith(".gz") else open(path, mode)


def find_ids(lines, explicit):
    """Return {component: [weight ids]}; LT may collect several columns."""
    ids = {"LL": [], "LT": [], "TT": []}
    for line in lines:
        for wid, text in WT_DEF.findall(line):
            comp = classify(text)
            if comp and wid not in ids[comp]:
                ids[comp].append(wid)
    for comp, val in explicit.items():
        if val:
            ids[comp] = [v.strip() for v in val.split(",")]
    missing = [c for c in ("LL", "LT", "TT") if not ids[c]]
    if missing:
        sys.exit(
            "could not identify %s in <initrwgt>; inspect the file and pass "
            "--ll-id/--lt-id/--tt-id explicitly." % ", ".join(missing)
        )
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("infile")
    p.add_argument("outfile", nargs="?",
                   help="default: weighted_events.lhe.gz next to the input")
    p.add_argument("--mode", required=True, choices=["INT", "LL", "LT", "TT"])
    p.add_argument("--sum-id",
                   help="comma-separated weight ids whose sum is the polarised sum, "
                        "for a reweight that produced a single combined column "
                        "(INT mode only)")
    p.add_argument("--ll-id")
    p.add_argument("--lt-id")
    p.add_argument("--tt-id")
    a = p.parse_args()

    if a.outfile is None:
        a.outfile = os.path.join(os.path.dirname(os.path.abspath(a.infile)),
                                 "weighted_events.lhe.gz")

    with open_any(a.infile, "r") as f:
        lines = f.readlines()

    combined = [x.strip() for x in a.sum_id.split(",")] if a.sum_id else None
    if combined:
        if a.mode != "INT":
            sys.exit("--sum-id is only meaningful with --mode INT")
        ids = {"combined": combined}
    else:
        ids = find_ids(lines, {"LL": a.ll_id, "LT": a.lt_id, "TT": a.tt_id})
    print("input : %s" % a.infile)
    print("output: %s" % a.outfile)
    print("mode  : %s" % a.mode)
    print("polarisation columns: %s" % {k: v for k, v in ids.items()})
    def fmt(v):
        return "%+15.6e" % v if v is not None else "%15s" % "n/a"

    print()
    print("  %-6s %15s %15s %15s %15s %15s %15s" %
          ("event", "w_inclusive", "w_LL", "w_LT", "w_TT",
           "w_interference", "assigned w"))
    print("  " + "-" * 100)

    out, weights = [], []
    i, n, nevt = 0, len(lines), 0
    while i < n:
        line = lines[i]
        if line.startswith("<init>"):
            out.append(line)
            out.append(lines[i + 1])          # beam / PDF record
            out.append("@@XS@@\n")            # cross-section line, filled later
            i += 3
            continue
        if line.startswith("<event>"):
            out.append(line)
            i += 1
            hdr = lines[i].split()
            old_w = float(hdr[2])
            cur = {}
            j = i + 1
            while j < n and "</event>" not in lines[j]:
                for wid, val in WGT.findall(lines[j]):
                    cur[wid] = float(val)
                j += 1
            if combined:
                missing = [w for w in combined if w not in cur]
                if missing:
                    sys.exit("event %d lacks weights %s" % (nevt + 1, missing))
                w_ll, w_lt, w_tt = None, None, None
                s = sum(cur[w] for w in combined)
            else:
                missing = [w for c in ("LL", "LT", "TT") for w in ids[c] if w not in cur]
                if missing:
                    sys.exit("event %d lacks weights %s" % (nevt + 1, missing))
                w_ll = sum(cur[w] for w in ids["LL"])
                w_lt = sum(cur[w] for w in ids["LT"])
                w_tt = sum(cur[w] for w in ids["TT"])
                s = w_ll + w_lt + w_tt
            w_int = old_w - s
            new_w = w_int if a.mode == "INT" else sum(cur[w] for w in ids[a.mode])
            nevt += 1
            weights.append(new_w)
            print("  %-6d %s %s %s %s %s %s"
                  % (nevt, fmt(old_w), fmt(w_ll), fmt(w_lt), fmt(w_tt),
                     fmt(w_int), fmt(new_w)))
            out.append("%4d %4d %+.8e %+.8e %+.8e %+.8e\n" % (
                int(hdr[0]), int(hdr[1]), new_w,
                float(hdr[3]), float(hdr[4]), float(hdr[5])))
            out.extend(lines[i + 1:j + 1])
            i = j + 1
            continue
        out.append(line)
        i += 1

    if not weights:
        sys.exit("no events found")

    mean = sum(weights) / len(weights)
    var = sum((w - mean) ** 2 for w in weights) / max(len(weights) - 1, 1)
    err = math.sqrt(var / len(weights))
    xmax = max(abs(w) for w in weights)
    out = ["  %+.8e %+.8e %+.8e 1\n" % (mean, err, xmax)
           if l == "@@XS@@\n" else l for l in out]

    with open_any(a.outfile, "w") as f:
        f.writelines(out)

    neg = sum(1 for w in weights if w < 0)
    print("  " + "-" * 100)
    print("events: %d" % len(weights))
    print("mode %-3s  <w> = %+.6e pb   err = %.2e   max|w| = %.3e"
          % (a.mode, mean, err, xmax))
    print("negative weights: %d (%.1f%%)" % (neg, 100.0 * neg / len(weights)))
    print("written: %s" % a.outfile)


if __name__ == "__main__":
    main()
