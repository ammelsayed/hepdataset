"""
Top-level CLI dispatcher for the hepdataset package.

Usage:
    hepdataset [SUBCOMMAND] [ARGS...]

If SUBCOMMAND is omitted, 'make' is used by default.
"""

import sys
from importlib import import_module


SUBCOMMANDS = {
    "make":             ("hepdataset.make_dataset",           "main"),
    "samples_reader":   ("hepdataset.samples_reader",         "main"),
    "merge_samples":    ("hepdataset.merge_samples",          "main"),
    "basic1_delphes":   ("hepdataset.loops.basic1_delphes",   "main"),
    "basic2_delphes":   ("hepdataset.loops.basic2_delphes",   "main"),
    "basic3_delphes":   ("hepdataset.loops.basic3_delphes",   "main"),
    "adaptive_delphes": ("hepdataset.loops.adaptive_delphes", "main"),
}


def _print_help():
    print(__doc__.strip())
    print("\nAvailable subcommands:")
    for name in SUBCOMMANDS:
        print(f"  {name}")
    print("\nRun 'hepdataset <subcommand> --help' for options.")


def main():
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return 0

    sub, rest = argv[0], argv[1:]

    # Unknown first token -> treat it as an argument to `make`
    if sub not in SUBCOMMANDS:
        sub, rest = "make", argv

    module_path, func_name = SUBCOMMANDS[sub]

    # Rewrite argv so argparse inside the subcommand sees a sensible prog name
    sys.argv = [f"hepdataset {sub}"] + rest

    module = import_module(module_path)
    return getattr(module, func_name)() or 0


if __name__ == "__main__":
    sys.exit(main())