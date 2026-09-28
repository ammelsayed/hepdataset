"""C++-bridge entry point for the explicit-getter Delphes loop.

Requires the optional extension built by ``make -C src/cpp``.
The Python loop uses direct getter tables; the bridge keeps branch and selector
behavior identical while exposing the same entry point to C++ callers.
"""
from .cpp_loop import call_loop, run_cli


def loop_tree(**loop_args):
    return call_loop("explicit_delphes", **loop_args)


def main():
    return run_cli("explicit_delphes")


if __name__ == "__main__":
    main()
