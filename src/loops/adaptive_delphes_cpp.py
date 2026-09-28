"""C++-bridge entry point for the adaptive Delphes loop.

Requires the optional extension built by ``make -C src/cpp``.
The physics/event-processing implementation is shared with adaptive_delphes.
"""
from .cpp_loop import call_loop, run_cli


def loop_tree(**loop_args):
    return call_loop("adaptive_delphes", **loop_args)


def main():
    return run_cli("adaptive_delphes")


if __name__ == "__main__":
    main()
