"""C++-bridge entry point for the fixed-schema basic1 Delphes loop."""
from .cpp_loop import call_loop, run_cli


def loop_tree(**loop_args):
    return call_loop("basic1_delphes", **loop_args)


def main():
    return run_cli("basic1_delphes")


if __name__ == "__main__":
    main()
