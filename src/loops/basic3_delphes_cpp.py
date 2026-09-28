"""C++-bridge entry point for the fixed-schema, channelized basic3 loop."""
from .cpp_loop import call_loop, run_cli


def loop_tree(**loop_args):
    return call_loop("basic3_delphes", **loop_args)


def main():
    return run_cli("basic3_delphes")


if __name__ == "__main__":
    main()
