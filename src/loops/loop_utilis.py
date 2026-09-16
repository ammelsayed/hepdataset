import os
import time
import argparse
from parallel_loop import add_parallel_arguments, run_in_parallel, format_time

LOOP_ARGUMENTS = {
    "inputRootFile": dict(
        cli="input_root_file",
        default=None,
        kwargs=dict(
            type=str,
            help="Path to the input Delphes ROOT file.",
        ),
    ),
    "treeName": dict(
        cli="--tree-name",
        default="Delphes",
        kwargs=dict(
            type=str,
            metavar="",
            help="Name of the output TTree.",
        ),
    ),
    "eventWeight": dict(
        cli="--event-weight",
        default=None,
        kwargs=dict(
            type=float,
            metavar="",
            help="Per-event weight. If omitted, computed from cross section and luminosity.",
        ),
    ),
    "cross_section": dict(
        cli="--cross-section",
        default=None,
        kwargs=dict(
            type=float,
            metavar="",
            help="Cross section in fb. Used with --luminosity.",
        ),
    ),
    "luminosity": dict(
        cli="--luminosity",
        default=None,
        kwargs=dict(
            type=float,
            metavar="",
            help="Integrated luminosity in fb^-1. Used with --cross-section.",
        ),
    ),
    "start_entry": dict(
        cli="--start-entry",
        default=0,
        kwargs=dict(
            type=int,
            metavar="",
            help="Entry to start processing from.",
        ),
    ),
    "end_entry": dict(
        cli="--end-entry",
        default=None,
        kwargs=dict(
            type=int,
            metavar="",
            help="Entry to stop processing at.",
        ),
    ),
    "show_progress": dict(
        cli="--show-progress",
        default=False,
        kwargs=dict(
            action="store_true",
            help="Show a progress bar during processing.",
        ),
    ),
    "debug_loop": dict(
        cli="--debug",
        default=False,
        kwargs=dict(
            action="store_true",
            help="Show debug information during processing.",
        ),
    ),
    "output_dir": dict(
        cli="--output-dir",
        default=".",
        kwargs=dict(
            type=str,
            metavar="",
            help="Directory where the output ROOT file will be written.",
        ),
    ),
    "output_file_name": dict(
        cli="--output-file-name",
        default="events.root",
        kwargs=dict(
            metavar="",
            help="Name of the output ROOT file.",
        ),
    ),
    "overwrite": dict(
        cli="--overwrite",
        default=False,
        kwargs=dict(
            action="store_true",
            help="Allow overwriting the output directory if it already exists.",
        ),
    ),
    "branches_config_path": dict(
        cli="--branches-config-path",
        default=None,
        kwargs=dict(
            type=str,
            metavar="",
            help="Path to the branches configuration YAML.",
        ),
    ),
}

def add_loop_arguments(parser):
    for key, spec in LOOP_ARGUMENTS.items():
        kwargs = dict(spec["kwargs"])
        # Positional arguments are required; only set default for optional flags.
        if spec["cli"].startswith("-"):
            kwargs.setdefault("default", spec["default"])
        parser.add_argument(spec["cli"], dest=key, **kwargs)

def make_loop_kwargs(args):
    return {key: getattr(args, key) for key in LOOP_ARGUMENTS}


def apply_loop_defaults(loop_args):
    """Fill in defaults for any key the caller did not supply."""
    filled = dict(loop_args)
    for key, spec in LOOP_ARGUMENTS.items():
        if filled.get(key) is None and spec["default"] is not None:
            filled[key] = spec["default"]
    return filled

def check_loop_args(loop_args, numberOfEntries):

    loop_args = apply_loop_defaults(loop_args)

    # Check start and end entries
    start_entry = loop_args["start_entry"]
    end_entry = loop_args["end_entry"]
    if start_entry < 0 or start_entry > numberOfEntries:
        raise ValueError(f"start_entry must be between 0 and {numberOfEntries}")
    if end_entry is None or end_entry > numberOfEntries:
        end_entry = numberOfEntries
    if end_entry < start_entry:
        raise ValueError("end_entry must be greater than or equal to start_entry")

    # Actual number of entries going to processed
    numberOfProcessedEntries = end_entry - start_entry

    # Check event weights
    eventWeight = loop_args["eventWeight"]
    cross_section = loop_args["cross_section"]
    luminosity = loop_args["luminosity"]
    if eventWeight is None:
        if (cross_section is None) != (luminosity is None):
            raise ValueError("cross section and luminosity must be provided together")
        if cross_section is not None and luminosity is not None:
            if numberOfProcessedEntries == 0:
                raise ValueError("Cannot compute an event weight when no entries will be processed")
            eventWeight = cross_section * luminosity / numberOfProcessedEntries
        else:
            eventWeight = 1.0

    # Validate / create output_dir 
    output_dir = loop_args["output_dir"]
    overwrite = loop_args["overwrite"]
    if output_dir is not None:
        if os.path.exists(output_dir):
            if not os.path.isdir(output_dir):
                raise ValueError(f"output_dir is not a directory: {output_dir}")
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists, cannot write there: {output_dir}"
                )
        else:
            print(f"Creating output directory : {output_dir}")
            os.makedirs(output_dir)

    # Correct the loop arguments
    loop_args["start_entry"] = start_entry
    loop_args["end_entry"] = end_entry
    loop_args["numberOfEntries"] = numberOfEntries
    loop_args["numberOfProcessedEntries"] = numberOfProcessedEntries
    loop_args["eventWeight"] = eventWeight
    loop_args["output_dir"] = output_dir

    if loop_args["show_progress"]:
        print(f"Reading ROOT file: {loop_args['inputRootFile']}")
        print(f"Total number of events: {numberOfEntries}")
        print(f"Processing events: {start_entry} to {end_entry - 1} ({numberOfProcessedEntries} events)")
        print(f"Weight per-event: {eventWeight}")

    return loop_args


def run_loop_cli(loop_tree, description="Process a Delphes ROOT file and write a flat tree with selected events."):
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description=description)
    add_loop_arguments(parser)
    add_parallel_arguments(parser)
    args = parser.parse_args()

    loop_kwargs = make_loop_kwargs(args)

    if args.parallel:
        out_path = run_in_parallel(
            loop_tree_method=loop_tree,
            max_workers=args.max_workers,
            n_chunks=args.n_chunks,
            merge_method=args.merge_method,
            **loop_kwargs
        )
    else:
        out_path = loop_tree(**loop_kwargs)

    print(f"Finished in {format_time(time.perf_counter() - start_time)}.")

    # print(f"\nOutput written to:")
    # print(out_path)
    
    return out_path