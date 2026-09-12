#!/usr/bin/env python3
"""Run a Delphes loop for every sample file and optionally merge the outputs."""

import argparse
import importlib
import inspect
import os
from pathlib import Path


def _load_loop(loop_file):
    module_name = Path(loop_file).stem
    if not module_name.isidentifier():
        raise ValueError(f"Invalid loop module name: {loop_file}")

    loops_dir = Path(__file__).resolve().parent / "loops"
    if str(loops_dir) not in os.sys.path:
        os.sys.path.insert(0, str(loops_dir))

    module = importlib.import_module(module_name)
    if not hasattr(module, "loop_tree"):
        raise ValueError(f"Loop module {module_name!r} has no loop_tree method")
    return module


def _sample_name(category, process_name, index):
    category = str(category).lower()
    category = "background" if category.startswith("bkg") else category
    category = "signal" if category.startswith("sig") else category
    return f"{category}_{process_name}_sample{index}"


def _paths_from_result(result):
    if isinstance(result, dict):
        return list(result.values())
    return [result]


def make_dataset(
    samples,
    branches_config,
    output_dir,
    luminosity=400.0,
    loop_file="adaptive_delphes",
    merge=False,
):
    """Process every input ROOT file and optionally create merged events.root files.

    Each input file is written separately as ``tree_name.root``. For channelized
    loops the path is ``output_dir/channel/region/tree_name.root``. With
    ``merge=True``, files in each channel/region directory are merged into
    ``events.root`` using ``hadd``; the individual sample files remain intact.
    """
    try:
        from .loops.delphes import count_entries, load_delphes
        from .samples_reader import SamplesReader
        from .loops.parallelization.delphes import hadd_files
    except ImportError:
        from loops.delphes import count_entries, load_delphes
        from samples_reader import SamplesReader
        from loops.parallelization.delphes import hadd_files

    samples = Path(samples).resolve()
    branches_config = Path(branches_config).resolve()
    output_dir = Path(output_dir).resolve()
    if not samples.is_file():
        raise FileNotFoundError(samples)
    if not branches_config.is_file():
        raise FileNotFoundError(branches_config)

    loop_module = _load_loop(loop_file)
    loop_parameters = {"branches_config_path": str(branches_config)}
    accepted = inspect.signature(loop_module.loop_tree).parameters
    loop_parameters = {
        key: value for key, value in loop_parameters.items() if key in accepted
    }
    sample_data = SamplesReader(str(samples)).read()
    load_delphes()
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_inputs = {}
    for category, processes in sample_data.items():
        for process_name, metadata in processes.items():
            input_files = metadata.get("files", [])
            if not input_files:
                continue

            total_events = sum(count_entries(path) for path in input_files)
            if total_events == 0:
                continue
            event_weight = (
                float(metadata.get("cross_section", 1.0))
                * 1000.0
                * luminosity
                / total_events
            )

            for index, input_file in enumerate(input_files):
                tree_name = _sample_name(category, process_name, index)
                result = loop_module.loop_tree(
                    inputRootFile=input_file,
                    treeName=tree_name,
                    eventWeight=event_weight,
                    output_dir=str(output_dir),
                    output_file_name=f"{tree_name}.root",
                    show_progress=False,
                    **loop_parameters,
                )

                if merge:
                    for sample_path in _paths_from_result(result):
                        sample_path = Path(sample_path)
                        merge_path = sample_path.with_name("events.root")
                        merge_inputs.setdefault(merge_path, []).append(sample_path)

    if merge:
        for target_path, source_paths in merge_inputs.items():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not hadd_files(str(target_path), [str(path) for path in source_paths]):
                raise RuntimeError(f"hadd failed for {target_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Process every ROOT file in a samples YAML card."
    )
    parser.add_argument("samples", help="Samples YAML file.")
    parser.add_argument("branches_config", help="Branches configuration YAML file.")
    parser.add_argument("output_dir", help="Directory for per-sample ROOT files.")
    parser.add_argument(
        "--luminosity", type=float, default=400.0, metavar="FB_INV",
        help="Integrated luminosity used for event weights (default: 400).",
    )
    parser.add_argument(
        "--loop-file", default="adaptive_delphes", metavar="MODULE",
        help="Loop module in src/loops (default: adaptive_delphes).",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Create events.root with hadd in every channel/region directory.",
    )
    args = parser.parse_args(argv)
    make_dataset(
        samples=args.samples,
        branches_config=args.branches_config,
        output_dir=args.output_dir,
        luminosity=args.luminosity,
        loop_file=args.loop_file,
        merge=args.merge,
    )


if __name__ == "__main__":
    main()
