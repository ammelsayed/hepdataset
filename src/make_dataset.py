"""Public dataset-building API and command-line entry point."""

import argparse
import importlib
import inspect
import os
import shutil
import subprocess
from pathlib import Path


def _hadd(target_path, source_paths):
    command = ["hadd", "-f", str(target_path), *map(str, source_paths)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"hadd failed for {target_path} (exit {result.returncode}):\n{result.stderr}"
        )


def _load_loop(loop_file):
    module_name = Path(loop_file).stem
    if not module_name.isidentifier():
        raise ValueError(f"Invalid loop module name: {loop_file}")
    loops_dir = Path(__file__).resolve().parent / "loops"
    if str(loops_dir) not in os.sys.path:
        os.sys.path.insert(0, str(loops_dir))
    module = importlib.import_module(module_name)
    if not hasattr(module, "loop_tree"):
        raise ValueError(f"Loop module {module_name!r} does not define loop_tree")
    return module


def _sample_name(category, process_name, index):
    category_name = str(category).lower()
    if category_name.startswith("bkg"):
        category_name = "bkg"
    elif category_name.startswith("sig"):
        category_name = "signal"
    return f"{category_name}_{process_name}_sample{index}"


def _analysis_output_dirs(output_dir, loop_module):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(loop_module, "get_signal_regions"):
        signal_regions, keys = loop_module.get_signal_regions()
        for channel, regions in signal_regions.items():
            for region in regions:
                (output_dir / channel / region).mkdir(parents=True, exist_ok=True)
        return {key: output_dir / key.replace("_", "/", 1) for key in keys}
    if hasattr(loop_module, "get_analysis_channel_keys"):
        channels = loop_module.get_analysis_channel_keys()
        for channel in channels:
            (output_dir / channel).mkdir(parents=True, exist_ok=True)
    return {}


def _returned_paths(result):
    if isinstance(result, dict):
        result = result.values()
    elif isinstance(result, (str, os.PathLike)):
        result = [result]
    return list(result or [])


def _final_path(path, tree_name, output_dir, known_signal_regions):
    path = Path(path)
    filename = path.name
    root_suffix = ".root"
    if filename.startswith(tree_name + "_") and filename.endswith(root_suffix):
        channel = filename[len(tree_name) + 1:-len(root_suffix)]
        return output_dir / channel / "events.root"
    if filename == tree_name + root_suffix and path.parent.name in known_signal_regions:
        channel, region = path.parent.name.split("_", 1)
        return output_dir / channel / region / "events.root"
    return output_dir / "events.root"


def make_dataset(
    samples,
    branches_config,
    output_dir,
    return_pdDataframe=True,
    run_parallel=True,
    n_chunks=None,
    max_workers=None,
    loop_file="adaptive_delphes",
    luminosity=400.0,
    show_progress=False,
    debug_loop=False,
):
    """Build ROOT datasets from a samples YAML file.

    Each input file becomes a distinct tree named
    ``{category}_{process}_sampleN`` with ``N`` starting at zero. Final files
    are named ``events.root`` under their analysis channel or region.
    """
    try:
        from .loops.branches_reader import BranchesHandler
        from .loops.delphes import count_entries, load_delphes
        from .parallelization.delphes import loop_tree_parallel
        from .samples_reader import SamplesReader
    except ImportError as error:
        if "no known parent package" not in str(error):
            raise
        from loops.branches_reader import BranchesHandler
        from loops.delphes import count_entries, load_delphes
        from parallelization.delphes import loop_tree_parallel
        from samples_reader import SamplesReader

    samples = Path(samples).resolve()
    branches_config = Path(branches_config).resolve()
    output_dir = Path(output_dir).resolve()
    if not samples.is_file():
        raise FileNotFoundError(samples)
    if not branches_config.is_file():
        raise FileNotFoundError(branches_config)

    branch_reader = BranchesHandler(str(branches_config))
    if not branch_reader.is_valid():
        branch_reader.print_validation()
        raise ValueError(f"Invalid branch configuration: {branches_config}")

    loop_module = _load_loop(loop_file)
    load_delphes()
    known_signal_regions = set()
    for key in _analysis_output_dirs(output_dir, loop_module):
        known_signal_regions.add(key)

    sample_data = SamplesReader(str(samples)).read()
    temp_root = output_dir / ".sample_outputs"
    temp_root.mkdir(parents=True, exist_ok=True)
    loop_parameters = {"branches_config_path": str(branches_config)}
    accepted = set(inspect.signature(loop_module.loop_tree).parameters)
    loop_parameters = {
        key: value for key, value in loop_parameters.items() if key in accepted
    }
    final_inputs = {}

    try:
        for category, processes in sample_data.items():
            for process_name, metadata in processes.items():
                files = metadata.get("files", [])
                if not files:
                    print(f"Skipping {process_name}: no input ROOT files")
                    continue
                total_events = sum(count_entries(path) for path in files)
                if total_events <= 0:
                    print(f"Skipping {process_name}: zero events")
                    continue
                weight = (
                    float(metadata.get("cross_section", metadata.get("cross_section_[pb]", 1.0)))
                    * 1000.0
                    * luminosity
                    / total_events
                )

                for index, input_file in enumerate(files):
                    tree_name = _sample_name(category, process_name, index)
                    sample_temp = temp_root / tree_name
                    if run_parallel:
                        result = loop_tree_parallel(
                            inputRootFile=input_file,
                            treeName=tree_name,
                            eventWeight=weight,
                            temp_dir_path=str(sample_temp),
                            show_progress=show_progress,
                            debug_loop=debug_loop,
                            n_chunks=n_chunks,
                            max_workers=max_workers,
                            loop_tree=loop_module.loop_tree,
                            loop_kwargs=loop_parameters,
                        )
                    else:
                        sample_temp.mkdir(parents=True, exist_ok=True)
                        result = loop_module.loop_tree(
                            inputRootFile=input_file,
                            treeName=tree_name,
                            eventWeight=weight,
                            temp_dir_path=str(sample_temp),
                            show_progress=show_progress,
                            debug_loop=debug_loop,
                            **loop_parameters,
                        )
                    for path in _returned_paths(result):
                        destination = _final_path(
                            path, tree_name, output_dir, known_signal_regions
                        )
                        final_inputs.setdefault(destination, []).append(path)
        final_outputs = []
        for destination, paths in final_inputs.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            _hadd(destination, paths)
            final_outputs.append(destination)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    if not return_pdDataframe:
        return None
    try:
        import pandas as pd
        import uproot
    except ImportError as error:
        raise ImportError(
            "return_pdDataframe=True requires the optional 'uproot' package"
        ) from error
    frames = []
    for path in final_outputs:
        with uproot.open(path) as root_file:
            for tree in root_file.values():
                if hasattr(tree, "arrays"):
                    frames.append(tree.arrays(library="pd"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build HEP datasets from ROOT samples.")
    parser.add_argument("samples")
    parser.add_argument("branches_config")
    parser.add_argument("output_dir")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--n-chunks", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--loop-file", default="adaptive_delphes")
    args = parser.parse_args(argv)
    make_dataset(
        args.samples,
        args.branches_config,
        args.output_dir,
        return_pdDataframe=False,
        run_parallel=not args.serial,
        n_chunks=args.n_chunks,
        max_workers=args.max_workers,
        loop_file=args.loop_file,
    )


if __name__ == "__main__":
    main()
