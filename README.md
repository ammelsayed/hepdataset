# HEPDataset

HEPDataset converts Delphes ROOT files into analysis-ready ROOT trees for high-energy-physics studies. Version 5 reorganizes the project around a package-installed command-line interface, configurable branch cards, reusable selection modules, and optional parallel event processing in Python or native C++.

## Requirements

HEPDataset requires Python 3.8 or newer and a working [ROOT](https://root.cern/) installation with PyROOT and the Delphes classes available. ROOT is not installed by pip and must be installed separately through the ROOT/Delphes distribution appropriate for the analysis environment. The Python dependencies are installed by the package metadata.

The input files must be ROOT files containing the tree and branches expected by the selected loop. Use your local Delphes ROOT file for validation; the example samples card still contains placeholder paths that must be replaced with local files.

## Installation

Install the released package with:

```bash
python -m pip install hepdataset
```

Install the current repository in editable mode, including test and development tools, with:

```bash
git clone https://github.com/ammelsayed/hepdataset.git
cd hepdataset
python -m pip install -e '.[test,dev]'
```

The package version is defined in `pyproject.toml`. The v5 release line uses version `5.0.0`. Verify the installed distribution with:

```bash
python -m pip show hepdataset
hepdataset --help
```

## Command-line interface

The installed `hepdataset` command dispatches to the supported tools:

```text
hepdataset make
hepdataset samples_reader
hepdataset merge_samples
hepdataset basic1_delphes
hepdataset basic2_delphes
hepdataset basic3_delphes
hepdataset adaptive_delphes
hepdataset native_delphes
```

If the first argument is not a recognized subcommand, it is treated as an argument to `make`. The explicit form is recommended because it makes scripts unambiguous. Every subcommand provides its own help:

```bash
hepdataset make --help
hepdataset samples_reader --help
hepdataset adaptive_delphes --help
hepdataset native_delphes --help
```

## Creating a dataset

The main workflow reads a samples card, processes each ROOT file, and writes the selected output trees:

```bash
hepdataset make samples.yml \
  --branches-config-file branches_config.yml \
  --output-dir HEPDataset \
  --working-luminosity 400 \
  --loop-method adaptive_delphes
```

Important options are:

| Option | Meaning |
| --- | --- |
| `samples.yml` | Input samples card. YAML and JSON are supported by the samples reader. |
| `--branches-config-file PATH` | Branch and feature configuration. The packaged default is used when this option is omitted. |
| `--output-dir PATH` | Destination directory; defaults to `HEPDataset`. |
| `--working-luminosity VALUE` | Analysis luminosity used for event weights; defaults to `400.0`. |
| `--loop-method NAME` | Loop implementation, such as `adaptive_delphes`, `native_delphes`, `basic1_delphes`, `basic2_delphes`, or `basic3_delphes`. |
| `--merge-proc-samples` | Merge files belonging to the same process after processing. |
| `--max-workers N` | Maximum number of worker processes used for merging or parallel processing. |
| `--n-chunks N` | Number of event ranges used by a parallel loop. |
| `--show-progress` | Print processing and selection summaries. |
| `--background` | Detach the dataset run and write output to `--log-file`. |

For example:

```bash
hepdataset make samples.yml \
  --branches-config-file branches_config.yml \
  --output-dir output \
  --loop-method adaptive_delphes \
  --max-workers 4 \
  --n-chunks 8 \
  --merge-proc-samples \
  --show-progress
```

Background execution starts a detached Python process and prints the process-group termination command:

```bash
hepdataset make samples.yml --background --log-file hepdataset.log
```

The output layout depends on the selected loop. Channelized loops normally create directories such as:

```text
output/
  0L/JJ/
  1L/lepJ/
  1L/lepJJ/
  2OSL/leplepJ/
  ObjectSelection/
  EventSelection/
```

Each channel or region contains ROOT files with the selected branches. Selection summaries and histograms are written under `ObjectSelection` and `EventSelection` when output is enabled.

## Samples cards

A samples card is organized by category and process. Each process contains one or more ROOT files and its global normalization information:

```yaml
Background:
  ttbar:
    cross_section: 0.7528
    k_factor: 1.0
    files:
      - /data/ttbar/run_01_delphes.root
      - /data/ttbar/run_02_delphes.root
Signal:
  signal_1000:
    cross_section: 0.0023793
    k_factor: 1.0
    files:
      - /data/signal/m1000_delphes.root
```

The reader removes missing or invalid paths and fills omitted uncertainty and k-factor fields with documented defaults. It also calculates the total number of generated events using ROOT. Inspect or convert a card with:

```bash
hepdataset samples_reader samples.yml --inspect
hepdataset samples_reader samples.yml --print
hepdataset samples_reader samples.yml --print latex
hepdataset samples_reader samples.yml --json parsed.json
hepdataset samples_reader samples.yml --xml parsed.xml
```

The example card is [`tests/samples_example1.yml`](tests/samples_example1.yml). Its machine-specific paths must be replaced before use.

## Branch configuration

The adaptive and native loops read a YAML branch card. The packaged default is [`src/defaults/branches_config_minimal.yml`](src/defaults/branches_config_minimal.yml), and an editable example is [`tests/branches_config_example1.yml`](tests/branches_config_example1.yml). A card defines:

- object types, multiplicities, representations, and kinematic variables;
- global event variables and event shapes;
- multi-object combinations and their two-body observables;
- optional derived quantities such as transverse-mass and `MT2` features.

Use an explicit path when developing an analysis:

```bash
hepdataset adaptive_delphes input.root \
  --branches-config-path tests/branches_config_example1.yml \
  --output-dir output
```

The exact option names for each loop are available through its `--help` output. Branch names must match the objects exposed by the Delphes tree; a syntactically valid card can still request branches that are absent from a particular sample.

## Merging ROOT outputs

The standalone merge utility operates on a directory of per-sample ROOT outputs:

```bash
hepdataset merge_samples output --merge-samples
```

To merge the per-process files and create `events.root`:

```bash
hepdataset merge_samples output --merge
```

The merge operations remove source files only after the output entry count has been checked. Keep a backup if the source files are valuable.

## Python API

The top-level package exposes the dataset function lazily:

```python
from hepdataset import make_dataset

make_dataset(
    samples_file="samples.yml",
    branches_config_file="branches_config.yml",
    output_dir="output",
    loop_method="adaptive_delphes",
    working_luminosity=400.0,
)
```

The reusable implementation modules are organized as follows:

```text
src/
  cli.py                 # installed command dispatcher
  make_dataset.py        # multi-sample orchestration and background mode
  samples_reader.py      # YAML/JSON samples parsing and inspection
  merge_samples.py       # ROOT output merging
  core/                  # selections, kinematics, Delphes utilities, branch cards
  loops/                 # basic and adaptive loop implementations
  utils/                 # auxiliary utilities
  defaults/              # packaged default branch configuration
```

The loop modules share common argument validation and can be used through the dispatcher or imported as Python modules. ROOT-dependent modules should be imported only in an environment where PyROOT and the Delphes classes are configured.

## Parallel processing

Parallel event processing divides the selected entry range into independent chunks. For a small sample, start with one worker and one chunk. Increase `--max-workers` and `--n-chunks` only after checking memory use and ROOT I/O performance:

```bash
hepdataset adaptive_delphes input.root \
  --parallel \
  --max-workers 4 \
  --n-chunks 8 \
  --output-dir output
```

The Python loop methods use the package's Python parallel orchestration. The native C++ loop uses a separate approach: the coordinator obtains the entry count from the C++ executable, starts one native executable subprocess per entry range, waits for the child processes, and merges their ROOT files with ROOT's `hadd`. It does not use `ProcessPoolExecutor` and Python does not read/process the event entries in this mode. Parallel ROOT workflows should be validated against a serial run on a small input file.

## Native C++ loop

`native_delphes` is a native C++ implementation of the configurable adaptive Delphes loop. The C++ program reads the ROOT `Delphes` tree through `TChain`/`ExRootTreeReader`, performs lepton and fat-jet selections and channel classification, calculates configured observables (including configured MT2), and writes channel trees plus object/event selection metadata. There is no Python embedding or callback during event processing. The Python layer only validates the YAML branch card, exports a versioned tab-delimited branch manifest, and launches/waits for the C++ executable; this keeps the full Python `BranchesHandler` out of the C++ event loop.

Build requirements are a C++17 compiler, `make`, ROOT with `root-config`, and a Delphes build containing `libDelphes.so` and the ExRootAnalysis headers. Set the ROOT and Delphes paths, then build:

```bash
export ROOTSYS=/path/to/root
export PATH="$ROOTSYS/bin:$PATH"
export DELPHES_DIR=/path/to/Delphes
make -C src/loops/native_cpp
```

The standalone command accepts the same core range/output controls as the Python adaptive loop:

```bash
hepdataset native_delphes /path/to/tag_1_delphes_events.root \
  --branches-config-path src/defaults/branches_config_minimal.yml \
  --start-entry 0 --end-entry 1000 \
  --output-dir native-output --output-file-name events.root --overwrite
```

The compiled executable can also be invoked directly with a branch manifest:

```bash
src/loops/native_cpp/native_adaptive_delphes \
  --input-root-file input.root --branch-manifest branches.tsv \
  --start-entry 0 --end-entry 1000 --output-dir native-output --overwrite
```

The wrapper exports the manifest from the validated YAML card into a temporary file automatically. To use C++ subprocess sharding from `make_dataset`:

```bash
hepdataset make samples.yml \
  --branches-config-file src/defaults/branches_config_minimal.yml \
  --loop-method native_delphes --max-workers 4 --n-chunks 8 \
  --output-dir native-dataset
```

`--max-workers` caps concurrently running C++ subprocesses; `--n-chunks` controls how many entry ranges are created. The native mode writes ROOT selector summaries/histograms, not Python `ObjectSelector`/`EventSelector` objects, and returns no in-memory `TTree` objects. Use `adaptive_delphes` when those Python objects or Python-side plotting are required. The native mode processes the adaptive loop's lepton, fat-jet, small-R jet, MET, and scalar-HT configuration and uses the same branch card as its input schema; it does not currently implement the distinct fixed-schema `basic1_delphes`, `basic2_delphes`, or `basic3_delphes` loop contracts.

## Testing and validation

Install the test dependencies and run the test suite with:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Before using a generated dataset for a physics result, confirm that input trees and branches are correct, verify cross-section and luminosity units, inspect event weights and missing-object conventions, compare serial and parallel output on a controlled sample, and compare selection yields with an independent implementation when possible.

## Contributing

Contributions are welcome through issues and pull requests. Changes to physics definitions should include numerical tests or controlled examples. Changes to configuration schemas should update the example cards and this README. Please do not commit generated ROOT files, private data paths, credentials, or machine-specific build artifacts.

## License and citation

HEPDataset is distributed under the MIT License; see [`LICENSE`](LICENSE). The manuscript and bibliography are available in [`paper/paper.md`](paper/paper.md) and [`paper/paper.bib`](paper/paper.bib).
