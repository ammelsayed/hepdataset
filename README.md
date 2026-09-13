# HEPDataset

HEPDataset is a Python framework for constructing machine learning ready datasets from high energy physics ROOT files. The current workflow is designed for ROOT files produced by the Delphes fast detector simulation. It reads the `Delphes` event tree, selects reconstructed objects, computes configurable kinematic and event level quantities, assigns events to analysis channels, applies cross section and luminosity weights, and writes channel organized ROOT datasets.

The package is intended for phenomenological studies of signals and backgrounds at collider experiments. A typical workflow begins with Monte Carlo event generation, continues with detector response simulation using Delphes, and uses HEPDataset to transform the resulting ROOT files into a feature representation for a boosted decision tree, a neural network, or another machine learning framework.

The main design goal is to make high dimensional feature extraction straightforward and reproducible. The `branches_config.yml` card defines the objects and observables to read. The event loop can process many requested features during one traversal of each input file and can assign events to analysis channels during that same traversal. The processing of large files can be divided into independent event ranges and executed in parallel.

## Current status

HEPDataset is beta research software. The current production path uses PyROOT and Delphes classes. The repository also contains several loop implementations with different levels of fixed and adaptive feature access. Uproot based loops and additional C++ loop implementations are planned as complementary input and processing backends. The current command line workflow should be treated as the supported high level interface.

## Installation

Install the package in a Python virtual environment with:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install hepdataset
```

For a local checkout, install the project in editable mode:

```bash
git clone https://github.com/ammelsayed/hepdataset.git
cd hepdataset
python -m pip install -e .
```

The Python dependencies declared by the project include NumPy, pandas, PyYAML, tabulate, tqdm, Numba, mt2, and Uproot. The current Delphes processing path additionally requires a working PyROOT installation and a Delphes installation that provides the Delphes libraries, classes, and `ExRootAnalysis` headers.

Set `DELPHES_HOME` to the directory containing the Delphes installation when it is not located at the default path used by the current loader:

```bash
export DELPHES_HOME=/path/to/Delphes
```

The ROOT and Delphes environment must be configured before processing ROOT files. The input files must contain a Delphes tree named `Delphes` unless a custom loop explicitly implements a different convention.

## Command line usage

After installation, inspect the available options with:

```bash
hepdataset --help
```

The command accepts two YAML configuration files and an output directory:

```text
hepdataset [-h]
           [--luminosity FB_INV]
           [--loop-file MODULE]
           [--merge]
           [--max-workers N]
           [--n-chunks N]
           [--show-progress]
           [--debug]
           samples branches_config output_dir
```

A typical command is:

```bash
hepdataset \
    tests/samples_example1.yml \
    tests/branches_config_example1.yml \
    output \
    --luminosity 400 \
    --merge \
    --max-workers 8 \
    --n-chunks 16 \
    --show-progress
```

The positional arguments are as follows:

| Argument | Description |
| --- | --- |
| `samples` | YAML file containing signal and background processes, cross sections, and ROOT file paths. |
| `branches_config` | YAML file defining objects, representations, kinematic variables, event variables, and multi object combinations. |
| `output_dir` | Directory in which the per sample and optional merged ROOT files are written. |

The available options are:

| Option | Description |
| --- | --- |
| `--luminosity FB_INV` | Integrated luminosity used to calculate event weights. The default is `400`. |
| `--loop-file MODULE` | Loop module in `src/loops`. The default is `adaptive_delphes`. |
| `--merge` | Create `events.root` with `hadd` in each channel or region directory. Without this option, individual sample files are retained. |
| `--max-workers N` | Maximum number of worker processes running simultaneously. |
| `--n-chunks N` | Number of event range jobs created for each input ROOT file. |
| `--show-progress` | Display progress information and loop summaries. |
| `--debug` | Print debugging information from the selected loop. |

The package calculates a nominal event weight from the process cross section, the requested luminosity, and the total number of generated events. The current normalization assumes that the cross sections in the sample card are in pb and that the luminosity is in fb⁻¹. Users should verify the units and generator weight conventions for their own samples before using the output for physics interpretation.

## Sample configuration

The sample card groups processes under categories such as `Background` and `Signal`. Each process provides a cross section and one or more ROOT file paths:

```yaml
Background:
  ttbar:
    cross_section: 0.7528  # pb
    files:
      - /data/samples/ttbar_01.root
      - /data/samples/ttbar_02.root

Signal:
  mass_1000:
    cross_section: 0.0023793  # pb
    files:
      - /data/samples/signal_mass_1000.root
```

The sample reader counts events in the `Delphes` tree and skips processes that have no usable files or zero entries. A missing cross section is assigned the default value `1.0`, but production analyses should always provide an explicit value.

A complete example is available in [`tests/samples_example1.yml`](tests/samples_example1.yml). The paths in that file are examples and must be replaced with paths available in the local environment.

## Branch configuration

The branch configuration card defines the physics representation. It contains three main sections.

The `objects` section defines the number of objects, their representations, and the single object kinematic quantities to retain. A representative configuration is:

```yaml
objects:
  Lepton:
    count: 4
    representations: ["Lepton"]
    kinematics: ["Pt", "Eta", "Phi", "M", "Px", "Py", "Pz", "Energy", "IsolationVar"]
  FatJet:
    count: 2
    representations: ["FatJet", "SoftDroppedFatJet"]
    kinematics: ["Pt", "Eta", "Phi", "M", "Px", "Py", "Pz", "Energy", "Tau1", "Tau2", "Tau3"]
  Jet:
    count: 4
    representations: ["Jet"]
    kinematics: ["Pt", "Eta", "Phi", "M", "Px", "Py", "Pz", "Energy"]
  MET:
    count: 1
    representations: ["MET"]
    kinematics: ["MET", "Px", "Py", "Pz", "Eta", "Phi"]
```

The `event-variables` section defines global scalars and event shape observables:

```yaml
event-variables:
  global_scalars: ["HT", "LT", "ST", "Meff", "Significance_MET"]
  event_shapes: ["Sphericity", "Aplanarity", "Circularity", "Centrality"]
```

The `multi-objects` section defines the combinations of objects and the order of the derived observables:

```yaml
multi-objects:
  Nmax: 3
  combo_set: ["Lepton", "FatJet", "MET"]
  include_same_represenations: true
  basic_kinematics: ["Pt", "Eta", "Phi", "M", "Px", "Py", "Pz", "Energy"]
  2body_kinematics: ["DeltaR", "DeltaPhi", "DeltaEta", "MtW"]
  include_mt2: true
  include_trival_kinematics: false
  trival_kinematics: ["Sphericity", "Aplanarity", "Circularity", "Centrality", "ScalarSumPT", "VectorSumPT"]
```

The feature space can contain many single object, pairwise, higher order, and global observables. Depending on the configuration, the available feature categories include transverse momentum, pseudorapidity, azimuthal angle, mass, energy, Cartesian momentum components, tracking and reconstruction parameters, angular separations, transverse mass, `MT2`, scalar and vector sums, MET quantities, hadronic and leptonic activity, effective mass, and event shapes. The configuration system can generate up to approximately 12,000 candidate branches, although large configurations increase processing time and many branches may be undefined for a particular event sample.

Validate and inspect a branch configuration before processing:

```bash
python src/loops/branches_reader.py \
    tests/branches_config_example1.yml \
    --inspect
```

The example branch card is available at [`tests/branches_config_example1.yml`](tests/branches_config_example1.yml). Its object and feature definitions should be adapted to the actual branches available in the Delphes files and to the physics analysis.

## Output structure

Each input ROOT file is processed separately. For channelized loop modules, the output is organized by analysis channel and region:

```text
output/
  0L/
    JJ/
      background_ttbar_sample0.root
  1L/
    lepJ/
      background_ttbar_sample0.root
    lepJJ/
      signal_mass_1000_sample0.root
  2OSL/
    leplepJ/
      background_ttbar_sample0.root
```

The exact channel and region names are defined by the selected loop. The output trees contain the requested branches together with the event weights used by the loop. When `--merge` is supplied, the individual sample files in each channel or region are merged into `events.root` with ROOT's `hadd` utility:

```text
output/
  1L/
    lepJ/
      events.root
      background_ttbar_sample0.root
      signal_mass_1000_sample0.root
```

The merged file is convenient for downstream training or yield studies, while the individual files preserve the sample-level outputs until the merge has completed successfully.

## Parallel processing

The default dataset workflow uses the parallel loop machinery to divide each input file into independent entry ranges. Each worker reads its assigned range and writes isolated temporary ROOT files. The parent process merges the temporary results and then moves the channel outputs to the requested destination. This avoids concurrent writes to a shared ROOT file and allows the number of workers and chunks to be adjusted for the available CPU and storage resources.

For a small test sample, use one worker or a small number of chunks. For large samples, increase `--max-workers` and `--n-chunks` according to the available resources. More workers do not always produce better performance because ROOT input output and temporary file merging can become the limiting factors.

The repository also contains a separate utility for running the external `DelphesHepMC2` executable over HEPMC files. That utility is an upstream conversion step and is not required when Delphes ROOT files already exist.

## Python API

The package exports the high level `make_dataset` function. It can be called from a Python script:

```python
import hepdataset as hepds

hepds.make_dataset(
    samples="tests/samples_example1.yml",
    branches_config="tests/branches_config_example1.yml",
    output_dir="output",
    luminosity=400.0,
    loop_file="adaptive_delphes",
    merge=True,
    max_workers=8,
    n_chunks=16,
    show_progress=True,
    debug_loop=False,
)
```

The same function can be imported explicitly:

```python
from hepdataset import make_dataset

make_dataset(
    samples="samples.yml",
    branches_config="branches_config.yml",
    output_dir="output",
)
```

The high level function reads every configured input file, loads the selected loop module, calculates the process weight, runs the loop, writes the channel outputs, and optionally performs the final merges. Lower level loop functions are available in `src/loops` for researchers who need direct control over event ranges, output paths, or custom analysis behavior.

## Loop modules and extension points

The high level command loads a module from `src/loops` and requires that it provide a `loop_tree` function. The default module is `adaptive_delphes`. Other current loop modules include `basic1_delphes.py`, `basic2_delphes.py`, and `basic3_delphes.py`. These loops provide different levels of fixed and adaptive event feature handling.

A custom loop can be selected with:

```bash
hepdataset samples.yml branches_config.yml output \
    --loop-file my_loop_module
```

The current Delphes loops use PyROOT to read the event content. Future Uproot based loops and C++ loops are intended to provide alternative ROOT processing backends while preserving the same high level dataset construction model. The loop interface is designed around a ROOT input path and a channel organized result. Depending on the loop, the result can be a dictionary of trees or paths to written ROOT files.

Analysis channel classification is provided through the analysis channel logic, and object selection is provided through the object selection logic. These components are intended to become more configurable through dedicated YAML cards in future releases.

## Testing and validation

Install the test dependencies with:

```bash
python -m pip install -e '.[test]'
```

Run the available tests with:

```bash
pytest
```

Before using a generated dataset for a physics result, validate the following independently. Confirm that all input files contain the expected `Delphes` tree and branches. Check the object selection cut flow. Compare serial and parallel processing on a small sample. Verify the cross section and luminosity units. Inspect branch values, missing object conventions, generator weights, and event weights. Compare selected yields with an independent analysis implementation when possible.

## Contributing

Contributions are welcome in the form of issue reports, documentation improvements, tests, configuration improvements, new loop backends, physics feature implementations, performance improvements, and pull requests. Please open an issue before undertaking a substantial change so that the intended behavior can be discussed publicly.

A typical contribution workflow is:

```bash
git clone https://github.com/ammelsayed/hepdataset.git
cd hepdataset
python -m pip install -e '.[test,dev]'
git switch -c feature/short-description
# edit the source, tests, documentation, or configuration
git diff --check
pytest
git add path/to/changed/files
git commit -m "Describe the contribution"
git push origin feature/short-description
```

Open a pull request against the `main` branch and describe the scientific motivation, the affected input and output behavior, the validation performed, and any limitations. Contributions that change physics definitions should include numerical tests or controlled examples whenever possible. Contributions that change configuration schemas should update the example YAML cards and the README. Contributors should preserve the MIT license and avoid committing generated ROOT files, private data paths, credentials, or machine specific build artifacts.

The project is intended to follow open source practices relevant to Journal of Open Source Software submissions. These practices include a browsable repository, an issue tracker, public pull requests, a clear license, documentation, tests, and an explicit account of research use. See the [JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html#submission-requirements) and the [JOSS example paper](https://joss.readthedocs.io/en/latest/example_paper.html) for the journal's expectations.

## License

HEPDataset is distributed under the MIT License. See [`LICENSE`](LICENSE).

## Citation and contact

For questions, bug reports, and feature requests, use the [GitHub issue tracker](https://github.com/ammelsayed/hepdataset/issues). The JOSS manuscript and its independent bibliography are located in [`paper/paper.md`](paper/paper.md) and [`paper/paper.bib`](paper/paper.bib).
