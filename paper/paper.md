---
title: 'HEPDataset: A configurable framework for constructing machine-learning datasets from Delphes ROOT files'
tags:
  - Python
  - High Energy Physics
  - Machine Learning
authors:
  - name: A.M.M Elsayed
    orcid: 0000-0002-4955-4958
    equal-contrib: true
    affiliation: 1
  - name: Yusheng Wu
    corresponding: true
    orcid: 0000-0002-1528-4865
    equal-contrib: true 
    affiliation: 1
affiliations:
 - name: Department of Modern Physics and State Key Laboratory of Particle Detection and Electronics, University of Science and Technology of China, Hefei, China
   index: 1
date: 17 July 2026
bibliography: paper.bib
---
# Summary

Searches for physics beyond the Standard Model (BSM) at the LHC increasingly rely
on multivariate classifiers — boosted decision trees (BDTs) or neural networks —
trained on flat, per-event feature tables built from reconstructed-object
kinematics, rather than on a small number of hand-picked cut variables. Building
these tables from `Delphes` [@delphes] fast-detector-simulation output is,
however, a recurring and largely bespoke engineering task: every analysis
re-implements object quality selections, signal-region classification, N-body
kinematic combinatorics, and cross-section-weighted sample bookkeeping from
scratch.

`HEPDataset` is a Python package that generalizes this task into a
configurable pipeline. Users supply (i) a registry of background and signal
`Delphes` ROOT samples with associated cross sections, defined via YAML;
(ii) object definitions specifying which reconstructed objects and kinematic
branches to read, including automatically generated combinatorial N-body
observables ($\Delta R$, $\Delta \phi$, $M_{T}$, $M_{T2}$, and others) for
user-chosen object groupings; and (iii) a signal-region classification function
that maps per-event reconstructed objects to a signal-region label. The
package reads events via `ROOT`'s `TTreeReader`/`ExRootTreeReader` interfaces
[@root], applies the user's object and region logic, computes event-shape and
kinematic observables (some accelerated via `numba` [@numba]), and writes the
result to per-region tabular output (ROOT trees, Parquet, or CSV) suitable for
direct use with tools such as `XGBoost` [@xgboost]. Event processing is
parallelized across chunks of each sample, with per-chunk outputs merged via
`hadd`, allowing datasets built from many signal mass points and background
processes to scale across available cores without manual chunk management.

Machine learning has become an essential component of modern
high-energy physics (HEP), with applications ranging from event
classification and anomaly detection to particle reconstruction and
jet tagging. Despite the availability of mature software for event generation,
detector simulation, and data analysis, there remains no lightweight
framework dedicated to transforming simulated Delphes events into
machine-learning-ready tabular datasets.

HEPDataset is an open-source Python framework designed to bridge this
gap. The package reads Delphes ROOT files, performs configurable object
selection, computes user-defined physics observables, constructs
multi-object kinematic variables, and exports flattened datasets in
formats suitable for downstream machine-learning workflows.

The framework is analysis-independent and is configured through simple
YAML files together with optional user-defined Python analysis modules.
This allows users to describe physics objects, define signal regions,
specify feature sets, and process arbitrary collections of signal and
background samples without modifying the core software.

# Statement of need

Several mature, public frameworks exist for confronting BSM models with LHC
data using `Delphes`-level simulation: `MadAnalysis 5`
[@madanalysis5a; @madanalysis5b], `CheckMATE` [@checkmate1; @checkmate2], and
`Rivet` [@rivet]. These tools excel at *recasting* — reproducing a specific,
already-published experimental analysis (its cuts, signal regions, and
efficiencies) so that a new theoretical model can be tested against it. Their
unit of output is typically a small number of signal-region yields or
efficiencies, matched to a specific published cut-and-count or simplified-likelihood
analysis.

This is a different problem from the one facing an analysis that is *itself*
being designed around a multivariate classifier, before any public
recast-ready implementation exists. In that setting, what is needed is not a
yield in a fixed set of signal regions, but a large, flat table of per-event
kinematic features — spanning single-object kinematics, all relevant N-body
combinations, and event-shape variables — computed consistently across many
background processes and many signal mass points, correctly weighted by
cross section and luminosity, and split by an analysis-specific signal-region
definition that may itself evolve during BDT development. Assembling this by
hand for every new analysis leads to substantial duplicated engineering effort
across phenomenology groups, and to pipelines whose object/branch/region logic
is difficult to disentangle from I/O and parallelization concerns, making
them hard to validate or reuse.

`HEPDataset` targets this gap directly: it separates *what to read*
(object and kinematic definitions), *how to select* (per-object quality
selections and a user-supplied signal-region function), and *what samples to
run over* (a YAML-configured, cross-section-weighted sample registry) from the
I/O, chunking, and merging machinery, which is handled once, generically, for
any configuration. The result is a reusable tool for producing ML-ready
datasets from `Delphes` samples that complements, rather than duplicates,
existing recasting frameworks: it is intended for the dataset-construction
stage of a BDT- or NN-based search, upstream of tools such as `pyhf`
[@pyhf] that are used for the subsequent statistical inference.

Preparing machine-learning datasets is one of the most repetitive
tasks in collider phenomenology.
Although Delphes provides detector-level ROOT files and ROOT itself
offers flexible event access, each analysis typically develops a
custom event loop that performs object selection, computes derived
kinematic variables, flattens event information, and writes a dataset
for machine learning.

These analysis-specific implementations often duplicate substantial
amounts of code, making them difficult to reuse, maintain, or compare
across different analyses. 
Furthermore, introducing additional reconstructed objects or derived
features usually requires modifying multiple components of the analysis
pipeline.

HEPDataset addresses this problem by providing a configurable feature
engineering framework for Delphes events.
Rather than hard-coding variables inside the event loop, users specify

\begin{itemize}
\item physics objects,
\item object representations,
\item analysis selections,
\item signal-region definitions,
\item desired observables,
\item output formats,
\end{itemize}

which are automatically translated into a complete event-processing
pipeline.

# State of the field

A description of how this software compares to other commonly-used packages in the research area. 
If related tools exist, provide a clear “build vs. contribute” justification explaining your unique scholarly contribution and why existing alternatives are insufficient.

# Software design

An explanation of the trade-offs you weighed, the design/architecture you chose,
and why it matters for your research application.
This should demonstrate meaningful design thinking beyond a superficial code structure description.

# Acknowledgements

The author acknowledges the support of the University of Science and
Technology of China and valuable discussions with members of the
particle physics group.


# AI usage disclosure

Transparent disclosure of any use of generative AI in the software creation, documentation, 
or paper authoring. If no AI tools were used, state this explicitly. If AI tools were used, 
describe how they were used and how the quality and correctness of AI-generated content was verified.

# References
