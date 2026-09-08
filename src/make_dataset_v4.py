#!/usr/bin/env python3
import os
import sys
import time
import ROOT
from tqdm import tqdm

script_nb_version = 3
print(f"Making datasets with script version : {script_nb_version}")

def main(samples_file_dir, config_file_dir, analysis_channels_definition):
    pass


def loop_samples(samples_file_dir, config_file_dir, analysis_channels_definition):

    print("\n")
    
    # output root files paths (final destination)
    paths = {}
    for channelName, regionNamesList in signal_regions.items():
        for regionName in regionNamesList:
            key = f"{channelName}_{regionName}"
            paths[key] = os.path.join(baseDir, channelName, regionName, outputRootFileName)

    # Ensure TempReaderOutput exists
    os.makedirs(tempReaderDir, exist_ok=True)

    # Collect per-sample merged files for each SR, then hadd into final output
    # sample_files[sr_key] = list of per-sample ROOT file paths in TempReaderOutput
    sample_files = {sr_key: [] for sr_key in signal_regions_keys}
    
    for proc, metadata in processes.items():
        inputRootFilesList = metadata['dir']
        TotalCrossSection = metadata['cross_section_[pb]']

        if not inputRootFilesList:
            print(f"Skipping {proc}: no input files.")
            continue

        print(f"Processing {proc}: {len(inputRootFilesList)} sample root files.")
        numEvents = count_entries(inputRootFilesList)
        if numEvents == 0:
            print(f"Skipping {proc}: zero events.")
            continue
        perEventWeight = TotalCrossSection * 1000 * 400 / numEvents
        print(f"    > Total number of events : {numEvents}")
        print(f"    > Per-event weight : {perEventWeight}")

        for idx, inputRootFile in enumerate(inputRootFilesList, start = 1):
            treeName = f"{proc}_sample{idx}"
            merged_sample = loop_tree_advanced(inputRootFile, treeName, sampleWeight=perEventWeight, signal_regions_keys=signal_regions_keys)
            for sr_key, sample_path in merged_sample.items():
                sample_files[sr_key].append(sample_path)
                print(f"  {treeName}: merged {sample_path} -> will go into {paths[sr_key]}")
            print("\n")

    # Final merge: hadd all per-sample files for each SR into the final output
    print("Final merge into ReaderOutput ..")
    for sr_key in signal_regions_keys:
        if not sample_files[sr_key]:
            continue
        final_path = paths[sr_key]
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        ok = hadd_files(final_path, sample_files[sr_key])
        if ok:
            n_samples = len(sample_files[sr_key])
            print(f"  Final: {n_samples} sample files -> {final_path}")
        else:
            print(f"  WARNING: final hadd failed for {sr_key}")

    # Clean up TempReaderOutput
    if os.path.isdir(tempReaderDir):
        shutil.rmtree(tempReaderDir, ignore_errors=True)
        print(f"Cleaned up {tempReaderDir}")


def make_datasets():

    signal_regions, signal_regions_keys = get_signal_regions(isLoose = LooseSR)
    print("Signal regions:")
    for key, values in signal_regions.items():
        print(f"{key:<5} : {values}")

    read(get_processes(), signal_regions, signal_regions_keys)


if __name__ == '__main__':
    