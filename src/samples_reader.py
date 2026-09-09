import os
import ROOT
import argparse
from yaml import safe_load 
from pprint import pprint


class SamplesReader:

    def __init__(self, yml_path):
        self.yml_path = yml_path
        self.check_root = False 

    def get_nb_events(self, file_path):
        """
        Returns the number of events in a ROOT file.
        """
        root_file = ROOT.TFile.Open(file_path)
        try:
            if not root_file or root_file.IsZombie():
                return 0
            tree = root_file.Get("Delphes")
            if not tree:
                return 0
            return tree.GetEntries()
        finally:
            if root_file and not root_file.IsZombie():
                root_file.Close()

    # check root health only on first time running this script
    def clean_root_files(self, dir_list):

        valid = [] 
        for path in dir_list:
            if not os.path.isfile(path):
                continue
            if not self.check_root:
                valid.append(path)
                continue
            root_file = ROOT.TFile.Open(path)
            try:
                if not root_file or root_file.IsZombie():
                    continue
                if not root_file.Get("Delphes"):
                    continue
                valid.append(path)
            finally:
                if root_file and not root_file.IsZombie():
                    root_file.Close()

        n_unvalid = len(dir_list) - len(valid)
        if n_unvalid > 0:
            print(f"Found {n_unvalid} unvalid files.")
            print(f" > Input: {dir_list}")

        # remove duplicates and return the valid list
        return list(set(valid)) 

    def read(self):
        """
        Reads a samples.yml file and returns a dictionary structured as:
        {
            "Background": {
                "process_name": {"cross_section": float, "files": [list_of_paths]},
                ...
            },
            "Signal": {
                "process_name": {"cross_section": float, "files": [list_of_paths]},
                ...
            }
        }
        """
        with open(self.yml_path, 'r') as f:
            data = safe_load(f)
            
        for category, processes in data.items():
            for proc_name, proc_info in processes.items():
                if 'files' in proc_info:
                    proc_info['files'] = self.clean_root_files(proc_info['files'])

                    if not proc_info['files']:
                        print(f"Warning: '{proc_name}' in '{category}' has no valid ROOT files!")
                    
                    proc_info['nb_events'] = sum(self.get_nb_events(f) for f in proc_info['files'])
                   
                    if 'cross_section' not in proc_info:
                        print(f"Warning: '{proc_name}' in '{category}' is missing 'cross_section'. Going to set it to 1.0 by default.")
                        proc_info['cross_section'] = 1.0

                else:
                    print(f"Warning: '{proc_name}' in '{category}' is missing 'files'. Going to set it to an empty list by default.")
                    proc_info['files'] = []
                    proc_info['nb_events'] = 0
                    if 'cross_section' not in proc_info:
                        print(f"Warning: '{proc_name}' in '{category}' is missing 'cross_section'. Going to set it to 1.0 by default.")
                        proc_info['cross_section'] = 1.0
                    
        return data



if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description="Parse a samples.yml file and print the resulting dictionary.")
    parser.add_argument("yml_file", type=str, help="Path to the samples.yml file")
    parser.add_argument("--check-root", action="store_true", help="Check the health of ROOT files")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Read the yaml file
    samples_dict = SamplesReader(args.yml_file)
    samples_dict.check_root = args.check_root
    data = samples_dict.read()
    
    # Print the output nicely
    print("Parsed YAML Dictionary:")
    pprint(data)