import os
import uproot
from yaml import safe_load



class SamplesReader:

    def __init__(self, yml_path):
        self.yml_path = yml_path

    def get_nb_events(self, file_path):
        """
        Returns the number of events in a ROOT file.
        """
        try:
            with uproot.open(file_path) as f:
                return f["Delphes"].num_entries
        except Exception:
            return 0

    # check root health only on first time running this script
    def clean_root_files(self, dir_list):
        valid,  rejected = [], []
        for path in dir_list:

            # Must exist on disk and be a regular file
            if not os.path.isfile(path):
                rejected.append(path)
                continue

           # Must carry the .root extension
            if not path.endswith(".root"):
                rejected.append(path)
                continue

            # Must be openable by uproot and expose a Delphes tree
            try:
                with uproot.open(path) as f:
                    if "Delphes" not in f:
                        rejected.append(path)
                        continue
                valid.append(path)
            except Exception:
                rejected.append(path)

        # Report the bad files
        if rejected:
            print(f"Found {len(rejected)} unvalid files.")
            for path in rejected:
                print(f" > {path}")

        # remove duplicates and return the valid list
        return list(set(valid))

    def inspect(self):
        data = self.read()
        for category, processes in data.items():
            for proc_name, proc_info in processes.items():
                for file_path in proc_info.get('files', []):
                    # Open the file and read 10 evenly spaced entries to check for basket corruption
                    try:
                        with uproot.open(file_path) as f:
                            if "Delphes" not in f:
                                print(f"Error: no Delphes tree in {file_path}")
                                return
                            tree = f["Delphes"]
                            n = tree.num_entries
                            if n == 0:
                                print(f"Warning: empty tree in {file_path}")
                                return
                            step = max(1, n // 10)
                            for entry in range(0, n, step):
                                tree.arrays(entry_start=entry, entry_stop=entry + 1, library="np")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        return data

    def read(self):

        with open(self.yml_path, 'r') as f:
            data = safe_load(f)
        
        to_remove = []
        for category, processes in data.items():
            for proc_name, proc_info in processes.items():
                
                # Check for the .root files
                # Drop processes without valid ROOT files
                if 'files' not in proc_info:
                    print(f"Warning: '{proc_name}' in '{category}' is missing 'files'. Removing this process.")
                    to_remove.append((category, proc_name))
                    continue

                proc_info['files'] = self.clean_root_files(proc_info['files'])
                if not proc_info['files']:
                    print(f"Warning: '{proc_name}' in '{category}' has no valid ROOT files! Removing this process.")
                    to_remove.append((category, proc_name))
                    continue
                
                # Read the number of events
                proc_info['nb_events'] = sum(self.get_nb_events(f) for f in proc_info['files'])
                   
                
                # Check for the cross section data and the k-factors
                # usually k-factors are pT dependent of the event
                # for v5.0.0 we are going to assume they are global
                if 'cross_section' not in proc_info:
                    print(f"Warning: '{proc_name}' in '{category}' is missing 'cross_section'. Going to set it to 1.0 by default.")
                    proc_info['cross_section'] = 1.0
                
                if 'cross_section_err_high' not in proc_info:
                    print(f"Warning: '{proc_name}' in '{category}' is missing 'cross_section_err_high'. Going to set it to 0.0 by default.")
                    proc_info['cross_section_err_high'] = 0.0

                if 'cross_section_err_low' not in proc_info:
                    print(f"Warning: '{proc_name}' in '{category}' is missing 'cross_section_err_low'. Going to set it to 0.0 by default.")
                    proc_info['cross_section_err_low'] = 0.0

                if 'k_factor' not in proc_info:
                    print(f"Warning: '{proc_name}' in '{category}' is missing 'k_factor'. Going to set it to 1.0 by default.")
                    proc_info['k_factor'] = 1.0
        
        # Remove the skipped processes now that iteration is done
        for category, proc_name in to_remove:
            data[category].pop(proc_name, None)

        return data

    def print_table(self, data, fmt="plain"):
        import pandas as pd
        from tabulate import tabulate

        # Build rows from the parsed samples dict, tagging category headers as sections
        rows = []
        for category, processes in data.items():
            rows.append({"Process": category, "N_gen": "", "sigma": ""})
            for name, info in processes.items():
                rows.append({"Process": name, "N_gen": f"{info['nb_events']:,}", "sigma": info['cross_section']})

        df = pd.DataFrame(rows)

        if fmt == "latex":

            # Wrap data cells in math mode, set LaTeX-style column names
            for c in  ("N_gen", "sigma"):
                df[c] = df[c].apply(lambda x: f"${x}$")

            df.columns = ["Process", "$N_{\\mathrm{gen}}$", "$\\sigma_{\\mathrm{LO}}$ [pb]"]
            print(df.to_latex(index=False, escape=False, column_format="lcc"))

        else:
            print(tabulate(df, headers='keys', tablefmt="grid", showindex=False))


if __name__ == "__main__":    

    import argparse
    from pprint import pprint
    
    parser = argparse.ArgumentParser(description="Parse a samples.yml file and print the resulting dictionary.")
    parser.add_argument("yml_file", type=str, help="Path to the samples.yml file")
    parser.add_argument("--inspect", action="store_true", help="Deep inspect ROOT files by reading events")
    parser.add_argument("--print", dest="print_fmt", nargs="?", const="plain", default=None, choices=["latex", "plain"], help="Print the samples table ('plain' by default, or 'latex')")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Read the yaml file
    samples_dict = SamplesReader(args.yml_file)
    if args.inspect:
        data = samples_dict.inspect()
    else:
        data = samples_dict.read()
    
    # Print the output in the requested format
    if args.print_fmt:
        samples_dict.print_table(data, args.print_fmt)
    else:
        print("Parsed YAML Dictionary:")
        pprint(data)