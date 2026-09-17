import os
import uproot
import json
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ROOT import TFile
from yaml import safe_load
from tqdm import tqdm


#######################################################
#######################################################
# Helper methods
######################################################
#######################################################

# def inspect(data):
#     for category, processes in data.items():
#         for proc_name, proc_info in processes.items():
#             for file_path in proc_info.get('files', []):
#                 # Open the file and read 10 evenly spaced entries to check for basket corruption
#                 try:
#                     with uproot.open(file_path) as f:
#                         if "Delphes" not in f:
#                             print(f"Error: no Delphes tree in {file_path}")
#                             return
#                         tree = f["Delphes"]
#                         n = tree.num_entries
#                         if n == 0:
#                             print(f"Warning: empty tree in {file_path}")
#                             return
#                         step = max(1, n // 10)
#                         for entry in range(0, n, step):
#                             tree.arrays(entry_start=entry, entry_stop=entry + 1, library="np")
#                 except Exception as e:
#                     print(f"Error reading {file_path}: {e}")
#     return data

def inspect(data):

    for category, processes in data.items():

        print(f"\nInspecting {category.lower()} .root files ...")
        nb_processes = len(processes)

        for idx, (proc_name, proc_info) in enumerate(processes.items(), start = 1):

            print(f"({idx}/{nb_processes}) {proc_name}")
            
            for file_path in tqdm(proc_info.get('files', [])):
                # Open the file and read 10 evenly spaced entries to check for basket corruption
                f = None
                try:
                    f = TFile.Open(file_path)
                    if not f or f.IsZombie():
                        print(f"Error reading {file_path}: cannot open file")
                        return
                    tree = f.Get("Delphes")
                    if not tree:
                        print(f"Error: no Delphes tree in {file_path}")
                        return
                    n = tree.GetEntries()
                    if n == 0:
                        print(f"Warning: empty tree in {file_path}")
                        return
                    step = max(1, n // 10)
                    for entry in range(0, n, step):
                        tree.GetEntry(entry)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                finally:
                    if f is not None:
                        f.Close()
    return data


def print_table(data, fmt="plain"):

    print("\nPrinting table ...")

    if fmt == "plain":
        header = f"{'Process':<20} {'N_gen':>8} {'sigma [fb]':>10} {'+err':>8} {'-err':>8} {'K_F':>6} {'w_i':>10}"
        print("=" * len(header))
        print(header)
        print("=" * len(header))
        for idx, (category, processes) in enumerate(data.items()):
            if idx > 0: print("-" * len(header))
            print(category)
            print("-" * len(header))
            for name, info in processes.items():
                nb_events = info['nb_events']
                sigma     = info['cross_section']        * 1e3
                sigma_eh  = info['cross_section_err_high'] * 1e3
                sigma_el  = info['cross_section_err_low']  * 1e3
                k_factor  = info['k_factor']
                w         = sigma / nb_events
                print(f"{name:<20} {nb_events:>8,} {sigma:>10.3f} {sigma_eh:>8.3f} {sigma_el:>8.3f} {k_factor:>6.3f} {w:>10.3g}")
        print("=" * len(header))
        return None


    if fmt == "latex":

        # error formatting for the latex table
        def fmt_with_err(val, high, low, digits=3):
            if (high != 0.0) or (low != 0.0):
                return f"${val:.{digits}f}^{{+{high:.{digits}f}}}_{{-{low:.{digits}f}}}$"
            return f"${val:.{digits}f}$"

        def fmt_sci(x, digits=3):
            mant, exp = f"{x:.{digits}e}".split("e")
            if int(exp) != 0:
                return rf"{mant} \times 10^{{{int(exp)}}}"
            return rf"{mant}"

        ncols   = 5
        header  = r"Process & $N_{\mathrm{gen}}$ & $\sigma_{\mathrm{LO}}$ [fb] & $K_{\mathrm{F}}$ & $w_{i}$ \\"
        out = []
        out.append(r"\begin{table}")
        out.append(r"\centering")
        out.append(r"\scriptsize")
        out.append(r"\setlength{\tabcolsep}{2.5pt}")
        out.append(r"\renewcommand{\arraystretch}{1.05}")
        out.append(r"\begin{tabular}{lcccc}")
        out.append(r"\toprule")
        out.append(header)
        out.append(r"\midrule")
        for category, processes in data.items():
            out.append(rf"\multicolumn{{{ncols}}}{{l}}{{\textbf{{{category}}}}} \\")
            out.append(r"\midrule")   
            for name, info in processes.items():
                nb_events = info['nb_events']
                sigma     = info['cross_section']        * 1e3
                sigma_eh  = info['cross_section_err_high'] * 1e3
                sigma_el  = info['cross_section_err_low']  * 1e3
                k_factor  = info['k_factor']
                w         = sigma / nb_events
                sigma_str = fmt_with_err(sigma, sigma_eh, sigma_el)
                out.append(f"{name} & ${nb_events:,}$ & {sigma_str} & ${k_factor}$ & ${fmt_sci(w)}$ \\\\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\end{table}")
        print("\n".join(out))
        return None


#######################################################
#######################################################
# Main Class
######################################################
#######################################################

class SamplesReader:

    def __init__(self, path):
        self.path = path

        # Must exist and be a regular file
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if not os.path.isfile(path):
            raise IsADirectoryError(f"Input path is not a regular file: {path}")

        # Detect format from extension
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yml", ".yaml"):
            self.fmt = "yaml"
        elif ext == ".json":
            self.fmt = "json"
        elif ext == ".xml":
            raise NotImplementedError("Reading from .xml is not currently supported. Please use .yml, .yaml, or .json instead.")
        else:
            raise ValueError(f"Unsupported file extension '{ext}'. Supported formats: .yml, .yaml, .json.")

    # def get_nb_events(self, file_path):
    #     """
    #     Returns the number of events in a ROOT file.
    #     """
    #     try:
    #         with uproot.open(file_path) as f:
    #             return f["Delphes"].num_entries
    #     except Exception:
    #         return 0

    def get_nb_events(self, file_path):
        # ROOT's TTree::GetEntries() is C++ and reads only the header;
        # orders of magnitude faster than uproot for this specific call
        f = TFile.Open(file_path)
        try:
            if not f or f.IsZombie():
                return 0
            t = f.Get("Delphes")
            return t.GetEntries() if t else 0
        finally:
            if f and not f.IsZombie():
                f.Close()

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

            # # Must be openable by uproot and expose a Delphes tree
            # try:
            #     with uproot.open(path) as f:
            #         if "Delphes" not in f:
            #             rejected.append(path)
            #             continue
            #     valid.append(path)
            # except Exception:
            #     rejected.append(path)
        
            valid.append(path)

        # Report the bad files
        if rejected:
            print(f"Found {len(rejected)} unvalid file(s):")
            for path in rejected:
                print(f" > {path}")

        # remove duplicates and return the valid list
        return list(set(valid))

    def read(self):

        if self.fmt == "yaml":
            with open(self.path, "r") as f:
                data = safe_load(f)
        else: 
            with open(self.path, "r") as f:
                data = json.load(f)
        
        to_remove = []
        for category, processes in data.items():

            print(f"Reading {category.lower()} processes ...")
            nb_processes = len(processes)

            for idx, (proc_name, proc_info) in enumerate(processes.items(), start = 1):
                
                print(f"({idx}/{nb_processes}) {proc_name}")

                # Check for the .root files
                # Drop processes without valid ROOT files
                if 'files' not in proc_info:
                    print(f"Warning: '{proc_name}' in '{category}' is missing 'files'. Removing this process.")
                    to_remove.append((category, proc_name))
                    continue
                
                print("Reading .root files ...")
                proc_info['files'] = self.clean_root_files(proc_info['files'])
                if not proc_info['files']:
                    print(f"Warning: '{proc_name}' in '{category}' has no valid ROOT files! Removing this process.")
                    to_remove.append((category, proc_name))
                    continue
                
                # Read the number of events
                print("Calculating total number of events ...")
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

    @staticmethod
    def to_json(data, path):
        """Dump the parsed samples dictionary to a JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def to_xml(data, path):
        """Dump the parsed samples dictionary to an XML file."""
        def safe(tag):
            tag = re.sub(r"[^A-Za-z0-9_.-]", "_", str(tag))
            return tag if tag[:1].isalpha() or tag[:1] == "_" else "_" + tag

        def build(parent, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    build(ET.SubElement(parent, safe(k)), v)
            elif isinstance(obj, list):
                for item in obj:
                    build(ET.SubElement(parent, "item"), item)
            else:
                parent.text = str(obj)

        root = ET.Element("samples")
        build(root, data)
        pretty = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(path, "w") as f:
            f.write(pretty)


def main():
    import argparse  
    parser = argparse.ArgumentParser(description="Parse a samples.yml file and print the resulting dictionary.")
    parser.add_argument("yml_file", type=str, help="Path to the samples file (.yml, .yaml, or .json)")
    parser.add_argument("--inspect", action="store_true", help="Deep inspect ROOT files by reading events")
    parser.add_argument("--print", dest="print_fmt", nargs="?", const="plain", default=None, choices=["latex", "plain"], help="Print the samples table ('plain' by default, or 'latex')")
    parser.add_argument("--json", type=str, metavar="PATH", help="Write the parsed YAML to a JSON file")
    parser.add_argument("--xml",  type=str, metavar="PATH", help="Write the parsed YAML to an XML file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Read the yaml file
    reader = SamplesReader(args.yml_file)
    data = reader.read()

    if args.inspect:
       inspect(data)
    
    if args.print_fmt:
        print_table(data, args.print_fmt)

    if args.json:
        reader.to_json(data, args.json)
        print(f"Wrote JSON -> {args.json}")

    if args.xml:
        reader.to_xml(data, args.xml)
        print(f"Wrote XML  -> {args.xml}")


if __name__ == "__main__":

    main()