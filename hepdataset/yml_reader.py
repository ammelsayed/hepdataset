import argparse
from yaml import safe_load 
from pprint import pprint

def read_samples(yml_path):
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
    with open(yml_path, 'r') as f:
        data = safe_load(f)
        
    # simple validation to warn if a process is missing required keys
    for category, processes in data.items():
        for proc_name, proc_info in processes.items():
            if 'cross_section' not in proc_info or 'files' not in proc_info:
                print(f"Warning: '{proc_name}' in '{category}' is missing 'cross_section' or 'files'!")
                
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse a samples.yml file and print the resulting dictionary."
    )
    parser.add_argument(
        "yml_file", 
        type=str, 
        help="Path to the samples.yml file"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Read the yaml file
    samples_dict = read_samples(args.yml_file)
    
    # Print the output nicely
    print("Parsed YAML Dictionary:")
    pprint(samples_dict)