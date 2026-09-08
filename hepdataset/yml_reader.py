from yaml import safe_load as yml_safe_load

def get_paths_from_yml(bkg_proc_name):
    print(f"Reading .root files paths for {bkg_proc_name}")
    with open('/data/ammelsayed/stuff/bkgModeling/bkg_directories.yml', 'r') as f:
        data = yml_safe_load(f)
    return clean(data[bkg_proc_name])