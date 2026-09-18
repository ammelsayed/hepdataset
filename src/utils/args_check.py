
def args_check(
    eventWeight = None,
    cross_section = None,
    luminosity = None,
    start_entry = 0,
    end_entry = None,
    output_dir = None,
    output_file_name = "events.root",
    allow_overwrite = False,
):

    # Read the input file
    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)

    # Basic checks
    numberOfEntries = TreeReader.GetEntries()
    if start_entry < 0 or start_entry > numberOfEntries:
        raise ValueError(f"start_entry must be between 0 and {numberOfEntries}")
    if end_entry is None or end_entry > numberOfEntries:
        end_entry = numberOfEntries
    if end_entry < start_entry:
        raise ValueError("end_entry must be greater than or equal to start_entry")

    numberOfProcessedEntries = end_entry - start_entry
    if eventWeight is None:
        if (cross_section is None) != (luminosity is None):
            raise ValueError("cross section and luminosity must be provided together")
        if cross_section is not None and luminosity is not None:
            if numberOfEntries == 0:
                raise ValueError("Cannot calculate an event weight for an empty ROOT file")
            eventWeight = cross_section * luminosity / numberOfProcessedEntries
        else:
            eventWeight = 1.0

    if show_progress:
        print(f"Reading ROOT file: {inputRootFile}")
        print(f"Total events in file: {numberOfEntries}")
        print(f"Processing events: {start_entry} to {end_entry - 1} ({numberOfProcessedEntries} events)")
        print(f"Event weight: {eventWeight}")
    
    # Check output_dir is None, we use "./HEPDatasetOutput" as defaul
    if output_dir is None:
        output_dir = "./HEPDatasetOutput"

    # Check if the output_dir already exists or not
    # Allow overwrite if allow_overwrite == True
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise ValueError(f"output_dir is not a directory: {output_dir}")
        
        print(f"The output directory exists; overwriting its contents: {output_dir}")
        if not allow_overwrite:
            raise FileExistsError(f"Output directory already exists, cannot write there: {output_dir}")
    else:
        print(f"Creating output directory : {output_dir}")
        os.makedirs(output_dir)
    
    # Check output_file_name
    if not output_file_name or not output_file_name.endswith(".root"):
        raise ValueError("output_file_name must end with .root!")