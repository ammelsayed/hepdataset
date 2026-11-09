from pathlib import Path
from event_selection import EventSelector

input_dir = Path("HepDataSet_0to3L_SplitByLeptonFlavour").resolve()

# read and merge for the backgrounds
event_selection_dir = input_dir / "EventSelection"
config_file = Path("event_selection_0to3L_SplitByLeptonFlavour.yml")
background_selectors = []
signal_selectors = []

for process_dir in sorted(event_selection_dir.iterdir()):
    if not process_dir.is_dir():
        continue
    is_background = process_dir.name.startswith("bkg_")
    if not is_background and not process_dir.name.startswith("sig_"):
        continue

    root_files = list(process_dir.glob("*.root"))
    if len(root_files) != 1:
        raise ValueError(f"Expected one ROOT file in {process_dir}, found {len(root_files)}")

    selector = EventSelector(config_file).ReadEventSelectionSummary(root_files[0], process_dir.name)
    if is_background:
        background_selectors.append(selector)
    else:
        signal_selectors.append((process_dir.name, selector))

if not background_selectors:
    raise ValueError(f"No bkg_ process directories found in {event_selection_dir}")

report = [
    "# Event Selection Report",
    "## Combined backgrounds",
    EventSelector.Merge(background_selectors).EventSelectionSummaryMarkdown(),
]
for process_name, selector in signal_selectors:
    report.extend([f"## {process_name}", selector.EventSelectionSummaryMarkdown()])

report_file = input_dir / "event_selection_report.md"
report_file.write_text("\n\n".join(report) + "\n", encoding="utf-8")
print(f"Wrote {report_file}")
