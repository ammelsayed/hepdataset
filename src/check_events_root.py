import ROOT, os

bad_files = []
for root, dirs, files in os.walk('.'):
    if 'events.root' not in files:
        continue
    p = os.path.join(root, 'events.root')
    rf = ROOT.TFile.Open(p)
    if not rf or rf.IsZombie():
        print(f'{p}: UNREADABLE')
        bad_files.append(p)
        continue
    rec = rf.TestBit(ROOT.TFile.kRecovered)
    bad = []
    for key in rf.GetListOfKeys():
        if key.GetClassName() != 'TTree':
            continue
        t = rf.Get(key.GetName())
        if not t:
            bad.append(key.GetName())
    if bad or rec:
        tag = 'RECOVERED-INCOMPLETE ' if rec else ''
        print(f'{p}: {tag}corrupt trees: {", ".join(bad)}')
        bad_files.append(p)
    rf.Close()

print('===SUMMARY===')
if not bad_files:
    print('ALL OK')
else:
    print(f'{len(bad_files)} BAD FILES:')
    for f in bad_files:
        print(' ', f)
