def NormalizeHist(hist):
    hist_new = hist.Clone()
    name = hist.GetName() + "_normalized"
    hist_new.SetName(name)
    hist_new.SetTitle(name)
    TotalN = hist.Integral()
    if TotalN != 0:
        hist_new.Scale(1.0 / TotalN)
    return hist_new