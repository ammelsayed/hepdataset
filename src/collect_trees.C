void collect_trees()
{
    TFile *out = new TFile("events.root", "RECREATE");

    void *dirp = gSystem->OpenDirectory(".");
    const char *name;
    while ((name = gSystem->GetDirEntry(dirp))) {
        TString fname = name;
        if (!fname.EndsWith(".root")) continue;
        if (fname == "events.root")   continue;

        TString treeName = fname;
        treeName.Remove(treeName.Length() - 5);

        TFile *f = TFile::Open(fname);
        if (!f || f->IsZombie()) continue;

        TTree *t = (TTree*)f->Get(treeName);
        if (t) {
            out->cd();
            TTree *newt = t->CloneTree(-1, "fast");
            newt->Write();
            printf("copied %s\n", treeName.Data());
        }
        f->Close();
        delete f;
    }
    gSystem->FreeDirectory(dirp);
    out->Close();
    delete out;
}