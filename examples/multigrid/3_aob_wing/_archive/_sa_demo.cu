// TODO : smooth prolongation
if constexpr (is_bsr && !Prolongation::structured) {
    int nlevels = mg.getNumLevels();
    for (int ilevel = 0; ilevel < nlevels; ilevel++) {
        int n_mm_iters = 6; // TBD on how to set this
        mg.grid[ilevel].smoothProlongation(n_mm_iters);
    }
}