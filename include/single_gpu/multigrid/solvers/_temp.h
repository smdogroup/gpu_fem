int inner_pre_smooth = pre_smooth * (double_smooth ? 1 << i_level : 1);
grids[i_level].smoothDefect(inner_pre_smooth, debug, inner_pre_smooth - 1);

grids[i_level + 1].restrict_defect(grids[i_level].d_defect);

coarse_solver->solve(grids[i_level].d_defect, grids[i_level].d_soln);

grids[i_level].prolongate(grids[i_level + 1].d_soln);

int inner_post_smooth = post_smooth * (double_smooth ? 1 << i_level : 1);
grids[i_level].smoothDefect(inner_post_smooth, debug, inner_post_smooth - 1);



coarse_solver->solve(grids[n_levels-1].d_defect, grids[n_levels-1].d_soln);