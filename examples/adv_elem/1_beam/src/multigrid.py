

import numpy as np
import scipy as sp
import sys


def debug_plot(dof_per_node, grid, vec1, vec2):
    import matplotlib.pyplot as plt
    vpn = dof_per_node
    fig, ax = plt.subplots(vpn, 2, figsize=(12, 9))
    # print(f"{vec1.shape=} {vec2.shape=} {grid.num_nodes=}")
    for iv in range(vpn):
        ax[iv,0].plot(grid.xvec, vec1[iv::vpn])
        ax[iv,1].plot(grid.xvec, vec2[iv::vpn])
    plt.show()

def debug_plot_diff(dof_per_node, grid1, grid2, vec1, vec2):
    import matplotlib.pyplot as plt
    vpn = dof_per_node
    fig, ax = plt.subplots(vpn, 2, figsize=(12, 9))
    # print(f"{vec1.shape=} {vec2.shape=} {grid.num_nodes=}")
    for iv in range(vpn):
        ax[iv,0].plot(grid1.xvec, vec1[iv::vpn])
        ax[iv,1].plot(grid2.xvec, vec2[iv::vpn])
    plt.show()

class VcycleSolver:
    def __init__(
        self,
        grids:list,
        smoothers:list,
        n_cycles:int=100,
        n_smooth:int=1,
        print:bool=True,
        print_freq:int=1,
        plot:bool=False
    ):
        self.grids = grids # aka assemblers
        self.smoothers = smoothers
        self.n_cycles = n_cycles
        self.n_smooth = n_smooth
        self.print = print
        self.print_freq = print_freq
        self.plot = plot

        self.i_cycle = 0

    def solve(self, rhs:np.ndarray):
        nlevels = len(self.grids)
        solns = [np.zeros_like(self.grids[i].force) for i in range(nlevels)]
        defects = [self.grids[i].force.copy() for i in range(nlevels)]
        # overwrite fine grid defect with rhs
        defects[0] = rhs.copy()
        mats = [self.grids[i].kmat.copy() for i in range(nlevels)]
        dpn = self.grids[0].dof_per_node

        init_defect_norm = np.linalg.norm(defects[0])
        if self.print: print(f"V-cycle multigrid solve with {nlevels} grids:")
        if self.print: print(f"\n\t{init_defect_norm=:.2e}\n----------------\n")
        converged = False

        true_soln = sp.sparse.linalg.spsolve(mats[0].copy(), defects[0])
        if self.plot:
            print(f"level 0 true soln")
            debug_plot_diff(dpn, self.grids[0], self.grids[0], vec1=defects[0], vec2=true_soln)

        for i_cycle in range(self.n_cycles):

            # smooth and restrict downwards
            for i in range(0, nlevels - 1):

                # pre-smooth
                pre_defect = defects[i].copy()
                # print(f"level {i=} {solns[i].shape=} {defects[i].shape=}")
                self.smoothers[i].smooth_defect(solns[i], defects[i])

                # # check defect is actual defect
                if i == 0 and self.plot:
                    new_defect = defects[i].copy()
                    act_defect = rhs.copy() - mats[i].dot(solns[i])
                    diff = new_defect - act_defect
                    diff_nrm = np.linalg.norm(diff)
                    defect_nrm = np.linalg.norm(act_defect)
                    print(f"{i_cycle=} {diff_nrm=:.4e} in pre-smooth level {i=} {defect_nrm=:.4e}")
                if self.plot: 
                    debug_plot(dpn, self.grids[i], vec1=pre_defect, vec2=defects[i])

                # restrict
                defects[i+1] = self.grids[i+1].restrict_defect(defects[i].copy())
                solns[i+1] *= 0.0 # resets coarse soln when you restrict defect

                if self.plot:
                    print(f"level {i} to {i+1} restrict")
                    debug_plot_diff(dpn, self.grids[i], self.grids[i+1], vec1=defects[i], vec2=defects[i+1])

            # coarse grid solve
            solns[nlevels-1] = sp.sparse.linalg.spsolve(mats[nlevels-1].copy(), defects[nlevels-1])

            if self.plot:
                print(f"coarse grid {nlevels-1} solve")
                debug_plot_diff(dpn, self.grids[i+1], self.grids[i+1], vec1=defects[nlevels-1], vec2=solns[nlevels-1])

            # prolong and post-smooth
            for i in range(nlevels-2, -1, -1):

                # proposed prolongate
                dx = self.grids[i].prolongate(solns[i+1])
                # print(f"{dx.shape=} {mats[i].shape=}")
                df = mats[i].dot(dx)

                fine_soln = sp.sparse.linalg.spsolve(mats[i].copy(), defects[i])# + solns[i]

                # line search scaling of prolongation (since coarse grid less nodes, one DOF scaling not appropriate on default, 
                # can be off by 2x, 4x or some other constant usually)
                # omega = np.dot(dx, defects[i]) / np.dot(dx, df)
                omega = 1.0

                # debug_plot(2, self.grids[i], vec1=-omega * df, vec2=defects[i])
                solns[i] += omega * dx
                defects[i] -= omega * df
                # if debug_print: print(f"\tprolong line search with {omega=:.2e}")

                if self.plot:
                    print(f"check prolongate exact (fine exact, fine prolong)")
                    debug_plot_diff(dpn, self.grids[i], self.grids[i], vec1=fine_soln, vec2=omega * dx)
                    post_soln = solns[i].copy()



                # check defect is actual defect
                # if i == 0:
                #     new_defect = defects[i].copy()
                #     act_defect = rhs.copy() - mats[i].dot(solns[i])
                #     diff = new_defect - act_defect
                #     diff_nrm = np.linalg.norm(diff)
                #     defect_nrm = np.linalg.norm(act_defect)
                #     print(f"{i_cycle=} {diff_nrm=:.4e} in prolong level {i=} {defect_nrm=:.4e}")

                if self.plot:
                    print(f"level {i+1} to {i} prolongate with {omega=:.2e}")
                    debug_plot_diff(dpn, self.grids[i+1], self.grids[i], vec1=solns[i+1], vec2=omega * dx)
                    print(f"level {i} prolong vs full-soln")
                    debug_plot_diff(dpn, self.grids[i], self.grids[i], vec1=omega * dx, vec2=solns[i])
                    post_soln = solns[i].copy()
                    post_defect = defects[i].copy()

                # post-smooth
                self.smoothers[i].smooth_defect(solns[i], defects[i])
                # debug_plot(dof_per_node, grids[0], vec1=post_init_defect, vec2=defects[i])

                # check defect is actual defect
                # if i == 0:
                #     new_defect = defects[i].copy()
                #     act_defect = rhs.copy() - mats[i].dot(solns[i])
                #     diff = new_defect - act_defect
                #     diff_nrm = np.linalg.norm(diff)
                #     defect_nrm = np.linalg.norm(act_defect)
                #     print(f"{i_cycle=} {diff_nrm=:.4e} in post-smooth level {i=} {defect_nrm=:.4e}")
                if self.plot:
                    print(f"post smooth")
                    debug_plot_diff(dpn, self.grids[i], self.grids[i], vec1=post_defect, vec2=defects[i])

            # check conv
            defect_norm = np.linalg.norm(defects[0])
            if i_cycle % self.print_freq == 0 and self.print: print(f"V[{i_cycle}] : {defect_norm=:.2e}")
            if defect_norm <= 1e-6 * init_defect_norm:
                converged = True
                break

        # store the number of incurred cycles
        self.i_cycle = i_cycle + 1

        converged_str = "converged" if converged else "didn't converge"
        if self.print: print(f"V-cycle multigrid {converged_str} in {self.i_cycle} steps")

        # check the residual on fine grid after this solve
        fine_resid = np.linalg.norm(self.grids[0].force.copy() - mats[0].dot(solns[0]))
        if self.print: print(f"\tcheck : {fine_resid=:.2e}")

        return solns[0]
        