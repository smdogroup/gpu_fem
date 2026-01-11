import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class ILU0Preconditioner:
    def __init__(self, A: sp.csr_matrix, dof_per_node=3):
        """
        Build ILU(0) preconditioner with node-wise random reordering.
        """
        A = A.copy()
        assert sp.isspmatrix_csr(A)
        n_dof = A.shape[0]
        n_nodes = n_dof // dof_per_node

        # Random permutation of nodes
        node_perm = np.random.permutation(n_nodes)
        dof_perm = np.concatenate([
            np.arange(i*dof_per_node, (i+1)*dof_per_node)
            for i in node_perm
        ])

        # temp debug
        # dof_perm = np.arange(n_dof)

        A_perm = A[dof_perm, :][:, dof_perm]
        # ilu = spla.spilu(A_perm, fill_factor=1.0, drop_tol=0.0)
        ilu = spla.spilu(A_perm, fill_factor=10.0, drop_tol=1e-5)

        self.solve = ilu.solve
        self.perm = dof_perm
        self.perm_inv = np.argsort(dof_perm)

    def apply(self, r):
        """
        Apply M⁻¹ to r: z = M⁻¹ r using ILU(0) solve.
        """
        r_perm = r[self.perm]
        z_perm = self.solve(r_perm)
        return z_perm[self.perm_inv]
    
def pcg(A, b, M: ILU0Preconditioner, x0=None, tol=1e-8, maxiter=200):
    """
    PCG with right preconditioning: solve A M⁻¹ y = b, x = M⁻¹ y
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Right-preconditioned system: A M⁻¹ y = b, where x = M⁻¹ y
    r = b - A @ x
    z = M.apply(r)
    p = z.copy()
    rz_old = np.dot(r, z)

    for i in range(maxiter):
        Ap = A @ p
        alpha = rz_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        if np.linalg.norm(r) < tol:
            print(f"PCG converged in {i+1} iterations.")
            return x

        z = M.apply(r)
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    print("PCG did not converge within maxiter.")
    return x

class SchwarzSmoother:
    def __init__(self, K:np.ndarray, nx_sd:int=4, overlap_frac:float=0.2, additive:bool=True):
        self.K = K.copy()
        self.nx_sd = nx_sd # num of subdomains in x-dir
        self.overlap_frac = overlap_frac
        self.additive = additive

        self.ndof = K.shape[0]
        self.nnodes = self.ndof // 3
        self.nx = int(self.nnodes**0.5)
        self.nxe = self.nx - 1
        self._nxe_sd_no = self.nxe / nx_sd # no overlap (keep as float here)
        self.nx_sd_no = self._nxe_sd_no + 1
        self._nxe_sd_wo = self._nxe_sd_no * overlap_frac # with overlap part each side (overlap only though)
        self.nx_sd_wo = self._nxe_sd_wo + 1
        self.N_sd = nx_sd**2
        self.h = 1.0 / self.nxe

        self.E, self.nu, self.thick = None, None, None

    def set_material(self, E, nu, thick):
        self.E, self.nu, self.thick = E, nu, thick

    def _get_1d_domain(self, ix_sd):
        """get the start and stop nodes for 1d subdomain"""
        ix_no = self._nxe_sd_no * np.array([ix_sd, ix_sd + 1])
        ix_no[-1] + 1
        # print(F"{ix_no=}")
        ix_no[0] -= self._nxe_sd_wo
        ix_no[-1] += self._nxe_sd_wo
        ix_no[0] = np.max([0.0, ix_no[0]])
        ix_no[-1] = np.min([self.nx - 1, ix_no[-1]])
        ix_no[0] = np.floor(ix_no[0])
        ix_no[-1] = np.ceil(ix_no[-1])
        return ix_no.astype(np.int32)

    @property
    def sd_nodes_list(self):
        nodes_list = []
        for i_sd in range(self.N_sd):
            ix_sd, iy_sd = i_sd % self.nx_sd, i_sd // self.nx_sd
            ix_bounds = self._get_1d_domain(ix_sd)
            iy_bounds = self._get_1d_domain(iy_sd)

            _sd_nodes = []
            for iy in range(iy_bounds[0], iy_bounds[1] + 1):
                for ix in range(ix_bounds[0], ix_bounds[1] + 1):
                    _sd_nodes += [self.nx * iy + ix]
            nodes_list += [_sd_nodes]
        return nodes_list
    
    def _get_boundary_nodes_on_sd(self, sd_nodes:np.ndarray):
        ix_list, iy_list = sd_nodes % self.nx, sd_nodes // self.nx
        ix_min, ix_max = np.min(ix_list), np.max(ix_list)
        iy_min, iy_max = np.min(iy_list), np.max(iy_list)
        x_bndry_mask = np.logical_or(ix_list == ix_min, ix_list == ix_max)
        y_bndry_mask = np.logical_or(iy_list == iy_min, iy_list == iy_max)
        bndry_mask = np.logical_or(x_bndry_mask, y_bndry_mask)
        return sd_nodes[bndry_mask]
    
    def _node_to_dof_list(self, node_list):
        return np.array([3*inode+idof for inode in node_list for idof in range(3)])
        # return np.array([3*inode+idof for inode in node_list for idof in range(1)]) # here try just w dof constrained on schwarz subdomains
    
    def _get_bndry_sd_dof(self, sd_dof, bndry_dof):
        bndry_sd_dof = []
        for i, inode in enumerate(sd_dof):
            if inode in bndry_dof:
                bndry_sd_dof += [i]
        return np.array(bndry_sd_dof)
    
    def _add_weak_moment_lhs(self, K_sd, b_sd, sd_nodes, c_xvals):
        """add weak moment bc penalty to the lhs and rhs for (normal moments Dirichlet)"""

        from ._dkt_plate_elem import get_weak_moment_kelem_felem

        assert(self.E)

        # get the bndry ix and iy coords
        ix_list, iy_list = sd_nodes % self.nx, sd_nodes // self.nx
        ix_min, ix_max = np.min(ix_list), np.max(ix_list)
        iy_min, iy_max = np.min(iy_list), np.max(iy_list)
        nx_red = ix_max - ix_min + 1
        # print(f"{nx_red=}")

        irt3 = 1.0 / np.sqrt(3)
        pt1, pt2 = 0.5 * (1 - irt3), 0.5 * (1 + irt3)

        for bndry_ind in range(4):
            # print(f"{bndry_ind=} \n-----------------\n")

            # quadpt weights are just 1, 1 each time with two-point quadrature

            if bndry_ind == 0: # xleft bndry, ix == ix_min
                ix_list, iy_list = [ix_min], [_ for _ in range(iy_min, iy_max)]
                xi_list, eta_list = [0.0], [pt1, pt2]
                x_elem = self.h * np.array([0.0, 1.0, 0.0])
                y_elem = self.h * np.array([0.0, 0.0, 1.0])

            elif bndry_ind == 1: # xright bndry, ix == ix_max
                ix_list, iy_list = [ix_max], [_ for _ in range(iy_min, iy_max)]
                xi_list, eta_list = [pt1, pt2], [0.0]
                x_elem = self.h * np.array([1.0, 1.0, 0.0])
                y_elem = self.h * np.array([0.0, 1.0, 1.0])

            elif bndry_ind == 2: # ybot bndry, iy == iy_min
                ix_list, iy_list = [_ for _ in range(ix_min, ix_max)], [iy_min]
                xi_list, eta_list = [pt1, pt2], [0.0]
                x_elem = self.h * np.array([0.0, 1.0, 0.0])
                y_elem = self.h * np.array([0.0, 0.0, 1.0])

            else: # ytop bndry, iy == iy_max
                ix_list, iy_list = [_ for _ in range(ix_min, ix_max)], [iy_min]
                xi_list, eta_list = [0.0], [pt1, pt2]
                x_elem = self.h * np.array([1.0, 1.0, 0.0])
                y_elem = self.h * np.array([0.0, 1.0, 1.0])

            # print(F"{ix_min=} {ix_max=} {iy_min=} {iy_max=}")
            # import matplotlib.pyplot as plt

            # main part now
            for ix in ix_list:
                for iy in iy_list:
                    for xi in xi_list:
                        for eta in eta_list:
                            
                            # print(f"{ix=} {iy=} {xi=} {eta=}")
                            inode = nx_red * (iy - iy_min) + (ix-ix_min)
                            inode_glob = self.nx * iy + ix

                            elem_nodes = [inode_glob, inode_glob+1, inode_glob+self.nx] if bndry_ind in [0, 2] else [inode_glob, inode_glob+self.nx, inode_glob + self.nx - 1]
                            elem_dof = np.array([3*_inode+_idof for _inode in elem_nodes for _idof in range(3)])
                            elem_vals = c_xvals[elem_dof]
                            Mn_kelem_term, Mn_felem_term = get_weak_moment_kelem_felem(elem_vals, self.E, self.thick, self.nu, x_elem, y_elem, xi=xi, eta=eta)
                            
                            elem_nodes_red = [inode, inode+1, inode+nx_red] if bndry_ind in [0, 2] else [inode, inode + nx_red, inode+nx_red-1]
                            elem_red_dof = np.array([3*_inode+_idof for _inode in elem_nodes_red for _idof in range(3)])
                            # it's linear thn on the bndry so only need one quadpt (or actually maybe that's not true?)
                            arr_ind = np.ix_(elem_red_dof, elem_red_dof)
                            # print(f"{inode=} {elem_nodes_red=} {elem_red_dof=} {K_sd.shape=}")
                            K_sd[arr_ind] += Mn_kelem_term
                            b_sd[elem_red_dof] += Mn_felem_term

        # plt.imshow(K_sd)
        # plt.show()
        # exit()

        return K_sd, b_sd      
    

    def smooth(self, b0, x0, omega:float=1.0):
        """do additive or multiplicative schwarz smoothing"""
        b = b0.copy()
        x0 = x0.copy()
        x = x0.copy()

        if not self.additive and not(omega == 1.0):
            print(f"warning changed {omega=:.2e} back to 1.0 for multiplicative schwarz") 
            omega = 1.0 # change back to full update if multiplicative schwarz

        nodes_list = self.sd_nodes_list
        for i_sd in range(self.N_sd):
            sd_nodes = np.array(nodes_list[i_sd])
            bndry_nodes = self._get_boundary_nodes_on_sd(sd_nodes)
            sd_dof, bndry_dof = self._node_to_dof_list(sd_nodes), self._node_to_dof_list(bndry_nodes)
            bndry_dof = bndry_dof[0::3] # only w dof displaced

            # now compute smaller problem
            bndry_sd_dof = self._get_bndry_sd_dof(sd_dof, bndry_dof)
            int_dof = np.array([idof for idof in sd_dof if not(idof in bndry_dof)])
            int_sd_dof = self._get_bndry_sd_dof(sd_dof, int_dof)
            K_sd = self.K[sd_dof, :][:, sd_dof]
            b_sd = b0[sd_dof] if self.additive else b[sd_dof]
            c_x = x0.copy() if self.additive else x.copy()

            # add weak moment term to lhs and rhs
            K_sd, b_sd = self._add_weak_moment_lhs(K_sd, b_sd, sd_nodes, c_x)
     
            # now compute reduced problem (only interior dof)
            K_sd_red = K_sd[int_sd_dof,:][:,int_sd_dof]
            b_sd_red = b_sd[int_sd_dof]
            K_FC = K_sd[int_sd_dof, :][:, bndry_sd_dof]
            b_sd_red -= np.dot(K_FC, c_x[bndry_dof]) # nz dirichlet bc adjustment

            # solve the reduced problem
            dx_sd = np.linalg.solve(K_sd_red, b_sd_red) * omega # damp the update here..
            x[int_dof] += dx_sd

            # now adjust the rhs
            dx_sd_full = np.zeros(sd_dof.shape[0])
            dx_sd_full[int_sd_dof] = dx_sd[:]
            # need to include nodal force updates on subdomain bndry also as interior nodes near it do affect these points
            b[sd_dof] -= np.dot(K_sd, dx_sd_full) 
    
        return x, b
    
    def multi_smooth(self, b0, x0, num_smooth:int=2, omega:float=1.0):
        """do multiple schwarz smoothing steps"""

        # b = b0.copy()
        # x = x0.copy()
        # for ismooth in range(num_smooth):
        #     x, b = self.smooth(b, x0=x, omega=omega)
        # return x, b
    
    # TODO : need zero disps each time like below?
        b = b0.copy()
        x, x0 = x0.copy(), x0.copy()
        for ismooth in range(num_smooth):
            _du, b = self.smooth(b, x0=np.zeros(self.ndof), omega=omega)
            # x += _du
            x -= _du # TODO : plus or minus here?
        return x, b

            

def gauss_seidel_csr(A, b, x0, num_iter=1):
    """
    Perform Gauss-Seidel smoothing for Ax = b
    A: csr_matrix (assumed square)
    b: RHS vector
    x0: initial guess
    num_iter: number of smoothing iterations
    Returns: updated solution x
    """
    x = x0.copy()
    n = A.shape[0]
    for _ in range(num_iter):
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            Ai = A.indices[row_start:row_end]
            Av = A.data[row_start:row_end]

            sum_ = 0.0
            diag = 0.0
            for idx, j in enumerate(Ai):
                if j == i:
                    diag = Av[idx]
                else:
                    sum_ += Av[idx] * x[j]
            x[i] = (b[i] - sum_) / diag
    return x

def block_gauss_seidel(A, b: np.ndarray, x0: np.ndarray, num_iter=1, ndof:int=3):
    """
    Perform Block Gauss-Seidel smoothing for 3 DOF per node.
    A: csr_matrix of size (3*nnodes, 3*nnodes)
    b: RHS vector (3*nnodes,)
    x0: initial guess (3*nnodes,)
    num_iter: number of smoothing iterations
    Returns updated solution vector x
    """
    x = x0.copy()
    n = A.shape[0] // ndof

    for it in range(num_iter):
        for i in range(n):
            row_block_start = i * ndof
            row_block_end = (i + 1) * ndof

            # Initialize block and RHS
            Aii = np.zeros((ndof, ndof))
            rhs = b[row_block_start:row_block_end].copy()

            for row_local, row in enumerate(range(row_block_start, row_block_end)):
                for idx in range(A.indptr[row], A.indptr[row + 1]):
                    col = A.indices[idx]
                    val = A.data[idx]

                    j = col // ndof
                    dof_j = col % ndof

                    col_block_start = j * ndof
                    col_block_end = (j + 1) * ndof

                    if j == i:
                        Aii[row_local, dof_j] = val  # Fill local diag block
                    else:
                        rhs[row_local] -= val * x[col]

            # Check for singular or ill-conditioned diagonal block
            try:
                x[row_block_start:row_block_end] = np.linalg.solve(Aii, rhs)
            except np.linalg.LinAlgError:
                print(f"Warning: singular block at node {i}, skipping update.")
                continue

    return x