import numpy as np
from .basis import second_order_quadrature, zero_order_quadrature, first_order_quadrature
from .basis import interp_lagrange, interp_lagrange_transpose
from .basis import get_lagrange_basis_2d_all

import scipy.sparse as sp

class ReissnerMindlinPlateElement_OptProlong:
    # reissner-mindlin C0-continuous element

    def __init__(
        self, 
        reduced_integrated:bool=False, 
        prolong_mode:str='locking-global', # 'standard'
        lam:float=1e-2,
    ):
        
        assert prolong_mode in ['locking-global', 'locking-local', 'standard']
                    
        self.dof_per_node = 3
        self.nodes_per_elem = 4
        self.ndof = self.dof_per_node * self.nodes_per_elem
        self.reduced_integrated = reduced_integrated
        self.prolong_mode = prolong_mode
        self.clamped = False
        self.lam = lam

        # cache for prolong/restrict operators
        self._P1_cache = {}   # key: nxe_coarse -> P1 (csr)
        self._P2_cache = {}   # key: nxe_coarse -> P2 (csr)
        self._P2_u3_cache = {}
        self._lock_P_cache = {}

    def get_kelem(self, E: float, nu: float, thick: float, elem_xpts: np.ndarray):
        """
        elem_xpts expected length 12: [x0,y0,z0?, x1,y1,z1?, x2,y2,z2?, x3,y3,z3?]
        You currently use x=elem_xpts[0::3], y=elem_xpts[1::3].
        Assumes an axis-aligned rectangular mapping (x_eta ~ 0, y_xi ~ 0) as in your code.
        """
        pts, wts = second_order_quadrature()
        if self.reduced_integrated:
            shear_pts, shear_wts = zero_order_quadrature()
            # shear_pts, shear_wts = first_order_quadrature()
        else:
            shear_pts, shear_wts = second_order_quadrature()

        kelem = np.zeros((self.ndof, self.ndof))

        # material constants
        D0 = E * thick**3 / (12.0 * (1.0 - nu**2))
        Db = D0 * np.array([
            [1.0,  nu,          0.0],
            [nu,   1.0,         0.0],
            [0.0,  0.0, (1.0-nu)/2.0],
        ])
        ks = 5.0 / 6.0
        G = E / (2.0 * (1.0 + nu))
        Ds = (ks * G * thick) * np.eye(2)

        x = elem_xpts[0::3]
        y = elem_xpts[1::3]

        # ---- BENDING: depends on rotation gradients only ----
        for ii, xi in enumerate(pts):
            for jj, eta in enumerate(pts):
                wt = wts[ii] * wts[jj]

                N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)

                # geometry derivatives
                x_xi  = np.dot(Nxi,  x); x_eta = np.dot(Neta, x)
                y_xi  = np.dot(Nxi,  y); y_eta = np.dot(Neta, y)

                # if you're truly axis-aligned rectangles, use diagonal map
                # but keep the general 2x2 inverse in case x_eta,y_xi aren't exactly zero
                J = x_xi * y_eta - x_eta * y_xi
                invJ = 1.0 / J
                xi_x  =  y_eta * invJ
                xi_y  = -x_eta * invJ
                eta_x = -y_xi  * invJ
                eta_y =  x_xi  * invJ

                # physical grads of shape functions
                Nx = Nxi * xi_x + Neta * eta_x
                Ny = Nxi * xi_y + Neta * eta_y

                # curvature vector for RM:
                # kxx = d(thx)/dx
                # kyy = d(thy)/dy
                # kxy = d(thx)/dy + d(thy)/dx   (engineering curvature)
                Bb = np.zeros((3, self.ndof))

                # thx dofs are 1::3, thy dofs are 2::3
                Bb[0, 1::3] = Nx          # d(thx)/dx
                Bb[1, 2::3] = Ny          # d(thy)/dy
                Bb[2, 1::3] = Ny          # d(thx)/dy
                Bb[2, 2::3] = Nx          # d(thy)/dx

                kelem += (Bb.T @ Db @ Bb) * (wt * J)

        # ---- SHEAR: w_x - thx, w_y - thy (or plus; choose one and stick to it) ----
        for ii, xi in enumerate(shear_pts):
            for jj, eta in enumerate(shear_pts):
                wt = shear_wts[ii] * shear_wts[jj]

                N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)

                x_xi  = np.dot(Nxi,  x); x_eta = np.dot(Neta, x)
                y_xi  = np.dot(Nxi,  y); y_eta = np.dot(Neta, y)

                J = x_xi * y_eta - x_eta * y_xi
                invJ = 1.0 / J
                xi_x  =  y_eta * invJ
                xi_y  = -x_eta * invJ
                eta_x = -y_xi  * invJ
                eta_y =  x_xi  * invJ

                Nx = Nxi * xi_x + Neta * eta_x
                Ny = Nxi * xi_y + Neta * eta_y

                Bs = np.zeros((2, self.ndof))

                # gamma_xz = w_x + thx
                Bs[0, 0::3] = Nx
                Bs[0, 1::3] = N

                # gamma_yz = w_y + thy
                Bs[1, 0::3] = Ny
                Bs[1, 2::3] = N

                kelem += (Bs.T @ Ds @ Bs) * (wt * J)

        return kelem

    def get_felem(self, mag, elem_xpts:np.ndarray):
        """get element load vector"""

        pts, wts = second_order_quadrature()
        felem = np.zeros(self.ndof)
        x = elem_xpts[0::3]
        y = elem_xpts[1::3]

        for ipt in range(9):
            ii, jj = ipt % 3, ipt // 3
            xi = pts[ii]; eta = pts[jj]
            wt = wts[ii] * wts[jj]
            # basis (need N to map load; Nxi/Neta to get geometry jacobian)
            N, Nxi, Neta= get_lagrange_basis_2d_all(
                xi, eta, 
            )

            # geometry jacobian determinant
            x_xi  = np.dot(Nxi,  x);  x_eta = np.dot(Neta, x)
            y_xi  = np.dot(Nxi,  y);  y_eta = np.dot(Neta, y)
            J = x_xi * y_eta - x_eta * y_xi

            # physical point (x,y) at this quadrature point
            xq = float(np.dot(N, x))
            yq = float(np.dot(N, y))

            q = float(mag(xq, yq))   # distributed transverse load
            # q *= 60.0 # not sure where this correction is coming from tbh

            # consistent nodal load contribution: ∫ N^T q dA = Σ N_i q * wt * J
            fN = q * wt * J * N  # length 9

            # Apply load to the DOFs that contribute to transverse displacement.
            felem[0::3] += fN  # w

        return felem
    
    # ----------------------------
    # Prolongation / Restriction
    # ----------------------------
    def _build_P1_scalar(self, nxe_coarse: int) -> sp.csr_matrix:
        """
        1D scalar nodal prolongation from nxe_coarse -> 2*nxe_coarse.
        Nodes: nc = nxe_coarse+1, nf = 2*nxe_coarse+1 = 2*nc-1.
        """
        if nxe_coarse in self._P1_cache:
            return self._P1_cache[nxe_coarse]

        nc = nxe_coarse + 1
        nf = 2 * nxe_coarse + 1  # = 2*nc - 1

        # COO assembly
        nnz = 2 * nf  # loose upper bound
        rows = []
        cols = []
        vals = []

        # even fine nodes copy coarse
        for i in range(nc):
            rows.append(2 * i)
            cols.append(i)
            vals.append(1.0)

        # odd fine nodes are averages
        for i in range(nc - 1):
            r = 2 * i + 1
            rows += [r, r]
            cols += [i, i + 1]
            vals += [0.5, 0.5]

        P1 = sp.coo_matrix((vals, (rows, cols)), shape=(nf, nc)).tocsr()
        self._P1_cache[nxe_coarse] = P1
        return P1

    def _build_P2_scalar(self, nxe_coarse: int) -> sp.csr_matrix:
        """
        2D scalar prolongation on square grid: P2 = kron(P1, P1)
        using lexicographic node numbering inode = i + nx*j.
        """
        if nxe_coarse in self._P2_cache:
            return self._P2_cache[nxe_coarse]

        P1 = self._build_P1_scalar(nxe_coarse)
        P2 = sp.kron(P1, P1, format="csr")
        self._P2_cache[nxe_coarse] = P2
        return P2
    
    def _build_P2_uncoupled3(self, nxe_coarse: int) -> sp.csr_matrix:
        """
        2D uncoupled prolongation for 3 DOF per node, node-wise ordering:
        [w0, thx0, thy0, w1, thx1, thy1, ...]
        Constructed as kron(P2_scalar, I3) so each nodal DOF interpolates the same way.
        Shape: (3*nnodes_f, 3*nnodes_c)
        """
        if not hasattr(self, "_P2_u3_cache"):
            self._P2_u3_cache = {}

        if nxe_coarse in self._P2_u3_cache:
            return self._P2_u3_cache[nxe_coarse]

        P2s = self._build_P2_scalar(nxe_coarse)  # (nnodes_f, nnodes_c)
        I3 = sp.eye(3, format="csr")

        # For node-wise interleaved DOFs, this is the correct Kronecker order:
        # kron(P2s, I3) maps [node_c ⊗ dof] -> [node_f ⊗ dof]
        P = sp.kron(P2s, I3, format="csr")

        self._P2_u3_cache[nxe_coarse] = P
        return P
    
    def _apply_bcs_to_P(self, P: sp.csr_matrix, nxe_c: int) -> sp.csr_matrix:
        """
        Enforce Dirichlet BC structure directly on P (fine rows, coarse cols).
        For simply-supported: constrain w on boundary nodes.
        For clamped: constrain w, thx, thy on boundary nodes.
        """
        nxe_f = 2 * nxe_c
        nx_f = nxe_f + 1
        nx_c = nxe_c + 1

        nnodes_f = nx_f * nx_f
        nnodes_c = nx_c * nx_c

        # which dofs are constrained at a boundary node
        if self.clamped:
            dofs = (0, 1, 2)   # w, thx, thy
        else:
            dofs = (0,)        # w only

        fixed_rows_f = []
        for inode in range(nnodes_f):
            i = inode % nx_f
            j = inode // nx_f
            on_edge = (i == 0) or (i == nx_f - 1) or (j == 0) or (j == nx_f - 1)
            if on_edge:
                base = 3 * inode
                for a in dofs:
                    fixed_rows_f.append(base + a)

        fixed_cols_c = []
        for inode in range(nnodes_c):
            i = inode % nx_c
            j = inode // nx_c
            on_edge = (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1)
            if on_edge:
                base = 3 * inode
                for a in dofs:
                    fixed_cols_c.append(base + a)

        fixed_rows_f = np.array(sorted(set(fixed_rows_f)), dtype=int)
        fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=int)

        # IMPORTANT: for Dirichlet dofs, we want the prolongation to output exactly 0,
        # independent of coarse values. So:
        #  - zero those fine rows
        #  - zero those coarse columns (optional but recommended for consistency)
        P = P.tolil()
        P[fixed_rows_f, :] = 0.0
        P[:, fixed_cols_c] = 0.0
        P = P.tocsr()
        P.eliminate_zeros()
        return P

    
    def _locking_aware_prolong_global_v1(self, nxe_c: int, length: float = 1.0):

        if nxe_c in self._lock_P_cache:
            return self._lock_P_cache[nxe_c]

        assert self.reduced_integrated

        shear_pts, _ = zero_order_quadrature()

        nxe_f = 2 * nxe_c
        nx_f  = nxe_f + 1
        nnodes_f = nx_f**2
        nelems_f = nxe_f**2
        N_f = 3 * nnodes_f

        nx_c  = nxe_c + 1
        nnodes_c = nx_c**2
        nelems_c = nxe_c**2
        N_c = 3 * nnodes_c

        # ----------------------------
        # Build G_f and G_c (dense)
        # NOTE: you fixed x_f/y_f ordering already 👍
        # ----------------------------
        G_f = np.zeros((2 * nelems_f, N_f))
        dx_f = length / nxe_f
        x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
        y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

        for ielem_f in range(nelems_f):
            ex = ielem_f % nxe_f
            ey = ielem_f // nxe_f
            loc_nodes = np.array([
                ex + nx_f * ey,
                (ex + 1) + nx_f * ey,
                (ex + 1) + nx_f * (ey + 1),
                ex + nx_f * (ey + 1)
            ])
            loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)])

            xi = shear_pts[0]; eta = shear_pts[0]
            N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)

            x_xi  = np.dot(Nxi,  x_f); x_eta = np.dot(Neta, x_f)
            y_xi  = np.dot(Nxi,  y_f); y_eta = np.dot(Neta, y_f)

            J = x_xi * y_eta - x_eta * y_xi
            invJ = 1.0 / J
            xi_x  =  y_eta * invJ
            xi_y  = -x_eta * invJ
            eta_x = -y_xi  * invJ
            eta_y =  x_xi  * invJ

            Nx = Nxi * xi_x + Neta * eta_x
            Ny = Nxi * xi_y + Neta * eta_y

            Bs = np.zeros((2, 12))
            Bs[0, 0::3] = Nx
            Bs[0, 1::3] = N
            Bs[1, 0::3] = Ny
            Bs[1, 2::3] = N

            G_f[2 * ielem_f: 2 * ielem_f + 2, loc_dof] += Bs

        G_c = np.zeros((2 * nelems_c, N_c))
        dx_c = length / nxe_c
        x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
        y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

        for ielem_c in range(nelems_c):
            ex = ielem_c % nxe_c
            ey = ielem_c // nxe_c
            loc_nodes = np.array([
                ex + nx_c * ey,
                (ex + 1) + nx_c * ey,
                (ex + 1) + nx_c * (ey + 1),
                ex + nx_c * (ey + 1)
            ])
            loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)])

            xi = shear_pts[0]; eta = shear_pts[0]
            N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)

            x_xi  = np.dot(Nxi,  x_c); x_eta = np.dot(Neta, x_c)
            y_xi  = np.dot(Nxi,  y_c); y_eta = np.dot(Neta, y_c)

            J = x_xi * y_eta - x_eta * y_xi
            invJ = 1.0 / J
            xi_x  =  y_eta * invJ
            xi_y  = -x_eta * invJ
            eta_x = -y_xi  * invJ
            eta_y =  x_xi  * invJ

            Nx = Nxi * xi_x + Neta * eta_x
            Ny = Nxi * xi_y + Neta * eta_y

            Bs = np.zeros((2, 12))
            Bs[0, 0::3] = Nx
            Bs[0, 1::3] = N
            Bs[1, 0::3] = Ny
            Bs[1, 2::3] = N

            G_c[2 * ielem_c: 2 * ielem_c + 2, loc_dof] += Bs

        # ----------------------------
        # P_gam: elementwise injection (your choice)
        # ----------------------------
        P_gam = np.zeros((2 * nelems_f, 2 * nelems_c))
        for ielem_f in range(nelems_f):
            ex = ielem_f % nxe_f; ey = ielem_f // nxe_f
            ex_c = ex // 2; ey_c = ey // 2
            ielem_c = ex_c + ey_c * nxe_c
            P_gam[2 * ielem_f,     2 * ielem_c]     = 1.0
            P_gam[2 * ielem_f + 1, 2 * ielem_c + 1] = 1.0

        RHS = P_gam @ G_c  # (2*nelems_f, 3*nnodes_c)

        # ----------------------------
        # Build P0 and ENFORCE BCs ON P0 ITSELF
        # ----------------------------
        P_0 = self._build_P2_uncoupled3(nxe_c)   # csr
        P_0 = self._apply_bcs_to_P(P_0, nxe_c)   # csr, now BC structure is in the matrix

        lam = float(self.lam)

        # ----------------------------
        # Now define free fine rows / free coarse cols consistent with BC structure
        # (Since BC rows/cols are already zero in P0, we keep them fixed and solve only interior)
        # ----------------------------
        all_rows_f = np.arange(3 * nnodes_f, dtype=int)
        all_cols_c = np.arange(3 * nnodes_c, dtype=int)

        # reuse the SAME logic as _apply_bcs_to_P to generate fixed sets
        # (I’m writing it inline to keep this block self-contained.)
        if self.clamped:
            dofs = (0, 1, 2)
        else:
            dofs = (0,)

        fixed_rows_f = []
        for inode in range(nnodes_f):
            i = inode % nx_f; j = inode // nx_f
            if (i == 0) or (i == nx_f - 1) or (j == 0) or (j == nx_f - 1):
                base = 3 * inode
                for a in dofs:
                    fixed_rows_f.append(base + a)

        fixed_cols_c = []
        for inode in range(nnodes_c):
            i = inode % nx_c; j = inode // nx_c
            if (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1):
                base = 3 * inode
                for a in dofs:
                    fixed_cols_c.append(base + a)

        fixed_rows_f = np.array(sorted(set(fixed_rows_f)), dtype=int)
        fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=int)

        free_rows_f = np.setdiff1d(all_rows_f, fixed_rows_f, assume_unique=False)
        free_cols_c = np.setdiff1d(all_cols_c, fixed_cols_c, assume_unique=False)

        # ----------------------------
        # Solve for P_free (interior rows/cols only)
        # ----------------------------
        A = G_f[:, free_rows_f]          # (2*nelems_f, nfree_rows_f)
        RHS_free = RHS[:, free_cols_c]   # (2*nelems_f, nfree_cols_c)

        idx = np.ix_(free_rows_f, free_cols_c)
        P0_free = P_0[idx].toarray()     # dense (nfree_rows_f, nfree_cols_c)

        M = A.T @ A + lam * np.eye(free_rows_f.size)
        rhs = A.T @ RHS_free + lam * P0_free
        P_free = np.linalg.solve(M, rhs)

        # ----------------------------
        # Assemble: start from P0 (with BC structure), overwrite only the free/free block
        # ----------------------------
        P = P_0.toarray()
        P[idx] = P_free

        self._lock_P_cache[nxe_c] = P.copy()
        return P

    def _locking_aware_prolong_global_v2(self, nxe_c: int, length: float = 1.0):

        if nxe_c in self._lock_P_cache:
            return self._lock_P_cache[nxe_c]

        assert self.reduced_integrated

        shear_pts, _ = zero_order_quadrature()

        nxe_f = 2 * nxe_c
        nx_f  = nxe_f + 1
        nnodes_f = nx_f**2
        nelems_f = nxe_f**2
        N_f = 3 * nnodes_f

        nx_c  = nxe_c + 1
        nnodes_c = nx_c**2
        nelems_c = nxe_c**2
        N_c = 3 * nnodes_c

        # ----------------------------
        # Build G_f and G_c (dense)  (unchanged)
        # ----------------------------
        G_f = np.zeros((2 * nelems_f, N_f))
        dx_f = length / nxe_f
        x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
        y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

        for ielem_f in range(nelems_f):
            ex = ielem_f % nxe_f
            ey = ielem_f // nxe_f
            loc_nodes = np.array([
                ex + nx_f * ey,
                (ex + 1) + nx_f * ey,
                (ex + 1) + nx_f * (ey + 1),
                ex + nx_f * (ey + 1)
            ])
            loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)])

            xi = shear_pts[0]; eta = shear_pts[0]
            N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)

            x_xi  = np.dot(Nxi,  x_f); x_eta = np.dot(Neta, x_f)
            y_xi  = np.dot(Nxi,  y_f); y_eta = np.dot(Neta, y_f)

            J = x_xi * y_eta - x_eta * y_xi
            invJ = 1.0 / J
            xi_x  =  y_eta * invJ
            xi_y  = -x_eta * invJ
            eta_x = -y_xi  * invJ
            eta_y =  x_xi  * invJ

            Nx = Nxi * xi_x + Neta * eta_x
            Ny = Nxi * xi_y + Neta * eta_y

            Bs = np.zeros((2, 12))
            Bs[0, 0::3] = Nx
            Bs[0, 1::3] = N
            Bs[1, 0::3] = Ny
            Bs[1, 2::3] = N

            G_f[2 * ielem_f: 2 * ielem_f + 2, loc_dof] += Bs

        G_c = np.zeros((2 * nelems_c, N_c))
        dx_c = length / nxe_c
        x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
        y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

        for ielem_c in range(nelems_c):
            ex = ielem_c % nxe_c
            ey = ielem_c // nxe_c
            loc_nodes = np.array([
                ex + nx_c * ey,
                (ex + 1) + nx_c * ey,
                (ex + 1) + nx_c * (ey + 1),
                ex + nx_c * (ey + 1)
            ])
            loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)])

            xi = shear_pts[0]; eta = shear_pts[0]
            N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)

            x_xi  = np.dot(Nxi,  x_c); x_eta = np.dot(Neta, x_c)
            y_xi  = np.dot(Nxi,  y_c); y_eta = np.dot(Neta, y_c)

            J = x_xi * y_eta - x_eta * y_xi
            invJ = 1.0 / J
            xi_x  =  y_eta * invJ
            xi_y  = -x_eta * invJ
            eta_x = -y_xi  * invJ
            eta_y =  x_xi  * invJ

            Nx = Nxi * xi_x + Neta * eta_x
            Ny = Nxi * xi_y + Neta * eta_y

            Bs = np.zeros((2, 12))
            Bs[0, 0::3] = Nx
            Bs[0, 1::3] = N
            Bs[1, 0::3] = Ny
            Bs[1, 2::3] = N

            G_c[2 * ielem_c: 2 * ielem_c + 2, loc_dof] += Bs

        # ----------------------------
        # P_gam: elementwise injection (unchanged)
        # ----------------------------
        P_gam = np.zeros((2 * nelems_f, 2 * nelems_c))
        for ielem_f in range(nelems_f):
            ex = ielem_f % nxe_f; ey = ielem_f // nxe_f
            ex_c = ex // 2; ey_c = ey // 2
            ielem_c = ex_c + ey_c * nxe_c
            P_gam[2 * ielem_f,     2 * ielem_c]     = 1.0
            P_gam[2 * ielem_f + 1, 2 * ielem_c + 1] = 1.0

        RHS = P_gam @ G_c   # (2*nelems_f, 3*nnodes_c)

        # ----------------------------
        # Baseline P0 (DO NOT zero rows/cols here; beam-style uses E for fine BCs)
        # ----------------------------
        P_0 = self._build_P2_uncoupled3(nxe_c)   # csr
        lam = float(self.lam)

        # ----------------------------
        # Coarse BC columns (eliminate like you already do)
        # ----------------------------
        if self.clamped:
            constrained_dofs = (0, 1, 2)
        else:
            constrained_dofs = (0,)  # w only

        fixed_cols_c = []
        for inode in range(nnodes_c):
            i = inode % nx_c; j = inode // nx_c
            on_edge = (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1)
            if on_edge:
                base = 3 * inode
                for a in constrained_dofs:
                    fixed_cols_c.append(base + a)

        fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=int)
        all_cols_c = np.arange(3 * nnodes_c, dtype=int)
        free_cols_c = np.setdiff1d(all_cols_c, fixed_cols_c, assume_unique=False)

        # ----------------------------
        # Fine BC rows: BEAM-STYLE -> keep them in the solve, add E P = 0
        # ----------------------------
        fixed_rows_f = []
        for inode in range(nnodes_f):
            i = inode % nx_f; j = inode // nx_f
            on_edge = (i == 0) or (i == nx_f - 1) or (j == 0) or (j == nx_f - 1)
            if on_edge:
                base = 3 * inode
                for a in constrained_dofs:
                    fixed_rows_f.append(base + a)

        fixed_rows_f = np.array(sorted(set(fixed_rows_f)), dtype=int)

        # Solve rows = interior + boundary constrained rows (so shear near boundary has the needed vars)
        all_rows_f = np.arange(3 * nnodes_f, dtype=int)

        # IMPORTANT:
        # - If SS: only boundary w are constrained; thx/thy are free and remain in all_rows_f anyway.
        # - We keep ALL rows as unknowns (like beam), but if you want to reduce size,
        #   you can set solve_rows_f = union(interior_rows, fixed_rows_f) instead of all_rows_f.
        solve_rows_f = all_rows_f

        # Build E selecting the constrained fine rows
        # E has one row per constrained DOF: E[k, fixed_rows_f[k]] = 1
        nE = fixed_rows_f.size
        E = np.zeros((nE, solve_rows_f.size), dtype=float)

        # map global row index -> local index in solve_rows_f
        # (here solve_rows_f is all_rows_f, so it's identity; keep it general anyway)
        pos = -np.ones(3 * nnodes_f, dtype=int)
        pos[solve_rows_f] = np.arange(solve_rows_f.size, dtype=int)
        for rE, gdof in enumerate(fixed_rows_f):
            E[rE, pos[gdof]] = 1.0

        # ----------------------------
        # Form beam-style augmented system:
        #   A = [G_f; E]
        #   B = [RHS; 0]
        # and solve only for free coarse columns
        # ----------------------------
        A = G_f[:, solve_rows_f]                 # (2*nelems_f, nsolve_rows_f)
        B = RHS[:, free_cols_c]                  # (2*nelems_f, nfree_cols_c)

        A_aug = np.vstack([A, E])                # (2*nelems_f + nE, nsolve_rows_f)
        B_aug = np.vstack([B, np.zeros((nE, B.shape[1]))])

        # P0 block corresponding to solve_rows_f x free_cols_c
        idx0 = np.ix_(solve_rows_f, free_cols_c)
        P0_free = P_0[idx0].toarray()            # (nsolve_rows_f, nfree_cols_c)

        M = A_aug.T @ A_aug + lam * np.eye(solve_rows_f.size)
        rhs = A_aug.T @ B_aug + lam * P0_free
        P_free = np.linalg.solve(M, rhs)         # (nsolve_rows_f, nfree_cols_c)

        # ----------------------------
        # Assemble full P:
        # - Start with P0 (keeps interpolation sane)
        # - overwrite free coarse cols with optimized columns
        # - enforce coarse fixed cols = 0 exactly (consistent with your earlier elimination)
        # ----------------------------
        P = P_0.toarray()

        P[:, fixed_cols_c] = 0.0                 # exact coarse BC columns
        P[np.ix_(solve_rows_f, free_cols_c)] = P_free

        # Optional sanity: hard enforce fine BC rows exactly (should already hold via E)
        P[fixed_rows_f, :] = 0.0

        self._lock_P_cache[nxe_c] = P.copy()
        return P


    def prolongate(self, coarse_u: np.ndarray, nxe_coarse: int):
        """
        coarse_u: size = dpn * (nxc^2), where nxc = nxe_coarse+1
        returns fine_u: size = dpn * (nxf^2), where nxf = 2*nxe_coarse+1
        """
        dpn = self.dof_per_node
        nxc = nxe_coarse + 1
        Nc = nxc * nxc
        assert coarse_u.size == dpn * Nc

        nxe_f = 2 * nxe_coarse
        nxf = nxe_f + 1
        Nf = nxf * nxf

        if self.prolong_mode == 'locking-global':
            # P2 = self._locking_aware_prolong_global_v1(nxe_coarse, length=1.0)
            P2 = self._locking_aware_prolong_global_v2(nxe_coarse, length=1.0)
        elif self.prolong_mode == 'standard':
            P2 = self._build_P2_scalar(nxe_coarse)

        fine_u = np.zeros(dpn * Nf, dtype=float)

        # apply prolongation per dof-slice
        if self.prolong_mode == 'standard':
            # uncoupled prolong each DOF
            for a in range(dpn):
                fine_u[a::dpn] = P2 @ coarse_u[a::dpn]
        else:
            # coupled prolong between DOF per node
            fine_u = P2 @ coarse_u

        # enforce BCs the same way you currently do
        # self.apply_bcs_2d(fine_u, nxe_f)
        return fine_u

    def restrict_defect(self, fine_r: np.ndarray, nxe_fine: int):
        """
        Restriction as transpose of prolongation: R2 = P2.T.
        nxe_fine must be even; nxe_coarse = nxe_fine//2.
        """
        dpn = self.dof_per_node
        nxf = nxe_fine + 1
        Nf = nxf * nxf
        assert fine_r.size == dpn * Nf

        assert (nxe_fine % 2) == 0, "nxe_fine must be even (fine = 2*coarse)."
        nxe_coarse = nxe_fine // 2
        nxc = nxe_coarse + 1
        Nc = nxc * nxc

        if self.prolong_mode == 'locking-global':
            # P2 = self._locking_aware_prolong_global_v1(nxe_coarse, length=1.0)
            P2 = self._locking_aware_prolong_global_v2(nxe_coarse, length=1.0)
        elif self.prolong_mode == 'standard':
            P2 = self._build_P2_scalar(nxe_coarse)

        R2 = P2.T  # per your requirement

        fine_r = fine_r.copy()
        self.apply_bcs_2d(fine_r, nxe_fine)

        coarse_r = np.zeros(dpn * Nc, dtype=float)

        if self.prolong_mode == 'standard':
            # uncoupled prolong each DOF
            for a in range(dpn):
                coarse_r[a::dpn] = R2 @ fine_r[a::dpn]
        else:
            # coupled prolong between DOF per node
            coarse_r = R2 @ fine_r
        

        # self.apply_bcs_2d(coarse_r, nxe_coarse)
        return coarse_r

    def apply_bcs_2d(self, u: np.ndarray, nxe: int):
        nx = nxe + 1
        U = u.reshape((nx * nx, 3))

        for j in range(nx):
            for i in range(nx):
                on_edge = (i == 0) or (i == nx - 1) or (j == 0) or (j == nx - 1)
                if not on_edge:
                    continue
                k = i + nx * j
                U[k, 0] = 0.0
                if self.clamped:
                    U[k, 1] = 0.0
                    U[k, 2] = 0.0
