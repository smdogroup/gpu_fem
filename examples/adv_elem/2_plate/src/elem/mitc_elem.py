import numpy as np
import scipy.sparse as sp

from .basis import second_order_quadrature
from .basis import get_lagrange_basis_2d_all


class MITCPlateElement_OptProlong:
    """
    Q4 Reissner–Mindlin with MITC4 shear (tying points):
      gamma_xz tied at (xi, eta) = (0, ±b)
      gamma_yz tied at (xi, eta) = (±a, 0)

    Then at each *regular* quadrature point (no reduced integration),
    gamma_xz(xi,eta) is interpolated from the two tying points in eta,
    gamma_yz(xi,eta) is interpolated from the two tying points in xi.

    Locking-aware prolong constraints are applied on the *tying strains*:
      gamma_xz(0,+b)=0, gamma_xz(0,-b)=0, gamma_yz(+a,0)=0, gamma_yz(-a,0)=0
    giving 4 constraints per element (fine and coarse).
    """

    def __init__(
        self,
        prolong_mode: str = "locking-global",  # 'standard'
        lam: float = 1e-2,
        a: float = 1.0 / np.sqrt(3.0),        # typical choice consistent w/ 2x2 Gauss
        b: float = 1.0 / np.sqrt(3.0),
    ):
        assert prolong_mode in ["locking-global", "locking-local", "standard"]

        self.dof_per_node = 3
        self.nodes_per_elem = 4
        self.ndof = self.dof_per_node * self.nodes_per_elem

        self.prolong_mode = prolong_mode
        self.clamped = False
        self.lam = float(lam)

        self.a = float(a)
        self.b = float(b)
        assert abs(self.a) > 0.0 and abs(self.b) > 0.0

        # cache for prolong/restrict operators
        self._P1_cache = {}   # key: nxe_coarse -> P1 (csr)
        self._P2_cache = {}   # key: nxe_coarse -> P2 (csr)
        self._P2_u3_cache = {}
        self._lock_P_cache = {}

    # ----------------------------
    # MITC helpers
    # ----------------------------
    @staticmethod
    def _interp_1d_pm(s: float, a: float) -> np.ndarray:
        """
        Linear 1D Lagrange basis to interpolate from points at [-a, +a] to s.
        Returns [N_- , N_+], where:
          N_- = (a - s)/(2a), N_+ = (a + s)/(2a)
        """
        a = float(a)
        return np.array([(a - s) / (2.0 * a), (a + s) / (2.0 * a)], dtype=float)

    @staticmethod
    def _geom_map_and_grads(Nxi, Neta, x, y):
        x_xi = float(np.dot(Nxi, x))
        x_eta = float(np.dot(Neta, x))
        y_xi = float(np.dot(Nxi, y))
        y_eta = float(np.dot(Neta, y))

        J = x_xi * y_eta - x_eta * y_xi
        invJ = 1.0 / J

        xi_x = y_eta * invJ
        xi_y = -x_eta * invJ
        eta_x = -y_xi * invJ
        eta_y = x_xi * invJ

        Nx = Nxi * xi_x + Neta * eta_x
        Ny = Nxi * xi_y + Neta * eta_y
        return J, Nx, Ny

    def _Bs_rows_at_point(self, xi: float, eta: float, x: np.ndarray, y: np.ndarray):
        """
        Return the *pointwise* shear B rows (two 1x12 rows) for:
          gamma_xz = w_x + thx
          gamma_yz = w_y + thy
        evaluated at (xi,eta).
        """
        N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)
        J, Nx, Ny = self._geom_map_and_grads(Nxi, Neta, x, y)

        bx = np.zeros((12,), dtype=float)
        by = np.zeros((12,), dtype=float)

        # gamma_xz = w_x + thx
        bx[0::3] = Nx
        bx[1::3] = N

        # gamma_yz = w_y + thy
        by[0::3] = Ny
        by[2::3] = N

        return J, bx, by

    def _Bs_mitc_at_quad(self, xi: float, eta: float, x: np.ndarray, y: np.ndarray):
        """
        Build the *effective* MITC shear B matrix (2x12) at the quadrature point (xi,eta):
          row0 = interpolated gamma_xz from tying at (0, ±b) using eta-basis
          row1 = interpolated gamma_yz from tying at (±a, 0) using xi-basis
        """
        # Geometry/J evaluated at the quad point for integration measure
        Nq, Nxi_q, Neta_q = get_lagrange_basis_2d_all(xi, eta)
        Jq, _, _ = self._geom_map_and_grads(Nxi_q, Neta_q, x, y)

        # Tying evaluations
        _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x, y)
        _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x, y)

        _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x, y)
        _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x, y)

        # Interpolation weights
        w_eta = self._interp_1d_pm(eta, self.b)  # from [-b,+b] -> eta
        w_xi  = self._interp_1d_pm(xi,  self.a)  # from [-a,+a] -> xi

        row_gx = w_eta[0] * bx_m + w_eta[1] * bx_p
        row_gy = w_xi[0]  * by_m + w_xi[1]  * by_p

        Bs = np.vstack([row_gx, row_gy])  # 2x12
        return Jq, Bs

    # ----------------------------
    # Element stiffness (MITC4)
    # ----------------------------
    def get_kelem(self, E: float, nu: float, thick: float, elem_xpts: np.ndarray):
        """
        elem_xpts length 12: [x0,y0,z0?, x1,y1,z1?, x2,y2,z2?, x3,y3,z3?]
        Uses x=elem_xpts[0::3], y=elem_xpts[1::3].
        """
        pts, wts = second_order_quadrature()  # 2nd order in each direction (3-pt if that's your impl)

        kelem = np.zeros((self.ndof, self.ndof))

        # material constants
        D0 = E * thick**3 / (12.0 * (1.0 - nu**2))
        Db = D0 * np.array([
            [1.0,  nu,          0.0],
            [nu,   1.0,         0.0],
            [0.0,  0.0, (1.0 - nu) / 2.0],
        ])
        ks = 5.0 / 6.0
        G = E / (2.0 * (1.0 + nu))
        Ds = (ks * G * thick) * np.eye(2)

        x = elem_xpts[0::3]
        y = elem_xpts[1::3]

        # ---- BENDING (unchanged): depends on rotation gradients only ----
        for ii, xi in enumerate(pts):
            for jj, eta in enumerate(pts):
                wt = wts[ii] * wts[jj]

                N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)
                J, Nx, Ny = self._geom_map_and_grads(Nxi, Neta, x, y)

                Bb = np.zeros((3, self.ndof))

                Bb[0, 1::3] = Nx          # d(thx)/dx
                Bb[1, 2::3] = Ny          # d(thy)/dy
                Bb[2, 1::3] = Ny          # d(thx)/dy
                Bb[2, 2::3] = Nx          # d(thy)/dx

                kelem += (Bb.T @ Db @ Bb) * (wt * J)

        # ---- SHEAR (MITC): full integration, BUT shear strains are tied/interpolated ----
        for ii, xi in enumerate(pts):
            for jj, eta in enumerate(pts):
                wt = wts[ii] * wts[jj]

                Jq, Bs = self._Bs_mitc_at_quad(xi, eta, x, y)  # 2x12 effective
                kelem += (Bs.T @ Ds @ Bs) * (wt * Jq)

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
    # Prolongation / Restriction (same as your class, but locking constraints changed)
    # ----------------------------
    def _build_P1_scalar(self, nxe_coarse: int) -> sp.csr_matrix:
        if nxe_coarse in self._P1_cache:
            return self._P1_cache[nxe_coarse]

        nc = nxe_coarse + 1
        nf = 2 * nxe_coarse + 1  # = 2*nc - 1

        rows, cols, vals = [], [], []

        for i in range(nc):
            rows.append(2 * i)
            cols.append(i)
            vals.append(1.0)

        for i in range(nc - 1):
            r = 2 * i + 1
            rows += [r, r]
            cols += [i, i + 1]
            vals += [0.5, 0.5]

        P1 = sp.coo_matrix((vals, (rows, cols)), shape=(nf, nc)).tocsr()
        self._P1_cache[nxe_coarse] = P1
        return P1

    def _build_P2_scalar(self, nxe_coarse: int) -> sp.csr_matrix:
        if nxe_coarse in self._P2_cache:
            return self._P2_cache[nxe_coarse]
        P1 = self._build_P1_scalar(nxe_coarse)
        P2 = sp.kron(P1, P1, format="csr")
        self._P2_cache[nxe_coarse] = P2
        return P2

    def _build_P2_uncoupled3(self, nxe_coarse: int) -> sp.csr_matrix:
        if nxe_coarse in self._P2_u3_cache:
            return self._P2_u3_cache[nxe_coarse]
        P2s = self._build_P2_scalar(nxe_coarse)
        P = sp.kron(P2s, sp.eye(3, format="csr"), format="csr")
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

    def _locking_aware_prolong_global_mitc(self, nxe_c: int, length: float = 1.0):
        """
        Locking-aware prolongation where constraints are on MITC tying strains:
          [ gx(0,-b), gx(0,+b), gy(-a,0), gy(+a,0) ] = 0   per element
        => 4 constraints per element.
        """
        if nxe_c in self._lock_P_cache:
            return self._lock_P_cache[nxe_c]

        # sizes
        nxe_f = 2 * nxe_c

        nx_f = nxe_f + 1
        nnodes_f = nx_f**2
        nelems_f = nxe_f**2
        N_f = 3 * nnodes_f

        nx_c = nxe_c + 1
        nnodes_c = nx_c**2
        nelems_c = nxe_c**2
        N_c = 3 * nnodes_c

        # element reference coords (axis-aligned mapping as in your current code)
        dx_f = length / nxe_f
        x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
        y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

        dx_c = length / nxe_c
        x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
        y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

        # Build telling-strain operator G_f, G_c (dense):
        # rows per element: [gx(0,-b), gx(0,+b), gy(-a,0), gy(+a,0)]
        G_f = np.zeros((4 * nelems_f, N_f), dtype=float)
        for ielem_f in range(nelems_f):
            ex = ielem_f % nxe_f
            ey = ielem_f // nxe_f
            loc_nodes = np.array([
                ex + nx_f * ey,
                (ex + 1) + nx_f * ey,
                (ex + 1) + nx_f * (ey + 1),
                ex + nx_f * (ey + 1),
            ], dtype=int)
            loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)], dtype=int)

            # gx at (0, -b) and (0, +b)
            _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_f, y_f)
            _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_f, y_f)
            # gy at (-a, 0) and (+a, 0)
            _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_f, y_f)
            _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_f, y_f)

            r0 = 4 * ielem_f
            G_f[r0 + 0, loc_dof] += bx_m
            G_f[r0 + 1, loc_dof] += bx_p
            G_f[r0 + 2, loc_dof] += by_m
            G_f[r0 + 3, loc_dof] += by_p

        G_c = np.zeros((4 * nelems_c, N_c), dtype=float)
        for ielem_c in range(nelems_c):
            ex = ielem_c % nxe_c
            ey = ielem_c // nxe_c
            loc_nodes = np.array([
                ex + nx_c * ey,
                (ex + 1) + nx_c * ey,
                (ex + 1) + nx_c * (ey + 1),
                ex + nx_c * (ey + 1),
            ], dtype=int)
            loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)], dtype=int)

            _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_c, y_c)
            _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_c, y_c)
            _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_c, y_c)
            _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_c, y_c)

            r0 = 4 * ielem_c
            G_c[r0 + 0, loc_dof] += bx_m
            G_c[r0 + 1, loc_dof] += bx_p
            G_c[r0 + 2, loc_dof] += by_m
            G_c[r0 + 3, loc_dof] += by_p

        # Elementwise injection for tying strains (4-per-elem)
        P_gam = np.zeros((4 * nelems_f, 4 * nelems_c), dtype=float)
        for ielem_f in range(nelems_f):
            ex = ielem_f % nxe_f
            ey = ielem_f // nxe_f
            ielem_c = (ex // 2) + (ey // 2) * nxe_c

            rf = 4 * ielem_f
            rc = 4 * ielem_c
            P_gam[rf + 0, rc + 0] = 1.0
            P_gam[rf + 1, rc + 1] = 1.0
            P_gam[rf + 2, rc + 2] = 1.0
            P_gam[rf + 3, rc + 3] = 1.0

        RHS = P_gam @ G_c  # (4*nelems_f, 3*nnodes_c)

        # Baseline nodal prolong
        P_0 = self._build_P2_uncoupled3(nxe_c) # csr
        P_0 = self._apply_bcs_to_P(P_0, nxe_c)
        lam = float(self.lam)

        # Coarse BC columns (same logic as your v2)
        constrained_dofs = (0, 1, 2) if self.clamped else (0,)

        fixed_cols_c = []
        for inode in range(nnodes_c):
            i = inode % nx_c
            j = inode // nx_c
            if (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1):
                base = 3 * inode
                for a in constrained_dofs:
                    fixed_cols_c.append(base + a)
        fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=int)
        all_cols_c = np.arange(3 * nnodes_c, dtype=int)
        free_cols_c = np.setdiff1d(all_cols_c, fixed_cols_c, assume_unique=False)

        # Fine BC rows with beam-style E-constraint
        fixed_rows_f = []
        for inode in range(nnodes_f):
            i = inode % nx_f
            j = inode // nx_f
            if (i == 0) or (i == nx_f - 1) or (j == 0) or (j == nx_f - 1):
                base = 3 * inode
                for a in constrained_dofs:
                    fixed_rows_f.append(base + a)
        fixed_rows_f = np.array(sorted(set(fixed_rows_f)), dtype=int)

        solve_rows_f = np.arange(3 * nnodes_f, dtype=int)

        nE = fixed_rows_f.size
        Esel = np.zeros((nE, solve_rows_f.size), dtype=float)
        # solve_rows_f is identity, so:
        Esel[np.arange(nE), fixed_rows_f] = 1.0

        # Least squares solve:
        #   minimize ||G_f P - RHS||^2 + ||E P||^2 + lam ||P - P0||^2
        A = G_f[:, solve_rows_f]                  # (4*nelems_f, nsolve)
        B = RHS[:, free_cols_c]                   # (4*nelems_f, nfreecols)

        A_aug = np.vstack([A, Esel])
        B_aug = np.vstack([B, np.zeros((nE, B.shape[1]))])
        # A_aug = A
        # B_aug = B

        idx0 = np.ix_(solve_rows_f, free_cols_c)
        P0_free = P_0[idx0].toarray()

        M = A_aug.T @ A_aug + lam * np.eye(solve_rows_f.size)
        rhs = A_aug.T @ B_aug + lam * P0_free
        P_free = np.linalg.solve(M, rhs)

        # Assemble full P
        P = P_0.toarray()
        P[:, fixed_cols_c] = 0.0
        P[np.ix_(solve_rows_f, free_cols_c)] = P_free
        # P[fixed_rows_f, :] = 0.0

        self._lock_P_cache[nxe_c] = P.copy()
        return P

    def prolongate(self, coarse_u: np.ndarray, nxe_coarse: int):
        dpn = self.dof_per_node
        nxc = nxe_coarse + 1
        Nc = nxc * nxc
        assert coarse_u.size == dpn * Nc

        nxe_f = 2 * nxe_coarse
        nxf = nxe_f + 1
        Nf = nxf * nxf

        if self.prolong_mode == "locking-global":
            P = self._locking_aware_prolong_global_mitc(nxe_coarse, length=1.0)
        elif self.prolong_mode == "standard":
            P = self._build_P2_uncoupled3(nxe_coarse)
            P = self._apply_bcs_to_P(P, nxe_coarse)
        else:
            raise NotImplementedError("locking-local not implemented in this prototype")

        fine_u = P @ coarse_u

        return fine_u

    def restrict_defect(self, fine_r: np.ndarray, nxe_fine: int):
        dpn = self.dof_per_node
        nxf = nxe_fine + 1
        Nf = nxf * nxf
        assert fine_r.size == dpn * Nf
        assert (nxe_fine % 2) == 0

        nxe_coarse = nxe_fine // 2
        nxc = nxe_coarse + 1
        Nc = nxc * nxc

        if self.prolong_mode == "locking-global":
            P = self._locking_aware_prolong_global_mitc(nxe_coarse, length=1.0)
        elif self.prolong_mode == "standard":
            P = self._build_P2_uncoupled3(nxe_coarse)
            P = self._apply_bcs_to_P(P, nxe_coarse)
        else:
            raise NotImplementedError("locking-local not implemented in this prototype")

        R = P.T

        fine_r = fine_r.copy()
        self.apply_bcs_2d(fine_r, nxe_fine)

        coarse_r = R @ fine_r

        return coarse_r
