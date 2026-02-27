import numpy as np
import scipy.sparse as sp

from .basis import second_order_quadrature
from .basis import get_lagrange_basis_2d_all
from ._utils import debug_print_bsr_matrix


import numpy as np
import scipy.sparse as sp

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
        a: float = 1.0,        # typical choice consistent w/ 2x2 Gauss
        b: float = 1.0,
        n_lock_sweeps:int=10,
        omega:float=0.5,
        debug:bool=False,
    ):
        assert prolong_mode in ["locking-global", "locking-local", "standard", "energy-jacobi"]

        self.dof_per_node = 3
        self.nodes_per_elem = 4
        self.ndof = self.dof_per_node * self.nodes_per_elem
        self.debug = debug

        self.prolong_mode = prolong_mode
        self.clamped = False
        self.lam = float(lam)
        self.n_lock_sweeps = n_lock_sweeps
        self.omega = float(omega)

        self.a = float(a)
        self.b = float(b)
        assert abs(self.a) > 0.0 and abs(self.b) > 0.0

        # cache for prolong/restrict operators
        self._P1_cache = {}   # key: nxe_coarse -> P1 (csr)
        self._P2_cache = {}   # key: nxe_coarse -> P2 (csr)
        self._P2_u3_cache = {}
        self._lock_P_cache = {}
        self._kmat_cache = {}

    # # ----------------------------
    # # MITC helpers, with old (thx, thy) + grad w order
    # # ----------------------------
    # @staticmethod
    # def _interp_1d_pm(s: float, a: float) -> np.ndarray:
    #     """
    #     Linear 1D Lagrange basis to interpolate from points at [-a, +a] to s.
    #     Returns [N_- , N_+], where:
    #       N_- = (a - s)/(2a), N_+ = (a + s)/(2a)
    #     """
    #     a = float(a)
    #     return np.array([(a - s) / (2.0 * a), (a + s) / (2.0 * a)], dtype=float)

    # @staticmethod
    # def _geom_map_and_grads(Nxi, Neta, x, y):
    #     x_xi = float(np.dot(Nxi, x))
    #     x_eta = float(np.dot(Neta, x))
    #     y_xi = float(np.dot(Nxi, y))
    #     y_eta = float(np.dot(Neta, y))

    #     J = x_xi * y_eta - x_eta * y_xi
    #     invJ = 1.0 / J

    #     xi_x = y_eta * invJ
    #     xi_y = -x_eta * invJ
    #     eta_x = -y_xi * invJ
    #     eta_y = x_xi * invJ

    #     Nx = Nxi * xi_x + Neta * eta_x
    #     Ny = Nxi * xi_y + Neta * eta_y
    #     return J, Nx, Ny

    # def _Bs_rows_at_point(self, xi: float, eta: float, x: np.ndarray, y: np.ndarray):
    #     """
    #     Return the *pointwise* shear B rows (two 1x12 rows) for:
    #       gamma_xz = w_x + thx
    #       gamma_yz = w_y + thy
    #     evaluated at (xi,eta).
    #     """
    #     N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)
    #     J, Nx, Ny = self._geom_map_and_grads(Nxi, Neta, x, y)

    #     bx = np.zeros((12,), dtype=float)
    #     by = np.zeros((12,), dtype=float)

    #     # gamma_xz = w_x + thx
    #     bx[0::3] = Nx
    #     bx[1::3] = N

    #     # gamma_yz = w_y + thy
    #     by[0::3] = Ny
    #     by[2::3] = N

    #     return J, bx, by

    # def _Bs_mitc_at_quad(self, xi: float, eta: float, x: np.ndarray, y: np.ndarray):
    #     """
    #     Build the *effective* MITC shear B matrix (2x12) at the quadrature point (xi,eta):
    #       row0 = interpolated gamma_xz from tying at (0, ±b) using eta-basis
    #       row1 = interpolated gamma_yz from tying at (±a, 0) using xi-basis
    #     """
    #     # Geometry/J evaluated at the quad point for integration measure
    #     Nq, Nxi_q, Neta_q = get_lagrange_basis_2d_all(xi, eta)
    #     Jq, _, _ = self._geom_map_and_grads(Nxi_q, Neta_q, x, y)

    #     # Tying evaluations
    #     _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x, y)
    #     _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x, y)

    #     _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x, y)
    #     _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x, y)

    #     # Interpolation weights
    #     w_eta = self._interp_1d_pm(eta, self.b)  # from [-b,+b] -> eta
    #     w_xi  = self._interp_1d_pm(xi,  self.a)  # from [-a,+a] -> xi

    #     row_gx = w_eta[0] * bx_m + w_eta[1] * bx_p
    #     row_gy = w_xi[0]  * by_m + w_xi[1]  * by_p

    #     Bs = np.vstack([row_gx, row_gy])  # 2x12
    #     return Jq, Bs

    # # ----------------------------
    # # Element stiffness (MITC4)
    # # ----------------------------
    # def get_kelem(self, E: float, nu: float, thick: float, elem_xpts: np.ndarray):
    #     """
    #     elem_xpts length 12: [x0,y0,z0?, x1,y1,z1?, x2,y2,z2?, x3,y3,z3?]
    #     Uses x=elem_xpts[0::3], y=elem_xpts[1::3].
    #     """
    #     pts, wts = second_order_quadrature()  # 2nd order in each direction (3-pt if that's your impl)

    #     kelem = np.zeros((self.ndof, self.ndof))

    #     # material constants
    #     D0 = E * thick**3 / (12.0 * (1.0 - nu**2))
    #     Db = D0 * np.array([
    #         [1.0,  nu,          0.0],
    #         [nu,   1.0,         0.0],
    #         [0.0,  0.0, (1.0 - nu) / 2.0],
    #     ])
    #     ks = 5.0 / 6.0
    #     G = E / (2.0 * (1.0 + nu))
    #     Ds = (ks * G * thick) * np.eye(2)

    #     x = elem_xpts[0::3]
    #     y = elem_xpts[1::3]

    #     # ---- BENDING (unchanged): depends on rotation gradients only ----
    #     for ii, xi in enumerate(pts):
    #         for jj, eta in enumerate(pts):
    #             wt = wts[ii] * wts[jj]

    #             N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)
    #             J, Nx, Ny = self._geom_map_and_grads(Nxi, Neta, x, y)

    #             Bb = np.zeros((3, self.ndof))

    #             Bb[0, 1::3] = Nx          # d(thx)/dx
    #             Bb[1, 2::3] = Ny          # d(thy)/dy
    #             Bb[2, 1::3] = Ny          # d(thx)/dy
    #             Bb[2, 2::3] = Nx          # d(thy)/dx

    #             kelem += (Bb.T @ Db @ Bb) * (wt * J)

    #     # ---- SHEAR (MITC): full integration, BUT shear strains are tied/interpolated ----
    #     for ii, xi in enumerate(pts):
    #         for jj, eta in enumerate(pts):
    #             wt = wts[ii] * wts[jj]

    #             Jq, Bs = self._Bs_mitc_at_quad(xi, eta, x, y)  # 2x12 effective
    #             kelem += (Bs.T @ Ds @ Bs) * (wt * Jq)

    #     return kelem

    # ----------------------------
    # MITC helpers using (thy, -thx) directors + grad w ordering
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
    def _geom_map_and_grads(Nxi, Neta, x, y, debug: bool = False):
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

        # if debug:
        #     print(f"{Nxi=}\n{Neta=}")

        Nx = Nxi * xi_x + Neta * eta_x
        Ny = Nxi * xi_y + Neta * eta_y
        return J, Nx, Ny

    def _Bs_rows_at_point(self, xi: float, eta: float, x: np.ndarray, y: np.ndarray):
        """
        Return the *pointwise* shear B rows (two 1x12 rows) for the director choice:
            director = (thy, -thx)

        i.e. transverse shear strains:
            gamma_xz = w_x + thy
            gamma_yz = w_y - thx

        evaluated at (xi,eta).

        DOF ordering per node: [w, thx, thy]
        """
        N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)
        J, Nx, Ny = self._geom_map_and_grads(Nxi, Neta, x, y, self.debug)

        bx = np.zeros((12,), dtype=float)
        by = np.zeros((12,), dtype=float)

        # if self.debug:
        #     print(f"{J=}\n{Nx=}\n{Ny=}\n{N=}")

        # gamma_xz = w_x + thy
        bx[0::3] = Nx
        bx[2::3] = +N

        # gamma_yz = w_y - thx
        by[0::3] = Ny
        by[1::3] = -N

        return J, bx, by

    def _Bs_mitc_at_quad(self, xi: float, eta: float, x: np.ndarray, y: np.ndarray):
        """
        Build the *effective* MITC shear B matrix (2x12) at the quadrature point (xi,eta):

        Uses the same director convention as _Bs_rows_at_point:
            gamma_xz = w_x + thy   tied from (0, ±b) and interpolated in eta
            gamma_yz = w_y - thx   tied from (±a, 0) and interpolated in xi
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

        Note: BENDING curvatures are formed consistently with director = (thy, -thx).
        """
        pts, wts = second_order_quadrature()

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

        # ---- BENDING: consistent with director = (thy, -thx) ----
        # Let (rx, ry) = (thy, -thx). Curvatures:
        #   kappa_x  = d(rx)/dx  =  d(thy)/dx
        #   kappa_y  = d(ry)/dy  = -d(thx)/dy
        #   kappa_xy = d(rx)/dy + d(ry)/dx = d(thy)/dy - d(thx)/dx
        for ii, xi in enumerate(pts):
            for jj, eta in enumerate(pts):
                wt = wts[ii] * wts[jj]

                N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)
                J, Nx, Ny = self._geom_map_and_grads(Nxi, Neta, x, y)

                Bb = np.zeros((3, self.ndof))

                # kappa_x = d(thy)/dx
                Bb[0, 2::3] = Nx

                # kappa_y = -d(thx)/dy
                Bb[1, 1::3] = -Ny

                # kappa_xy = d(thy)/dy - d(thx)/dx
                Bb[2, 2::3] = Ny
                Bb[2, 1::3] = -Nx

                kelem += (Bb.T @ Db @ Bb) * (wt * J)

        # ---- SHEAR (MITC): full integration, tied/interpolated ----
        for ii, xi in enumerate(pts):
            for jj, eta in enumerate(pts):
                wt = wts[ii] * wts[jj]
                Jq, Bs = self._Bs_mitc_at_quad(xi, eta, x, y)
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

    def _locking_aware_prolong_global_mitc_v1(self, nxe_c: int, length: float = 1.0):
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

        # # Elementwise injection for tying strains (4-per-elem)
        # P_gam = np.zeros((4 * nelems_f, 4 * nelems_c), dtype=float)
        # for ielem_f in range(nelems_f):
        #     ex = ielem_f % nxe_f
        #     ey = ielem_f // nxe_f
        #     ielem_c = (ex // 2) + (ey // 2) * nxe_c

        #     rf = 4 * ielem_f
        #     rc = 4 * ielem_c
        #     P_gam[rf + 0, rc + 0] = 1.0
        #     P_gam[rf + 1, rc + 1] = 1.0
        #     P_gam[rf + 2, rc + 2] = 1.0
        #     P_gam[rf + 3, rc + 3] = 1.0

        # ---------------------------------------------------------
        # P_gam: bilinear averaging (Q1) from coarse elem-grid to fine elem-grid
        # Each strain component is interpolated independently.
        #
        # coarse elements live on a (nxe_c x nxe_c) grid with indices (Ex_c, Ey_c)
        # fine elements live on a (nxe_f x nxe_f) grid with indices (ex, ey)
        #
        # Map fine element center to coarse-index space:
        #   x_c = (ex + 0.5)/2 - 0.5   in [ -0.25, nxe_c - 0.75 ]
        # so that fine elements in a 2x2 block around a coarse element "see" neighbors.
        #
        # You can tweak the "-0.5" shift if you want less cross-element blending.
        # ---------------------------------------------------------
        P_gam = np.zeros((4 * nelems_f, 4 * nelems_c), dtype=float)

        # NOTE : elemwise injection gives very similar thin shell perf

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        for ielem_f in range(nelems_f):
            ex = ielem_f % nxe_f
            ey = ielem_f // nxe_f

            # fine element "center" mapped into coarse-element index space
            x = 0.5 * (ex + 0.5) - 0.5
            y = 0.5 * (ey + 0.5) - 0.5

            i0 = int(np.floor(x))
            j0 = int(np.floor(y))
            tx = x - i0
            ty = y - j0

            # clamp base so neighbors exist; edge blending degenerates gracefully
            i0 = clamp(i0, 0, nxe_c - 1)
            j0 = clamp(j0, 0, nxe_c - 1)

            i1 = clamp(i0 + 1, 0, nxe_c - 1)
            j1 = clamp(j0 + 1, 0, nxe_c - 1)

            # if clamped to boundary, kill the corresponding fraction
            if i1 == i0:
                tx = 0.0
            if j1 == j0:
                ty = 0.0

            w00 = (1.0 - tx) * (1.0 - ty)
            w10 = (tx)       * (1.0 - ty)
            w01 = (1.0 - tx) * (ty)
            w11 = (tx)       * (ty)

            # coarse element ids
            e00 = i0 + j0 * nxe_c
            e10 = i1 + j0 * nxe_c
            e01 = i0 + j1 * nxe_c
            e11 = i1 + j1 * nxe_c

            rf = 4 * ielem_f

            # for each tying-strain component, interpolate from coarse neighbors
            for comp in range(4):
                P_gam[rf + comp, 4 * e00 + comp] += w00
                P_gam[rf + comp, 4 * e10 + comp] += w10
                P_gam[rf + comp, 4 * e01 + comp] += w01
                P_gam[rf + comp, 4 * e11 + comp] += w11

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
        
        # direct solve
        P_free = np.linalg.solve(M, rhs)

        # same some states to element class for DEBUGGING in locking sandbox 
        # need to get it out of this class
        self.G_f = G_f
        self.G_c = G_c
        self.P_gam = P_gam
        self.P_0 = P_0
        self.M = M
        self.RHS = rhs
        self.free_cols_c = free_cols_c
        self.fixed_cols_c = fixed_cols_c
        self.solve_rows_f = solve_rows_f
        self.P_0_free = P0_free
        self.Mb = None

        # # block-Jacobi smoothing (3x3 nodal) instead of direct solve
        # P_free = P0_free.copy()
        # omega = 0.8
        # # n_smooth = 5 (not enough)
        # # n_smooth = 15
        # # n_smooth = 30 # still takes 80 krylov iterations
        # n_smooth = 60

        # n = M.shape[0]
        # assert n % 3 == 0

        # # precompute inv(diag 3x3 blocks)
        # Dinv = np.empty((n//3, 3, 3), dtype=M.dtype)
        # for b in range(n // 3):
        #     i = 3 * b
        #     Dinv[b] = np.linalg.inv(M[i:i+3, i:i+3])

        # for _ in range(n_smooth):
        #     R = rhs - M @ P_free
        #     for b in range(n // 3):
        #         i = 3 * b
        #         P_free[i:i+3, :] += omega * (Dinv[b] @ R[i:i+3, :])


        # Assemble full P
        P = P_0.toarray()
        P[:, fixed_cols_c] = 0.0
        P[np.ix_(solve_rows_f, free_cols_c)] = P_free
        # P[fixed_rows_f, :] = 0.0

        self._lock_P_cache[nxe_c] = P.copy()
        return P
    
    # def _locking_aware_prolong_global_mitc_v2(self, nxe_c: int, length: float = 1.0):
    #     """
    #     Locking-aware prolongation for MITC tying strains with:
    #     1) blended P_gam (injection + bilinear averaging)
    #     2) eliminate fine Dirichlet rows from the unknowns (hard BCs, no Esel)
    #     3) row-scaled constraint equations (conditioning)
    #     4) lam scaled to ||A^T A|| + tiny nugget, and explicit symmetrization
    #     5) final hard BC projection on P
    #     """
    #     if nxe_c in self._lock_P_cache:
    #         return self._lock_P_cache[nxe_c]

    #     import numpy as np
    #     import scipy.sparse as sp

    #     # --------------------
    #     # sizes
    #     # --------------------
    #     nxe_f = 2 * nxe_c

    #     nx_f = nxe_f + 1
    #     nnodes_f = nx_f**2
    #     nelems_f = nxe_f**2
    #     N_f = 3 * nnodes_f

    #     nx_c = nxe_c + 1
    #     nnodes_c = nx_c**2
    #     nelems_c = nxe_c**2
    #     N_c = 3 * nnodes_c

    #     # element reference coords (axis-aligned mapping as in your current code)
    #     dx_f = length / nxe_f
    #     x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
    #     y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

    #     dx_c = length / nxe_c
    #     x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
    #     y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

    #     # --------------------
    #     # Build tying-strain operators G_f, G_c (dense)
    #     # rows per element: [gx(0,-b), gx(0,+b), gy(-a,0), gy(+a,0)]
    #     # --------------------
    #     G_f = np.zeros((4 * nelems_f, N_f), dtype=float)
    #     for ielem_f in range(nelems_f):
    #         ex = ielem_f % nxe_f
    #         ey = ielem_f // nxe_f
    #         loc_nodes = np.array([
    #             ex + nx_f * ey,
    #             (ex + 1) + nx_f * ey,
    #             (ex + 1) + nx_f * (ey + 1),
    #             ex + nx_f * (ey + 1),
    #         ], dtype=int)
    #         loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)], dtype=int)

    #         _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_f, y_f)
    #         _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_f, y_f)
    #         _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_f, y_f)
    #         _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_f, y_f)

    #         r0 = 4 * ielem_f
    #         G_f[r0 + 0, loc_dof] += bx_m
    #         G_f[r0 + 1, loc_dof] += bx_p
    #         G_f[r0 + 2, loc_dof] += by_m
    #         G_f[r0 + 3, loc_dof] += by_p

    #     G_c = np.zeros((4 * nelems_c, N_c), dtype=float)
    #     for ielem_c in range(nelems_c):
    #         ex = ielem_c % nxe_c
    #         ey = ielem_c // nxe_c
    #         loc_nodes = np.array([
    #             ex + nx_c * ey,
    #             (ex + 1) + nx_c * ey,
    #             (ex + 1) + nx_c * (ey + 1),
    #             ex + nx_c * (ey + 1),
    #         ], dtype=int)
    #         loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)], dtype=int)

    #         _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_c, y_c)
    #         _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_c, y_c)
    #         _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_c, y_c)
    #         _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_c, y_c)

    #         r0 = 4 * ielem_c
    #         G_c[r0 + 0, loc_dof] += bx_m
    #         G_c[r0 + 1, loc_dof] += bx_p
    #         G_c[r0 + 2, loc_dof] += by_m
    #         G_c[r0 + 3, loc_dof] += by_p

    #     # --------------------
    #     # P_gam: blend injection + bilinear averaging
    #     # --------------------
    #     def clamp(v, lo, hi):
    #         return max(lo, min(hi, v))

    #     # injection
    #     P_inj = np.zeros((4 * nelems_f, 4 * nelems_c), dtype=float)
    #     for ielem_f in range(nelems_f):
    #         ex = ielem_f % nxe_f
    #         ey = ielem_f // nxe_f
    #         ielem_c = (ex // 2) + (ey // 2) * nxe_c
    #         rf = 4 * ielem_f
    #         rc = 4 * ielem_c
    #         for comp in range(4):
    #             P_inj[rf + comp, rc + comp] = 1.0

    #     # bilinear averaging on coarse elem-grid
    #     P_avg = np.zeros_like(P_inj)
    #     for ielem_f in range(nelems_f):
    #         ex = ielem_f % nxe_f
    #         ey = ielem_f // nxe_f

    #         # fine element center mapped into coarse-index space (same as your current)
    #         x = 0.5 * (ex + 0.5) - 0.5
    #         y = 0.5 * (ey + 0.5) - 0.5

    #         i0 = int(np.floor(x))
    #         j0 = int(np.floor(y))
    #         tx = x - i0
    #         ty = y - j0

    #         i0 = clamp(i0, 0, nxe_c - 1)
    #         j0 = clamp(j0, 0, nxe_c - 1)
    #         i1 = clamp(i0 + 1, 0, nxe_c - 1)
    #         j1 = clamp(j0 + 1, 0, nxe_c - 1)

    #         if i1 == i0:
    #             tx = 0.0
    #         if j1 == j0:
    #             ty = 0.0

    #         w00 = (1.0 - tx) * (1.0 - ty)
    #         w10 = (tx)       * (1.0 - ty)
    #         w01 = (1.0 - tx) * (ty)
    #         w11 = (tx)       * (ty)

    #         e00 = i0 + j0 * nxe_c
    #         e10 = i1 + j0 * nxe_c
    #         e01 = i0 + j1 * nxe_c
    #         e11 = i1 + j1 * nxe_c

    #         rf = 4 * ielem_f
    #         for comp in range(4):
    #             P_avg[rf + comp, 4 * e00 + comp] += w00
    #             P_avg[rf + comp, 4 * e10 + comp] += w10
    #             P_avg[rf + comp, 4 * e01 + comp] += w01
    #             P_avg[rf + comp, 4 * e11 + comp] += w11

    #     # blend (more injection keeps locality; more avg reduces jumps)
    #     alpha = 0.75  # try 0.6–0.9
    #     P_gam = alpha * P_inj + (1.0 - alpha) * P_avg

    #     RHS = P_gam @ G_c  # (4*nelems_f, 3*nnodes_c)

    #     # --------------------
    #     # Baseline nodal prolong + hard BC structure
    #     # --------------------
    #     P_0 = self._build_P2_uncoupled3(nxe_c).tocsr()
    #     P_0 = self._apply_bcs_to_P(P_0, nxe_c).tocsr()

    #     # which dofs are constrained at a boundary node
    #     constrained_dofs = (0, 1, 2) if self.clamped else (0,)

    #     # coarse fixed columns
    #     fixed_cols_c = []
    #     for inode in range(nnodes_c):
    #         i = inode % nx_c
    #         j = inode // nx_c
    #         if (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1):
    #             base = 3 * inode
    #             for a in constrained_dofs:
    #                 fixed_cols_c.append(base + a)
    #     fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=int)

    #     all_cols_c = np.arange(3 * nnodes_c, dtype=int)
    #     free_cols_c = np.setdiff1d(all_cols_c, fixed_cols_c, assume_unique=False)

    #     # fine fixed rows (Dirichlet rows) — we REMOVE these from the unknowns
    #     fixed_rows_f = []
    #     for inode in range(nnodes_f):
    #         i = inode % nx_f
    #         j = inode // nx_f
    #         if (i == 0) or (i == nx_f - 1) or (j == 0) or (j == nx_f - 1):
    #             base = 3 * inode
    #             for a in constrained_dofs:
    #                 fixed_rows_f.append(base + a)
    #     fixed_rows_f = np.array(sorted(set(fixed_rows_f)), dtype=int)

    #     all_rows_f = np.arange(3 * nnodes_f, dtype=int)
    #     free_rows_f = np.setdiff1d(all_rows_f, fixed_rows_f, assume_unique=False)

    #     # --------------------
    #     # Least-squares solve on FREE fine rows only
    #     #   minimize ||G_f(:,free_rows) P - RHS||^2 + lam_eff ||P - P0||^2
    #     # with row scaling + lam scaling + nugget
    #     # --------------------
    #     A = G_f[:, free_rows_f]          # (4*nelems_f, n_free_rows)
    #     B = RHS[:, free_cols_c]          # (4*nelems_f, n_free_cols)

    #     # Row scaling for conditioning
    #     row_norm = np.linalg.norm(A, axis=1)
    #     row_norm[row_norm == 0.0] = 1.0
    #     W = 1.0 / row_norm
    #     # clamp to avoid insane weights
    #     W = np.clip(W, 1e-2, 1e2)
    #     A = W[:, None] * A
    #     B = W[:, None] * B

    #     # Regularization target from baseline prolong
    #     idx0 = np.ix_(free_rows_f, free_cols_c)
    #     P0_free = P_0[idx0].toarray()

    #     # Build normal equations with scaled lambda and tiny nugget
    #     AtA = A.T @ A
    #     AtB = A.T @ B

    #     diag_mean = float(np.mean(np.diag(AtA))) if AtA.shape[0] > 0 else 1.0
    #     lam0 = float(self.lam)
    #     lam_eff = lam0 * diag_mean
    #     nugget = 1e-12 * diag_mean

    #     M = AtA + (lam_eff + nugget) * np.eye(AtA.shape[0])
    #     M = 0.5 * (M + M.T)  # enforce symmetry numerically
    #     rhs = AtB + lam_eff * P0_free

    #     P_free = np.linalg.solve(M, rhs)  # (n_free_rows, n_free_cols)

    #     # --------------------
    #     # Assemble full P and hard-enforce BC structure again
    #     # --------------------
    #     P = P_0.toarray()  # already BC-projected baseline

    #     # kill coarse boundary cols (consistency)
    #     P[:, fixed_cols_c] = 0.0

    #     # insert solved block on free rows/cols
    #     P[np.ix_(free_rows_f, free_cols_c)] = P_free

    #     # hard zero fine Dirichlet rows (no leakage)
    #     P[fixed_rows_f, :] = 0.0

    #     # final BC projection (belt + suspenders)
    #     P = self._apply_bcs_to_P(sp.csr_matrix(P), nxe_c).toarray()

    #     self._lock_P_cache[nxe_c] = P.copy()
    #     return P

    # def _locking_aware_prolong_local_mitc(self,
    #                                     nxe_c: int,
    #                                     length: float = 1.0,
    #                                     n_sweeps: int = 1,
    #                                     omega: float = 1.0,
    #                                     col_batch: int = 32):
    #     """
    #     SUPER-SIMPLIFIED local locking-aware prolongation for MITC tying strains.

    #     Solves approximately on free coarse columns:
    #         (Gf^T Gf + lam I) P = Gf^T (Pgam Gc) + lam P0

    #     using ELEMENT-BLOCK Gauss–Seidel (12 dof per fine element) but *matrix-free*:
    #     residual r_e = Be^T (Btarget_e - Be * P_e) + lam (P0_e - P_e)
    #     update: P_e <- P_e + omega * (He^{-1} r_e - P_e)   (He = Be^T Be + lam I)

    #     Notes:
    #     - No formation of Gf^T Gf or RHS.
    #     - Column batching keeps memory low.
    #     - Very GPU-mappable: gather 12 dofs, 4x12 matvec, 12x4 matvec, tiny 12x12 solve.
    #     """

    #     if nxe_c in self._lock_P_cache:
    #         return self._lock_P_cache[nxe_c]

    #     # -----------------------
    #     # sizes
    #     # -----------------------
    #     nxe_f = 2 * nxe_c
    #     nx_f = nxe_f + 1
    #     nn_f = nx_f * nx_f
    #     N_f = 3 * nn_f

    #     nx_c = nxe_c + 1
    #     nn_c = nx_c * nx_c
    #     N_c = 3 * nn_c

    #     lam = float(self.lam)

    #     # -----------------------
    #     # baseline prolong (fine x coarse) and BC application like your global code
    #     # -----------------------
    #     P0 = self._build_P2_uncoupled3(nxe_c)  # csr fine x coarse
    #     P0 = self._apply_bcs_to_P(P0, nxe_c)   # keep consistent with your code
    #     P0 = P0.toarray()

    #     # -----------------------
    #     # coarse fixed columns (same as your global function)
    #     # -----------------------
    #     constrained_dofs = (0, 1, 2) if self.clamped else (0,)
    #     fixed_cols_c = []
    #     for inode in range(nn_c):
    #         i = inode % nx_c
    #         j = inode // nx_c
    #         if (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1):
    #             base = 3 * inode
    #             for a in constrained_dofs:
    #                 fixed_cols_c.append(base + a)
    #     fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=np.int32)
    #     all_cols_c = np.arange(N_c, dtype=np.int32)
    #     free_cols_c = np.setdiff1d(all_cols_c, fixed_cols_c, assume_unique=False)

    #     # Start iterate with baseline on free cols
    #     X = P0[:, free_cols_c].copy()  # (N_f, nfree)

    #     # -----------------------
    #     # element reference coords (same style as your global code)
    #     # -----------------------
    #     dx_f = length / nxe_f
    #     x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
    #     y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

    #     dx_c = length / nxe_c
    #     x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
    #     y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

    #     # -----------------------
    #     # precompute per-element tying matrix Bf (4x12) for ALL fine elements (cheap-ish)
    #     # and also coarse element tying matrix Bc (4x12)
    #     # -----------------------
    #     ne_f = nxe_f * nxe_f
    #     ne_c = nxe_c * nxe_c

    #     Bf = np.zeros((ne_f, 4, 12), dtype=np.float64)
    #     for ey in range(nxe_f):
    #         for ex in range(nxe_f):
    #             ef = ex + ey * nxe_f

    #             _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_f, y_f)
    #             _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_f, y_f)
    #             _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_f, y_f)
    #             _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_f, y_f)

    #             Bf[ef, 0, :] = bx_m
    #             Bf[ef, 1, :] = bx_p
    #             Bf[ef, 2, :] = by_m
    #             Bf[ef, 3, :] = by_p

    #     Bc = np.zeros((ne_c, 4, 12), dtype=np.float64)
    #     for ey in range(nxe_c):
    #         for ex in range(nxe_c):
    #             ec = ex + ey * nxe_c

    #             _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_c, y_c)
    #             _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_c, y_c)
    #             _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_c, y_c)
    #             _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_c, y_c)

    #             Bc[ec, 0, :] = bx_m
    #             Bc[ec, 1, :] = bx_p
    #             Bc[ec, 2, :] = by_m
    #             Bc[ec, 3, :] = by_p

    #     # -----------------------
    #     # precompute per-fine-element coarse neighbor stencil (4 neighbors + bilinear weights)
    #     # (your P_gam idea, but stored as compact stencil)
    #     # -----------------------
    #     nbr = np.zeros((ne_f, 4), dtype=np.int32)
    #     wts = np.zeros((ne_f, 4), dtype=np.float64)

    #     def clamp(v, lo, hi):
    #         return max(lo, min(hi, v))

    #     for ey in range(nxe_f):
    #         for ex in range(nxe_f):
    #             ef = ex + ey * nxe_f

    #             x = 0.5 * (ex + 0.5) - 0.5
    #             y = 0.5 * (ey + 0.5) - 0.5
    #             i0 = int(np.floor(x))
    #             j0 = int(np.floor(y))
    #             tx = x - i0
    #             ty = y - j0

    #             i0 = clamp(i0, 0, nxe_c - 1)
    #             j0 = clamp(j0, 0, nxe_c - 1)
    #             i1 = clamp(i0 + 1, 0, nxe_c - 1)
    #             j1 = clamp(j0 + 1, 0, nxe_c - 1)
    #             if i1 == i0: tx = 0.0
    #             if j1 == j0: ty = 0.0

    #             w00 = (1.0 - tx) * (1.0 - ty)
    #             w10 = tx * (1.0 - ty)
    #             w01 = (1.0 - tx) * ty
    #             w11 = tx * ty

    #             e00 = i0 + j0 * nxe_c
    #             e10 = i1 + j0 * nxe_c
    #             e01 = i0 + j1 * nxe_c
    #             e11 = i1 + j1 * nxe_c

    #             nbr[ef, :] = (e00, e10, e01, e11)
    #             wts[ef, :] = (w00, w10, w01, w11)

    #     # -----------------------
    #     # helper: element -> its 12 global dofs (fine or coarse)
    #     # -----------------------
    #     def elem_dofs(ex, ey, nx):
    #         n0 = ex + nx * ey
    #         nodes = (n0, n0 + 1, n0 + 1 + nx, n0 + nx)
    #         dofs = np.empty(12, dtype=np.int32)
    #         k = 0
    #         for n in nodes:
    #             base = 3 * n
    #             dofs[k:k+3] = (base, base + 1, base + 2)
    #             k += 3
    #         return dofs

    #     fine_edofs = [None] * ne_f
    #     for ey in range(nxe_f):
    #         for ex in range(nxe_f):
    #             ef = ex + ey * nxe_f
    #             fine_edofs[ef] = elem_dofs(ex, ey, nx_f)

    #     coarse_edofs = [None] * ne_c
    #     for ey in range(nxe_c):
    #         for ex in range(nxe_c):
    #             ec = ex + ey * nxe_c
    #             coarse_edofs[ec] = elem_dofs(ex, ey, nx_c)

    #     # -----------------------
    #     # precompute tiny 12x12 block inverses He^{-1} = (Bf^T Bf + lam I)^{-1}
    #     # (same for all elements here since geometry uniform; if not, store per element)
    #     # -----------------------
    #     # since Bf[ef] is same for all ef in your current code, just use ef=0
    #     H = (Bf[0].T @ Bf[0]) + lam * np.eye(12)
    #     H = 0.5 * (H + H.T)
    #     Hinv = np.linalg.inv(H)

    #     # -----------------------
    #     # MAIN: sweeps, batch columns to keep memory sane
    #     # -----------------------
    #     nfree = free_cols_c.size
    #     for jb in range(0, nfree, col_batch):
    #         j1 = min(nfree, jb + col_batch)
    #         cols = free_cols_c[jb:j1]
    #         Xb = X[:, jb:j1]                 # (Nf, nb)
    #         P0b = P0[:, cols]                # (Nf, nb)

    #         for _ in range(n_sweeps):
    #             for ef in range(ne_f):
    #                 dofs_f = fine_edofs[ef]

    #                 # gather local unknowns for this batch
    #                 Xe = Xb[dofs_f, :]       # (12, nb)
    #                 P0e = P0b[dofs_f, :]     # (12, nb)

    #                 # compute TARGET tying strains on this fine element for these coarse columns:
    #                 # Btarget = sum_k w_k * (Bc[ec_k] @ I_local(col)) where "I_local(col)" means
    #                 # the restriction of coarse column to the 12 dofs of that coarse element.
    #                 #
    #                 # simplest: for each neighbor coarse element ec, take the 12 entries of the coarse column
    #                 # (which are just the fine baseline P0 restricted to those coarse dofs? NO — we need the
    #                 # "identity" mapping of column basis; but since we are building a prolongation column-by-column,
    #                 # the coarse column is literally a basis vector at that coarse dof.)
    #                 #
    #                 # We can compute coarse element tying strains for each column by applying Bc to the
    #                 # *coarse element restriction* of the coarse basis vector e_j:
    #                 #
    #                 # That restriction is 1 if that global coarse dof is inside the element, else 0.
    #                 #
    #                 # Here we compute it cheaply by checking membership in the element dof list.
    #                 #
    #                 e00, e10, e01, e11 = nbr[ef]
    #                 w00, w10, w01, w11 = wts[ef]

    #                 Btar = np.zeros((4, cols.size), dtype=np.float64)
    #                 for ec, w in ((e00, w00), (e10, w10), (e01, w01), (e11, w11)):
    #                     if w == 0.0:
    #                         continue
    #                     dofs_c = coarse_edofs[ec]  # 12 global coarse dofs
    #                     # for each column, build local unit entry if that coarse dof is in this element
    #                     # local_vec has shape (12, nb) with 0/1 entries
    #                     local_vec = np.zeros((12, cols.size), dtype=np.float64)
    #                     # membership test: for each col global dof, set the local index if present
    #                     # (still Python-y, but simple and easy to port to GPU adjacency tables)
    #                     for jj, gcol in enumerate(cols):
    #                         hit = np.where(dofs_c == gcol)[0]
    #                         if hit.size:
    #                             local_vec[hit[0], jj] = 1.0
    #                     Btar += w * (Bc[ec] @ local_vec)  # (4,nb)

    #                 # current tying strains from fine dofs:
    #                 Se = Bf[ef] @ Xe              # (4, nb)

    #                 # local residual:
    #                 # r = Bf^T (Btar - Se) + lam (P0e - Xe)
    #                 r = (Bf[ef].T @ (Btar - Se)) + lam * (P0e - Xe)   # (12, nb)

    #                 # block update
    #                 dX = Hinv @ r
    #                 Xb[dofs_f, :] = Xe + omega * (dX - Xe)

    #         # write back this batch
    #         X[:, jb:j1] = Xb

    #     # assemble full P
    #     P = P0.copy()
    #     P[:, fixed_cols_c] = 0.0
    #     P[:, free_cols_c] = X

    #     self._lock_P_cache[nxe_c] = P.copy()

    #     return P

    def _locking_aware_prolong_local_mitc_v1(self,
                                            nxe_c: int,
                                            length: float = 1.0,
                                            block_nodes: int = 1,
                                            n_sweeps: int = 1,
                                            omega: float = 1.0):
        """
        Local (block Gauss–Seidel) locking-aware prolongation for MITC tying strains.

        Like the beam version:
        Solve approx for X on *interior fine dofs only* (boundary fine dofs held fixed):
            (A^T A + lam I) X = A^T B + lam P0

        Here:
        A = G_f[:, interior_fine_dofs]
        B = (P_gam @ G_c)[:, free_coarse_cols]   (built without forming P_gam explicitly)
        P0 = baseline prolong restricted to interior fine dofs and free coarse cols

        Blocks:
        block_nodes=1 -> 3x3 blocks per interior node (w, thx, thy)
        block_nodes=2 -> 6x6 blocks for consecutive interior nodes in our ordering

        Returns
        -------
        P : dense ndarray, shape (3*nnodes_f, 3*nnodes_c)
        """

        if nxe_c in self._lock_P_cache:
            return self._lock_P_cache[nxe_c]

        if block_nodes not in (1, 2):
            raise ValueError("block_nodes must be 1 or 2")

        lam = float(self.lam)

        # -----------------------------
        # sizes
        # -----------------------------
        nxe_f = 2 * nxe_c

        nx_f = nxe_f + 1
        nnodes_f = nx_f**2
        nelems_f = nxe_f**2
        N_f = 3 * nnodes_f

        nx_c = nxe_c + 1
        nnodes_c = nx_c**2
        nelems_c = nxe_c**2
        N_c = 3 * nnodes_c

        # -----------------------------
        # baseline prolong (fine x coarse) + BCs
        # -----------------------------
        P0 = self._build_P2_uncoupled3(nxe_c)  # csr
        P0 = self._apply_bcs_to_P(P0, nxe_c)   # csr
        P0 = P0.toarray()                     # dense (N_f, N_c)

        # -----------------------------
        # coarse fixed columns (same as your global)
        # -----------------------------
        constrained_dofs = (0, 1, 2) if self.clamped else (0,)

        fixed_cols_c = []
        for inode in range(nnodes_c):
            i = inode % nx_c
            j = inode // nx_c
            if (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1):
                base = 3 * inode
                for a in constrained_dofs:
                    fixed_cols_c.append(base + a)
        fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=np.int32)

        all_cols_c = np.arange(N_c, dtype=np.int32)
        free_cols_c = np.setdiff1d(all_cols_c, fixed_cols_c, assume_unique=False)

        # -----------------------------
        # fine interior dofs as unknowns (boundary fine dofs held fixed)
        # -----------------------------
        interior_nodes = []
        for j in range(1, nx_f - 1):
            for i in range(1, nx_f - 1):
                interior_nodes.append(i + nx_f * j)
        interior_nodes = np.array(interior_nodes, dtype=np.int32)

        # unknown dofs ordered by interior node (contiguous blocks)
        unknown_dofs = np.empty(3 * interior_nodes.size, dtype=np.int32)
        k = 0
        for n in interior_nodes:
            base = 3 * n
            unknown_dofs[k:k+3] = (base, base + 1, base + 2)
            k += 3

        # -----------------------------
        # element reference coords
        # -----------------------------
        dx_f = length / nxe_f
        x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
        y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

        dx_c = length / nxe_c
        x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
        y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

        # -----------------------------
        # Build G_f and G_c as SPARSE (COO -> CSR)
        # Each element contributes 4 rows x 12 cols.
        # Row ordering: [gx(0,-b), gx(0,+b), gy(-a,0), gy(+a,0)] per element.
        # -----------------------------
        def build_G(nxe, nx, x_loc, y_loc, N_total):
            ne = nxe * nxe
            rows = []
            cols = []
            data = []

            for ey in range(nxe):
                for ex in range(nxe):
                    e = ex + ey * nxe

                    loc_nodes = np.array([
                        ex + nx * ey,
                        (ex + 1) + nx * ey,
                        (ex + 1) + nx * (ey + 1),
                        ex + nx * (ey + 1),
                    ], dtype=np.int32)
                    loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)],
                                    dtype=np.int32)

                    # tying strain rows (length 12 each)
                    _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_loc, y_loc)
                    _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_loc, y_loc)
                    _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_loc, y_loc)
                    _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_loc, y_loc)

                    B = np.vstack([bx_m, bx_p, by_m, by_p])  # (4,12)

                    r0 = 4 * e
                    for rr in range(4):
                        r = r0 + rr
                        # add 12 nnz
                        rows.extend([r] * 12)
                        cols.extend(loc_dof.tolist())
                        data.extend(B[rr, :].tolist())

            G = sp.coo_matrix((np.array(data), (np.array(rows), np.array(cols))),
                            shape=(4 * ne, N_total)).tocsr()
            return G

        G_f = build_G(nxe_f, nx_f, x_f, y_f, N_f)  # (4*nelems_f, N_f)
        G_c = build_G(nxe_c, nx_c, x_c, y_c, N_c)  # (4*nelems_c, N_c)

        # -----------------------------
        # Build RHS_strain = P_gam @ G_c WITHOUT forming P_gam
        # Using your bilinear weights per fine element, applied per component row.
        #
        # RHS_strain shape: (4*nelems_f, N_c) sparse
        # -----------------------------
        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        # We'll assemble RHS_strain in COO by accumulating scaled copies of G_c rows
        rhs_rows = []
        rhs_cols = []
        rhs_data = []

        Gc_csr = G_c.tocsr()

        for ey in range(nxe_f):
            for ex in range(nxe_f):
                ef = ex + ey * nxe_f

                # fine elem center mapped to coarse index space (your mapping)
                x = 0.5 * (ex + 0.5) - 0.5
                y = 0.5 * (ey + 0.5) - 0.5

                i0 = int(np.floor(x))
                j0 = int(np.floor(y))
                tx = x - i0
                ty = y - j0

                i0 = clamp(i0, 0, nxe_c - 1)
                j0 = clamp(j0, 0, nxe_c - 1)
                i1 = clamp(i0 + 1, 0, nxe_c - 1)
                j1 = clamp(j0 + 1, 0, nxe_c - 1)

                if i1 == i0: tx = 0.0
                if j1 == j0: ty = 0.0

                w00 = (1.0 - tx) * (1.0 - ty)
                w10 = (tx)       * (1.0 - ty)
                w01 = (1.0 - tx) * (ty)
                w11 = (tx)       * (ty)

                e00 = i0 + j0 * nxe_c
                e10 = i1 + j0 * nxe_c
                e01 = i0 + j1 * nxe_c
                e11 = i1 + j1 * nxe_c

                # For each comp, RHS row = sum_k w_k * (that same comp row of coarse element)
                for comp in range(4):
                    rf = 4 * ef + comp

                    for ec, w in ((e00, w00), (e10, w10), (e01, w01), (e11, w11)):
                        if w == 0.0:
                            continue
                        rc = 4 * ec + comp

                        # add w * G_c[rc, :]
                        start = Gc_csr.indptr[rc]
                        end   = Gc_csr.indptr[rc + 1]
                        cols = Gc_csr.indices[start:end]
                        vals = Gc_csr.data[start:end]

                        rhs_rows.extend([rf] * cols.size)
                        rhs_cols.extend(cols.tolist())
                        rhs_data.extend((w * vals).tolist())

        RHS_strain = sp.coo_matrix((np.array(rhs_data),
                                (np.array(rhs_rows), np.array(rhs_cols))),
                                shape=(4 * nelems_f, N_c)).tocsr()

        # -----------------------------
        # Reduced system on unknown fine dofs, free coarse cols:
        # A = G_f[:, unknown_dofs]
        # B = RHS_strain[:, free_cols_c]
        # X0 = P0[unknown_dofs, free_cols_c]
        # -----------------------------
        # A = G_f[:, unknown_dofs].tocsr()
        # B = RHS_strain[:, free_cols_c].tocsr()

        # Reduced system on unknown fine dofs, free coarse cols
        A = G_f[:, unknown_dofs].tocsr()

        # boundary dofs are the complement of unknown_dofs
        all_f = np.arange(N_f, dtype=np.int32)
        is_unknown = np.zeros(N_f, dtype=bool)
        is_unknown[unknown_dofs] = True
        boundary_dofs = all_f[~is_unknown]

        # IMPORTANT: shift RHS by fixed boundary contribution
        P_B = P0[np.ix_(boundary_dofs, free_cols_c)]         # fixed boundary rows (dense)
        # B = (RHS_strain[:, free_cols_c] - (G_f[:, boundary_dofs] @ P_B)).tocsr()
        GB_PB = (G_f[:, boundary_dofs] @ P_B)
        GB_PB = np.asarray(GB_PB)   # <-- important

        B = RHS_strain[:, free_cols_c] - GB_PB
        B = sp.csr_matrix(B)


        X = P0[np.ix_(unknown_dofs, free_cols_c)].copy()  # (n_unknown, n_free)
        P0_free = X.copy()

        # -----------------------------
        # Build sparse normal matrix M and dense RHS (same pattern as beam)
        # -----------------------------
        n_unknown = unknown_dofs.size
        I = sp.eye(n_unknown, format="csr")
        M = (A.T @ A) + lam * I            # sparse SPD
        RHS = (A.T @ B).toarray() + lam * P0_free  # dense (n_unknown, n_free)

        M = M.tocsr()

        # -----------------------------
        # Block structure: contiguous in our ordering (by interior node)
        # -----------------------------
        dofs_per_node = 3
        block_size = dofs_per_node * block_nodes
        n_blocks = int(np.ceil(n_unknown / block_size))

        def blk_slice(b):
            i0 = b * block_size
            i1 = min(n_unknown, (b + 1) * block_size)
            return slice(i0, i1)

        # Precompute diagonal block inverses
        Dinv = []
        for b in range(n_blocks):
            sl = blk_slice(b)
            Db = M[sl, sl].toarray()
            Db = 0.5 * (Db + Db.T)
            Dinv.append(np.linalg.inv(Db))

        # # -----------------------------
        # # Block Gauss–Seidel sweeps (same as beam)
        # # -----------------------------
        # import matplotlib.pyplot as plt 

        # spars = M @ RHS

        # for _ in range(n_sweeps):
        #     # print(f"sweep# {_}")
        #     # plt.spy(X)
        #     # plt.show()

        #     # plt.spy(RHS)
        #     # plt.show()

        #     for b in range(n_blocks):
        #         sl = blk_slice(b)

        #         Mb_all = M[sl, :]           # sparse
        #         r = RHS[sl, :] - Mb_all @ X
        #         r += (M[sl, sl] @ X[sl, :])  # add back diagonal

        #         dX = Dinv[b] @ r
        #         X[sl, :] = X[sl, :] + omega * (dX - X[sl, :])

        # # -----------------------------
        # # Assemble full P
        # # Start from baseline P0, zero fixed coarse cols, overwrite interior unknown rows
        # # -----------------------------
        # P = P0.copy()
        # P[:, fixed_cols_c] = 0.0
        # P[np.ix_(unknown_dofs, free_cols_c)] = X

        # self._lock_P_cache[nxe_c] = P.copy()
        # return P

                # -----------------------------
        # Build sparse normal matrix M and SPARSE RHS operator pieces
        # -----------------------------
        n_unknown = unknown_dofs.size
        I = sp.eye(n_unknown, format="csr")
        M = (A.T @ A) + lam * I            # sparse SPD
        M = M.tocsr()

        # Keep ATB sparse: (A.T @ B) is sparse (n_unknown x n_free)
        ATB = (A.T @ B).tocsr()

        # X is dense but we will project it each sweep to a fixed pattern
        X = P0[np.ix_(unknown_dofs, free_cols_c)].copy()  # dense (n_unknown, n_free)

        # -----------------------------
        # Block structure: contiguous in our ordering (by interior node)
        # -----------------------------
        dofs_per_node = 3
        block_size = dofs_per_node * block_nodes
        n_blocks = int(np.ceil(n_unknown / block_size))

        def blk_slice(b):
            i0 = b * block_size
            i1 = min(n_unknown, (b + 1) * block_size)
            return slice(i0, i1)

        # -----------------------------
        # Precompute diagonal block inverses
        # -----------------------------
        Dinv = []
        for b in range(n_blocks):
            sl = blk_slice(b)
            Db = M[sl, sl].toarray()
            Db = 0.5 * (Db + Db.T)
            Dinv.append(np.linalg.inv(Db))

        # -----------------------------
        # Build fixed sparsity masks from "one-level fill": S = M @ RHS
        #
        # Here RHS = ATB + lam*P0_free, but P0_free is dense; you likely want the
        # *locking-driven* pattern, so base it on M@ATB (and optionally union with P0 stencil).
        # -----------------------------

        # Pattern from locking constraints:
        S_lock = (M @ ATB).tocsr()   # sparse (n_unknown x n_free)

        # OPTIONAL: also allow baseline P0 stencil (keeps interpolation connectivity)
        # Convert P0_free to sparse pattern cheaply by thresholding exact zeros
        P0_free_dense = X.copy()
        P0_pat = sp.csr_matrix(P0_free_dense != 0.0)  # boolean-ish pattern
        # Union patterns (still sparse)
        # S_pat = (S_lock != 0) | (P0_pat != 0)
        S_pat = (S_lock.copy() != 0).astype(np.int8)
        S_pat = S_pat + (P0_pat.copy() != 0).astype(np.int8)
        S_pat.data[:] = 1        # force boolean pattern


        # Precompute per-block allowed column indices
        allowed_cols = []
        for b in range(n_blocks):
            sl = blk_slice(b)
            # columns that appear in any row of this block
            rows = np.arange(sl.start, sl.stop, dtype=np.int32)
            cols_b = set()
            for r in rows:
                start = S_pat.indptr[r]
                end   = S_pat.indptr[r + 1]
                cols_b.update(S_pat.indices[start:end].tolist())
            cols_b = np.array(sorted(cols_b), dtype=np.int32)
            allowed_cols.append(cols_b)

        # -----------------------------
        # Project X initially to allowed sparsity
        # -----------------------------
        for b in range(n_blocks):
            sl = blk_slice(b)
            keep = allowed_cols[b]
            if keep.size == 0:
                X[sl, :] = 0.0
                continue
            # zero everything then restore kept cols
            Xblk = X[sl, keep].copy()
            X[sl, :] = 0.0
            X[sl, keep] = Xblk

        # -----------------------------
        # Block Gauss–Seidel sweeps with projection
        # RHS_block is computed on the fly from sparse ATB + dense lam*P0
        # -----------------------------
        for _ in range(n_sweeps):
            for b in range(n_blocks):
                sl = blk_slice(b)
                keep = allowed_cols[b]

                # Compute RHS on the fly for this block-row
                # rhs = (ATB[sl,:]).toarray() + lam*P0_free[sl,:]
                rhs = ATB[sl, :].toarray() + lam * P0_free_dense[sl, :]

                # residual for GS: rhs - M[sl,:]@X + Mdiag*Xsl
                Mb_all = M[sl, :]  # sparse
                r = rhs - (Mb_all @ X)
                r += (M[sl, sl] @ X[sl, :])

                dX = Dinv[b] @ r
                Xnew = X[sl, :] + omega * (dX - X[sl, :])

                # Hard projection: enforce fixed sparsity per block
                if keep.size == 0:
                    X[sl, :] = 0.0
                else:
                    X[sl, :] = 0.0
                    X[sl, keep] = Xnew[:, keep]

        # -----------------------------
        # Assemble full P
        # -----------------------------
        P = P0.copy()
        P[:, fixed_cols_c] = 0.0
        P[np.ix_(unknown_dofs, free_cols_c)] = X

        # check and indeed it is a sparse version of P!
        # import matplotlib.pyplot as plt
        # plt.spy(P)
        # plt.show()

        self._lock_P_cache[nxe_c] = P.copy()
        return P

    def _locking_aware_prolong_local_mitc_v2_jacobi(self,
                                                nxe_c: int,
                                                length: float = 1.0,
                                                block_nodes: int = 1,
                                                n_sweeps: int = 10,
                                                omega: float = 1.5):
        """
        Local locking-aware prolongation (MITC tying strains) using BLOCK-JACOBI with
        fixed sparsity + ONLY sparse mat-mat products in the iteration (GPU-friendly).

        Solve approx on interior fine dofs only (boundary dofs held fixed):
            (A^T A + lam I) X = A^T B + lam P0

        Jacobi sweep (loop-free, SpMM-only):
            RHS = sparse_control(ATB + lam*P0_free)              (sparse)
            MX  = sparse_control(M @ X)                          (sparse SpMM)
            RES = sparse_control(RHS - MX)                       (sparse axpy + mask)
            X   = sparse_control(X + omega * (Dinv_op @ RES))    (SpMM + axpy + mask)

        Notes:
        - No Gauss–Seidel dependence (true Jacobi).
        - No per-block loop in the SWEEPS.
        - Fixed sparsity enforced via elementwise multiply with a CSR mask.
        """

        import numpy as np
        import scipy.sparse as sp

        if nxe_c in self._lock_P_cache:
            return self._lock_P_cache[nxe_c]

        if block_nodes not in (1, 2):
            raise ValueError("block_nodes must be 1 or 2")

        lam = float(self.lam)

        # -----------------------------
        # sizes
        # -----------------------------
        nxe_f = 2 * nxe_c

        nx_f = nxe_f + 1
        nnodes_f = nx_f**2
        nelems_f = nxe_f**2
        N_f = 3 * nnodes_f

        nx_c = nxe_c + 1
        nnodes_c = nx_c**2
        nelems_c = nxe_c**2
        N_c = 3 * nnodes_c

        # -----------------------------
        # baseline prolong (fine x coarse) + BCs
        # -----------------------------
        P0 = self._build_P2_uncoupled3(nxe_c)  # csr
        P0 = self._apply_bcs_to_P(P0, nxe_c)   # csr
        P0 = P0.toarray()                     # dense (N_f, N_c)

        # -----------------------------
        # coarse fixed columns
        # -----------------------------
        constrained_dofs = (0, 1, 2) if self.clamped else (0,)

        fixed_cols_c = []
        for inode in range(nnodes_c):
            i = inode % nx_c
            j = inode // nx_c
            if (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1):
                base = 3 * inode
                for a in constrained_dofs:
                    fixed_cols_c.append(base + a)
        fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=np.int32)

        all_cols_c = np.arange(N_c, dtype=np.int32)
        free_cols_c = np.setdiff1d(all_cols_c, fixed_cols_c, assume_unique=False)

        # -----------------------------
        # fine interior dofs as unknowns
        # -----------------------------
        interior_nodes = []
        for j in range(1, nx_f - 1):
            for i in range(1, nx_f - 1):
                interior_nodes.append(i + nx_f * j)
        interior_nodes = np.array(interior_nodes, dtype=np.int32)

        unknown_dofs = np.empty(3 * interior_nodes.size, dtype=np.int32)
        k = 0
        for n in interior_nodes:
            base = 3 * n
            unknown_dofs[k:k+3] = (base, base + 1, base + 2)
            k += 3

        # -----------------------------
        # element reference coords
        # -----------------------------
        dx_f = length / nxe_f
        x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
        y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

        dx_c = length / nxe_c
        x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
        y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

        # -----------------------------
        # Build G_f and G_c as SPARSE (COO -> CSR)
        # -----------------------------
        def build_G(nxe, nx, x_loc, y_loc, N_total):
            ne = nxe * nxe
            rows = []
            cols = []
            data = []

            for ey in range(nxe):
                for ex in range(nxe):
                    e = ex + ey * nxe

                    loc_nodes = np.array([
                        ex + nx * ey,
                        (ex + 1) + nx * ey,
                        (ex + 1) + nx * (ey + 1),
                        ex + nx * (ey + 1),
                    ], dtype=np.int32)
                    loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)],
                                    dtype=np.int32)

                    _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_loc, y_loc)
                    _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_loc, y_loc)
                    _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_loc, y_loc)
                    _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_loc, y_loc)

                    B = np.vstack([bx_m, bx_p, by_m, by_p])  # (4,12)

                    r0 = 4 * e
                    for rr in range(4):
                        r = r0 + rr
                        rows.extend([r] * 12)
                        cols.extend(loc_dof.tolist())
                        data.extend(B[rr, :].tolist())

            return sp.coo_matrix((np.array(data), (np.array(rows), np.array(cols))),
                                shape=(4 * ne, N_total)).tocsr()

        G_f = build_G(nxe_f, nx_f, x_f, y_f, N_f)  # (4*nelems_f, N_f)
        G_c = build_G(nxe_c, nx_c, x_c, y_c, N_c)  # (4*nelems_c, N_c)

        # -----------------------------
        # Build RHS_strain = P_gam @ G_c WITHOUT forming P_gam (assemble COO)
        # -----------------------------
        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        rhs_rows = []
        rhs_cols = []
        rhs_data = []
        Gc_csr = G_c.tocsr()

        for ey in range(nxe_f):
            for ex in range(nxe_f):
                ef = ex + ey * nxe_f

                x = 0.5 * (ex + 0.5) - 0.5
                y = 0.5 * (ey + 0.5) - 0.5

                i0 = int(np.floor(x))
                j0 = int(np.floor(y))
                tx = x - i0
                ty = y - j0

                i0 = clamp(i0, 0, nxe_c - 1)
                j0 = clamp(j0, 0, nxe_c - 1)
                i1 = clamp(i0 + 1, 0, nxe_c - 1)
                j1 = clamp(j0 + 1, 0, nxe_c - 1)

                if i1 == i0: tx = 0.0
                if j1 == j0: ty = 0.0

                w00 = (1.0 - tx) * (1.0 - ty)
                w10 = (tx)       * (1.0 - ty)
                w01 = (1.0 - tx) * (ty)
                w11 = (tx)       * (ty)

                e00 = i0 + j0 * nxe_c
                e10 = i1 + j0 * nxe_c
                e01 = i0 + j1 * nxe_c
                e11 = i1 + j1 * nxe_c

                for comp in range(4):
                    rf = 4 * ef + comp
                    for ec, w in ((e00, w00), (e10, w10), (e01, w01), (e11, w11)):
                        if w == 0.0:
                            continue
                        rc = 4 * ec + comp

                        start = Gc_csr.indptr[rc]
                        end   = Gc_csr.indptr[rc + 1]
                        cols = Gc_csr.indices[start:end]
                        vals = Gc_csr.data[start:end]

                        rhs_rows.extend([rf] * cols.size)
                        rhs_cols.extend(cols.tolist())
                        rhs_data.extend((w * vals).tolist())

        RHS_strain = sp.coo_matrix((np.array(rhs_data),
                                    (np.array(rhs_rows), np.array(rhs_cols))),
                                shape=(4 * nelems_f, N_c)).tocsr()

        # -----------------------------
        # Reduced system pieces: A, B (shifted by boundary contribution)
        # -----------------------------
        A = G_f[:, unknown_dofs].tocsr()

        all_f = np.arange(N_f, dtype=np.int32)
        is_unknown = np.zeros(N_f, dtype=bool)
        is_unknown[unknown_dofs] = True
        boundary_dofs = all_f[~is_unknown]

        P_B = P0[np.ix_(boundary_dofs, free_cols_c)]   # dense
        GB_PB = (G_f[:, boundary_dofs] @ P_B)
        GB_PB = np.asarray(GB_PB)

        B = RHS_strain[:, free_cols_c] - GB_PB
        B = sp.csr_matrix(B)

        # Baseline interior rows (dense)
        P0_free_dense = P0[np.ix_(unknown_dofs, free_cols_c)].copy()

        # -----------------------------
        # Build normal matrix M and ATB (sparse)
        # -----------------------------
        n_unknown = unknown_dofs.size
        I = sp.eye(n_unknown, format="csr")
        M = (A.T @ A) + lam * I
        M = M.tocsr()

        ATB = (A.T @ B).tocsr()  # sparse (n_unknown x n_free)

        # -----------------------------
        # Fixed sparsity mask (CSR) from locking-driven pattern (optionally union P0 stencil)
        # -----------------------------
        S_lock = (M @ ATB).tocsr()
        # S_lock = (M @ M @ ATB).tocsr()
        mask = (S_lock != 0).astype(np.int8)

        P0_pat = sp.csr_matrix(P0_free_dense != 0.0).astype(np.int8)
        mask = (mask + P0_pat).tocsr()
        mask.data[:] = 1  # boolean mask

        # # -----------------------------
        # # Build block-diagonal inverse operator Dinv_op ONCE (BSR)
        # # -----------------------------
        # dofs_per_node = 3
        # block_size = dofs_per_node * block_nodes
        # if (n_unknown % block_size) != 0:
        #     raise ValueError("For loop-free BSR Jacobi, n_unknown must be divisible by block_size")

        # n_blk = n_unknown // block_size

        # Mb = M.tobsr(blocksize=(block_size, block_size)).tocsr().tobsr(blocksize=(block_size, block_size))

        # diag_blocks = np.zeros((n_blk, block_size, block_size), dtype=float)

        # # one-time gather (not in sweeps)
        # for i in range(n_blk):
        #     start, end = Mb.indptr[i], Mb.indptr[i + 1]
        #     cols = Mb.indices[start:end]
        #     data = Mb.data[start:end]
        #     k = np.searchsorted(cols, i)
        #     if k >= cols.size or cols[k] != i:
        #         raise RuntimeError("Missing diagonal block in M (unexpected for SPD normal matrix).")
        #     Db = data[k]
        #     Db = 0.5 * (Db + Db.T)
        #     diag_blocks[i, :, :] = np.linalg.inv(Db)

        # Dinv_op = sp.bsr_matrix(
        #     (diag_blocks, (np.arange(n_blk), np.arange(n_blk))),
        #     shape=(n_unknown, n_unknown),
        #     blocksize=(block_size, block_size),
        # ).tocsr()

        # -----------------------------
        # Build block-diagonal inverse operator Dinv_op ONCE (BSR)  [FIXED]
        # -----------------------------
        dofs_per_node = 3
        block_size = dofs_per_node * block_nodes
        if (n_unknown % block_size) != 0:
            raise ValueError("For loop-free BSR Jacobi, n_unknown must be divisible by block_size")

        n_blk = n_unknown // block_size

        Mb = M.tobsr(blocksize=(block_size, block_size)).tocsr().tobsr(blocksize=(block_size, block_size))

        # diagonal blocks (n_blk, bs, bs)
        diag_blocks = np.zeros((n_blk, block_size, block_size), dtype=M.dtype)

        # one-time gather + invert
        for i in range(n_blk):
            start, end = Mb.indptr[i], Mb.indptr[i + 1]
            cols = Mb.indices[start:end]
            data = Mb.data[start:end]   # (nblocks_in_row, bs, bs)

            k = np.searchsorted(cols, i)
            if k >= cols.size or cols[k] != i:
                raise RuntimeError("Missing diagonal block in M (unexpected for SPD normal matrix).")

            Db = data[k]
            Db = 0.5 * (Db + Db.T)
            diag_blocks[i, :, :] = np.linalg.inv(Db)

        # Build a *diagonal* BSR using (data, indices, indptr) form
        Dinv_indptr  = np.arange(n_blk + 1, dtype=np.int32)   # one block per block-row
        Dinv_indices = np.arange(n_blk,     dtype=np.int32)   # diagonal column index
        Dinv_data    = diag_blocks                              # (nnz_blocks=n_blk, bs, bs)

        Dinv_op = sp.bsr_matrix(
            (Dinv_data, Dinv_indices, Dinv_indptr),
            shape=(n_unknown, n_unknown),
            blocksize=(block_size, block_size),
        ).tocsr()


        # -----------------------------
        # Fixed sparsity control (CSR mask multiply)
        # -----------------------------
        mask = mask.tocsr()
        def sparse_control(Xsp):
            return Xsp.multiply(mask)

        # RHS once (sparse)
        P0_free_sp = sp.csr_matrix(P0_free_dense)
        RHS = sparse_control((ATB + lam * P0_free_sp).tocsr())

        # Init X as sparse, projected to mask
        X = sparse_control(sp.csr_matrix(P0_free_dense))

        # -----------------------------
        # Loop-free per-sweep Jacobi: ONLY SpMM operations
        # -----------------------------
        for _ in range(n_sweeps):
            MX  = sparse_control((M @ X).tocsr())                 # SpMM
            RES = sparse_control((RHS - MX).tocsr())              # axpy + mask
            X   = sparse_control((X + omega * (Dinv_op @ RES)).tocsr())  # SpMM + axpy + mask

        # -----------------------------
        # Assemble full P
        # -----------------------------
        P = P0.copy()
        P[:, fixed_cols_c] = 0.0
        P[np.ix_(unknown_dofs, free_cols_c)] = X.toarray()

        self._lock_P_cache[nxe_c] = P.copy()
        return P

    def _locking_aware_prolong_local_mitc_v3_jacobi(
            self,
            nxe_c: int,
            length: float = 1.0,
            n_sweeps: int = 10,
            omega: float = 0.5,
            with_fillin: bool = False,
            use_mask: bool = True,
        ):
        """
        Locking-aware prolongation where constraints are on MITC tying strains:
        [ gx(0,-b), gx(0,+b), gy(-a,0), gy(+a,0) ] = 0   per element
        => 4 constraints per element.

        Same construction as _locking_aware_prolong_global_mitc_v1, but solves
        (A^T A + lam I) P = A^T B + lam P0
        using SpMM Jacobi (3x3 block diagonal) with optional fixed sparsity (mask).

        - with_fillin=True  : plain Jacobi (allows fill-in)
        - with_fillin=False : apply mask every sweep (keeps fixed sparsity)
        - use_mask=False    : equivalent to fill-in behavior (no masking)

        Mask choice: EXACTLY like your sandbox -> based on pattern of (M @ RHS).
        """

        import numpy as np
        import scipy.sparse as sp

        # cache with options so you can compare variants
        cache_key = ("local_mitc_v3_jacobi", int(nxe_c), float(length),
                    int(n_sweeps), float(omega), bool(with_fillin), bool(use_mask))
        if cache_key in self._lock_P_cache:
            return self._lock_P_cache[cache_key]

        # -----------------------------
        # same as v1 up to forming M, rhs
        # -----------------------------
        nxe_f = 2 * nxe_c

        nx_f = nxe_f + 1
        nnodes_f = nx_f**2
        nelems_f = nxe_f**2
        N_f = 3 * nnodes_f

        nx_c = nxe_c + 1
        nnodes_c = nx_c**2
        nelems_c = nxe_c**2
        N_c = 3 * nnodes_c

        dx_f = length / nxe_f
        x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
        y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

        dx_c = length / nxe_c
        x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
        y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

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

            _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_f, y_f)
            _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_f, y_f)
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

        # P_gam (dense) as in v1
        P_gam = np.zeros((4 * nelems_f, 4 * nelems_c), dtype=float)

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        for ielem_f in range(nelems_f):
            ex = ielem_f % nxe_f
            ey = ielem_f // nxe_f

            x = 0.5 * (ex + 0.5) - 0.5
            y = 0.5 * (ey + 0.5) - 0.5

            i0 = int(np.floor(x))
            j0 = int(np.floor(y))
            tx = x - i0
            ty = y - j0

            i0 = clamp(i0, 0, nxe_c - 1)
            j0 = clamp(j0, 0, nxe_c - 1)
            i1 = clamp(i0 + 1, 0, nxe_c - 1)
            j1 = clamp(j0 + 1, 0, nxe_c - 1)

            if i1 == i0: tx = 0.0
            if j1 == j0: ty = 0.0

            w00 = (1.0 - tx) * (1.0 - ty)
            w10 = (tx)       * (1.0 - ty)
            w01 = (1.0 - tx) * (ty)
            w11 = (tx)       * (ty)

            e00 = i0 + j0 * nxe_c
            e10 = i1 + j0 * nxe_c
            e01 = i0 + j1 * nxe_c
            e11 = i1 + j1 * nxe_c

            rf = 4 * ielem_f
            for comp in range(4):
                P_gam[rf + comp, 4 * e00 + comp] += w00
                P_gam[rf + comp, 4 * e10 + comp] += w10
                P_gam[rf + comp, 4 * e01 + comp] += w01
                P_gam[rf + comp, 4 * e11 + comp] += w11

        RHS = P_gam @ G_c  # dense (4*nelems_f, 3*nnodes_c)

        # baseline P0 (csr)
        P_0 = self._build_P2_uncoupled3(nxe_c)
        P_0 = self._apply_bcs_to_P(P_0, nxe_c)
        lam = float(self.lam)

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
        Esel[np.arange(nE), fixed_rows_f] = 1.0

        A = G_f[:, solve_rows_f]
        B = RHS[:, free_cols_c]

        A_aug = np.vstack([A, Esel])
        B_aug = np.vstack([B, np.zeros((nE, B.shape[1]))])

        idx0 = np.ix_(solve_rows_f, free_cols_c)
        P0_free = P_0[idx0].toarray()

        M   = A_aug.T @ A_aug + lam * np.eye(solve_rows_f.size)
        rhs = A_aug.T @ B_aug + lam * P0_free

        # -----------------------------
        # SpMM Jacobi solve (your sandbox logic, but inside method)
        # -----------------------------
        # keep these states for sandbox debugging
        self.G_f = G_f
        self.G_c = G_c
        self.P_gam = P_gam
        self.P_0 = P_0
        self.M = M
        self.RHS = rhs
        self.free_cols_c = free_cols_c
        self.fixed_cols_c = fixed_cols_c
        self.solve_rows_f = solve_rows_f
        self.P_0_free = P0_free

        # Convert M to CSR once (SpMM-friendly)
        LHS = sp.csr_matrix(M)

        # RHS and initial guess as CSR too (so LHS @ X stays sparse)
        RHS_csr = sp.csr_matrix(rhs)
        X = sp.csr_matrix(P0_free)

        # mask pattern EXACTLY from (M @ RHS) like your script
        mask = None
        if (not with_fillin) and use_mask:
            S_lock = (LHS @ RHS_csr)         # sparse
            # S_lock = (LHS @ LHS @ RHS_csr) # extra fillin not helpful
            mask = (S_lock != 0).astype(np.int8)  # sparse 0/1 CSR

        def sparse_control(Z):
            if mask is None:
                return Z
            # force sparse then elementwise mask
            if not sp.issparse(Z):
                Z = sp.csr_matrix(Z)
            return Z.multiply(mask)

        # block diagonal inverse as sparse operator (3x3 nodal blocks)
        block_size = 3
        n_unknown = X.shape[0]
        assert n_unknown % block_size == 0
        n_blk = n_unknown // block_size

        Mb = LHS.tobsr(blocksize=(block_size, block_size))

        diag_blocks = np.zeros((n_blk, block_size, block_size), dtype=float)
        for i in range(n_blk):
            start, end = Mb.indptr[i], Mb.indptr[i + 1]
            cols = Mb.indices[start:end]
            data = Mb.data[start:end]   # (nblocks_in_row, bs, bs)

            k = np.searchsorted(cols, i)
            if k >= cols.size or cols[k] != i:
                raise RuntimeError("Missing diagonal block in M (unexpected).")

            Db = data[k]
            Db = 0.5 * (Db + Db.T)
            diag_blocks[i, :, :] = np.linalg.inv(Db)

        Dinv_indptr  = np.arange(n_blk + 1, dtype=np.int32)
        Dinv_indices = np.arange(n_blk,     dtype=np.int32)
        Dinv_data    = diag_blocks

        Dinv_op = sp.bsr_matrix(
            (Dinv_data, Dinv_indices, Dinv_indptr),
            shape=(n_unknown, n_unknown),
            blocksize=(block_size, block_size),
        ).tocsr()

        # project initial guess
        X = sparse_control(X)

        # Jacobi sweeps: ONLY SpMM operations + axpy (+ mask when requested)
        if with_fillin or (mask is None):
            for _ in range(int(n_sweeps)):
                MX  = LHS @ X
                RES = RHS_csr - MX
                X   = X + float(omega) * (Dinv_op @ RES)
        else:
            for _ in range(int(n_sweeps)):
                MX  = sparse_control(LHS @ X)
                RES = sparse_control(RHS_csr - MX)
                X   = sparse_control(X + float(omega) * (Dinv_op @ RES))

        P_free = X.toarray()  # back to dense for assembly like v1

        # -----------------------------
        # Assemble full P (dense)
        # -----------------------------
        P = P_0.toarray()
        P[:, fixed_cols_c] = 0.0
        P[np.ix_(solve_rows_f, free_cols_c)] = P_free

        # looks like reasonable sparsity..
        # import matplotlib.pyplot as plt
        # plt.spy(P)
        # plt.show()

        self._lock_P_cache[cache_key] = P.copy()
        return P
    
    def _locking_aware_prolong_local_mitc_v4_jacobi(
            self,
            nxe_c: int,
            length: float = 1.0,
            n_sweeps: int = 10,
            omega: float = 0.5,
            with_fillin: bool = False,
            use_mask: bool = True,
        ):
        """
        Locking-aware prolongation where constraints are on MITC tying strains:
        [ gx(0,-b), gx(0,+b), gy(-a,0), gy(+a,0) ] = 0   per element
        => 4 constraints per element.

        Solve (G_f^T G_f + lam I) P = G_f^T (P_gam G_c) + lam P0
        using SpMM Jacobi (3x3 block diagonal) with optional fixed sparsity (mask).

        NOTE : IMPORTANT CHANGE (GPU-realistic BC handling):
        - No extra equations to enforce fine BCs (no Esel / augmentation).
        - Fine BCs are imposed by modifying rows+cols of M like a stiffness matrix:
                for i in fixed_rows_f:
                    M[i,:]=0, M[:,i]=0, M[i,i]=1, rhs[i,:]=P0[i,:]
        - Coarse BCs are imposed by zeroing the corresponding columns of (P_gam G_c)
            before forming rhs (and P0 already has coarse BCs applied).

        - with_fillin=True  : plain Jacobi (allows fill-in)
        - with_fillin=False : apply mask every sweep (keeps fixed sparsity)
        - use_mask=False    : equivalent to fill-in behavior (no masking)

        Mask choice: EXACTLY like your sandbox -> based on pattern of (M @ RHS).
        """

        import numpy as np
        import scipy.sparse as sp

        cache_key = ("local_mitc_v4_jacobi_bc_elim", int(nxe_c), float(length),
                    int(n_sweeps), float(omega), bool(with_fillin), bool(use_mask))
        if cache_key in self._lock_P_cache:
            return self._lock_P_cache[cache_key]

        # -----------------------------
        # same as v1 up to forming G_f, G_c, P_gam, RHS_full, P0
        # -----------------------------
        nxe_f = 2 * nxe_c

        nx_f = nxe_f + 1
        nnodes_f = nx_f**2
        nelems_f = nxe_f**2
        N_f = 3 * nnodes_f

        nx_c = nxe_c + 1
        nnodes_c = nx_c**2
        nelems_c = nxe_c**2
        N_c = 3 * nnodes_c

        dx_f = length / nxe_f
        x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
        y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

        dx_c = length / nxe_c
        x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
        y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

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

            _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_f, y_f)
            _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_f, y_f)
            _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_f, y_f)
            _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_f, y_f)

            if self.debug: 
                print(f"{bx_m=}\n{bx_p=}\n{by_m=}\n{by_p=}")

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

        # P_gam (dense) as in v1
        P_gam = np.zeros((4 * nelems_f, 4 * nelems_c), dtype=float)

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        for ielem_f in range(nelems_f):
            ex = ielem_f % nxe_f
            ey = ielem_f // nxe_f

            x = 0.5 * (ex + 0.5) - 0.5
            y = 0.5 * (ey + 0.5) - 0.5

            i0 = int(np.floor(x))
            j0 = int(np.floor(y))
            tx = x - i0
            ty = y - j0

            i0 = clamp(i0, 0, nxe_c - 1)
            j0 = clamp(j0, 0, nxe_c - 1)
            i1 = clamp(i0 + 1, 0, nxe_c - 1)
            j1 = clamp(j0 + 1, 0, nxe_c - 1)

            if i1 == i0: tx = 0.0
            if j1 == j0: ty = 0.0

            w00 = (1.0 - tx) * (1.0 - ty)
            w10 = (tx)       * (1.0 - ty)
            w01 = (1.0 - tx) * (ty)
            w11 = (tx)       * (ty)

            e00 = i0 + j0 * nxe_c
            e10 = i1 + j0 * nxe_c
            e01 = i0 + j1 * nxe_c
            e11 = i1 + j1 * nxe_c

            rf = 4 * ielem_f
            for comp in range(4):
                P_gam[rf + comp, 4 * e00 + comp] += w00
                P_gam[rf + comp, 4 * e10 + comp] += w10
                P_gam[rf + comp, 4 * e01 + comp] += w01
                P_gam[rf + comp, 4 * e11 + comp] += w11

        RHS_full = P_gam @ G_c  # dense (4*nelems_f, 3*nnodes_c)

        # baseline P0 (csr) already has BCs applied correctly
        P_0 = self._build_P2_uncoupled3(nxe_c)
        P_0 = self._apply_bcs_to_P(P_0, nxe_c)

        lam = float(self.lam)
        constrained_dofs = (0, 1, 2) if self.clamped else (0,)

        # coarse constrained cols
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

        # fine constrained rows
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

        # -----------------------------
        # NEW: impose coarse BCs by zeroing constrained columns of RHS term
        # (so G_f^T (P_gam G_c) respects coarse BCs column-wise)
        # -----------------------------
        RHS_full[:, fixed_cols_c] = 0.0

        # Build normal equations without augmentation
        A = G_f[:, solve_rows_f]                      # (4*nelems_f, N_f)
        B = RHS_full[:, free_cols_c]                  # (4*nelems_f, nfree_c)

        P0_free = P_0[np.ix_(solve_rows_f, free_cols_c)].toarray()

        M   = A.T @ A + lam * np.eye(solve_rows_f.size)
        rhs = A.T @ B + lam * P0_free

        # -----------------------------
        # NEW: enforce fine BCs by row/col elimination on M, and set rhs row to P0
        # (no extra constraint equations)
        # -----------------------------
        if fixed_rows_f.size > 0:
            M[fixed_rows_f, :] = 0.0
            M[:, fixed_rows_f] = 0.0
            for d in fixed_rows_f:
                M[d, d] = 1.0
            # np.fill_diagonal(M[np.ix_(fixed_rows_f, fixed_rows_f)], 1.0)

            # M[fixed_rows_f, :] = 0.0
            # M[:, fixed_rows_f] = 0.0
            # # M[fixed_rows_f, fixed_rows_f] = 1.0   # diagonal entries only, this doesn't work can't index two at once like that..
            # M[fixed_rows_f, fixed_rows_f] = 0.0           # clear the (cartesian) block
            # np.fill_diagonal(M[np.ix_(fixed_rows_f, fixed_rows_f)], 1.0)

            rhs[fixed_rows_f, :] = P0_free[fixed_rows_f, :]

        # import matplotlib.pyplot as plt
        # plt.imshow(M[:6,:6])
        # plt.show()

        # -----------------------------
        # SpMM Jacobi solve (same as before)
        # -----------------------------
        self.G_f = G_f
        self.G_c = G_c
        self.P_gam = P_gam
        self.P_0 = P_0
        self.M = M
        self.RHS = rhs
        self.free_cols_c = free_cols_c
        self.fixed_cols_c = fixed_cols_c
        self.solve_rows_f = solve_rows_f
        self.fixed_rows_f = fixed_rows_f
        self.P_0_free = P0_free

        LHS = sp.csr_matrix(M)
        RHS_csr = sp.csr_matrix(rhs)
        X = sp.csr_matrix(P0_free)

        # mask_mode = 1
        mask_mode = 2

        if mask_mode == 1:
            mask = None
            if (not with_fillin) and use_mask:
                S_lock = (LHS @ RHS_csr)
                # mask = (S_lock != 0).astype(np.int8)
                tol = 1e-24  # or something consistent with your scale
                mask = (abs(S_lock) > tol).astype(np.int8)

        elif mask_mode == 2:
            # for a,b = pm 1 of edges (need to be more careful part of sparsity isn't dropped out)
            def _structural_mask(A: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
                """
                Return structural pattern of A@B (no numeric cancellation):
                    mask_ij = 1 iff exists k with A_ik != 0 and B_kj != 0
                """
                # Make 0/1 matrices with same sparsity (values don't matter)
                A1 = A.copy()
                if A1.nnz:
                    A1.data[:] = 1.0
                B1 = B.copy()
                if B1.nnz:
                    B1.data[:] = 1.0

                S = A1 @ B1                 # numeric product but cannot cancel to zero structurally
                S.eliminate_zeros()          # just in case
                return (S != 0).astype(np.int8)

            mask = None
            if (not with_fillin) and use_mask:
                # 1) Always keep original P0 sparsity (so entries like (0,6) never drop)
                P0_pat = (sp.csr_matrix(P0_free) != 0).astype(np.int8)

                # 2) Add one-level structural fill (your "M@RHS" idea, but structural not numeric)
                MR_pat = _structural_mask(LHS, RHS_csr)

                # 3) Final mask is union
                mask = (P0_pat + MR_pat)
                mask.data[:] = 1  # turn any 1/2/... into 1
                mask.eliminate_zeros()

        def sparse_control(Z):
            if mask is None:
                return Z
            if not sp.issparse(Z):
                Z = sp.csr_matrix(Z)
            return Z.multiply(mask)

        block_size = 3
        n_unknown = X.shape[0]
        assert n_unknown % block_size == 0
        n_blk = n_unknown // block_size

        Mb = LHS.tobsr(blocksize=(block_size, block_size))
        self.Mb = Mb

        diag_blocks = np.zeros((n_blk, block_size, block_size), dtype=float)
        for i in range(n_blk):
            start, end = Mb.indptr[i], Mb.indptr[i + 1]
            cols = Mb.indices[start:end]
            data = Mb.data[start:end]
            # print(f"diag block {i=} : {cols=}")

            # k = np.searchsorted(cols, i)
            # if k >= cols.size or cols[k] != i:
            #     raise RuntimeError("Missing diagonal block in M (unexpected).")

            # Db = data[k]

            # find diagonal block index in this BSR row (cols is not guaranteed sorted)
            matches = np.where(cols == i)[0]
            if matches.size == 0:
                raise RuntimeError("Missing diagonal block in M (unexpected).")
            k = matches[0]
            Db = data[k]

            Db = 0.5 * (Db + Db.T)
            # print(f"{Db=}")
            diag_blocks[i, :, :] = np.linalg.inv(Db)

        Dinv_indptr  = np.arange(n_blk + 1, dtype=np.int32)
        Dinv_indices = np.arange(n_blk,     dtype=np.int32)
        Dinv_data    = diag_blocks

        Dinv_op = sp.bsr_matrix(
            (Dinv_data, Dinv_indices, Dinv_indptr),
            shape=(n_unknown, n_unknown),
            blocksize=(block_size, block_size),
        ).tocsr()
        
        # debug_print_bsr_matrix(Mb, name="locking-kmat")

        X = sparse_control(X)

        if with_fillin or (mask is None):
            for _ in range(int(n_sweeps)):
                MX  = LHS @ X
                RES = RHS_csr - MX
                X   = X + float(omega) * (Dinv_op @ RES)
        else:
            for _ in range(int(n_sweeps)):
                MX  = sparse_control(LHS @ X)
                RES = sparse_control(RHS_csr - MX)
                X   = sparse_control(X + float(omega) * (Dinv_op @ RES))

        P_free = X.toarray()

        # -----------------------------
        # Assemble full P (dense)
        # -----------------------------
        P = P_0.toarray()
        P[:, fixed_cols_c] = 0.0
        P[np.ix_(solve_rows_f, free_cols_c)] = P_free

        self.P = P

        self._lock_P_cache[cache_key] = P.copy()
        return P
    
    def _locking_aware_prolong_local_mitc_v5_jacobi(
            self,
            nxe_c: int,
            length: float = 1.0,
            n_sweeps: int = 10,
            omega: float = 0.5,
            with_fillin: bool = True,
            use_mask: bool = True,
        ):
        """
        v5 change (your requirement):
        - DO NOT eliminate coarse BC DOFs from the solve (no free_cols_c).
        - Keep RHS_csr and X with ncols = N_c = 3*nnodes_c (multiple of 3).
        - Impose coarse BCs by zeroing constrained coarse columns in:
                (P_gam @ G_c) term  AND  P0 rows (i.e., P0[:, fixed_cols_c]=0)
            so the solve is consistent while keeping full 3-dof node blocks.
        - Impose fine BCs via row/col elimination on M and rhs row reset to P0, as before.

        Solve (G_f^T G_f + lam I) P = G_f^T (P_gam G_c) + lam P0
        using SpMM Jacobi (3x3 block diagonal) with optional fixed sparsity (mask).
        """

        import numpy as np
        import scipy.sparse as sp

        cache_key = ("local_mitc_v5_jacobi_no_coarse_elim",
                    int(nxe_c), float(length), int(n_sweeps),
                    float(omega), bool(with_fillin), bool(use_mask))
        if cache_key in self._lock_P_cache:
            return self._lock_P_cache[cache_key]

        # -----------------------------
        # same as v1-v4 up to forming G_f, G_c, P_gam, RHS_full, P0
        # -----------------------------
        nxe_f = 2 * nxe_c

        nx_f = nxe_f + 1
        nnodes_f = nx_f**2
        nelems_f = nxe_f**2
        N_f = 3 * nnodes_f

        nx_c = nxe_c + 1
        nnodes_c = nx_c**2
        nelems_c = nxe_c**2
        N_c = 3 * nnodes_c

        dx_f = length / nxe_f
        x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
        y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

        dx_c = length / nxe_c
        x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
        y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

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

            _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_f, y_f)
            _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_f, y_f)
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

        # # P_gam (dense) as in v1
        # P_gam = np.zeros((4 * nelems_f, 4 * nelems_c), dtype=float)

        # def clamp(v, lo, hi):
        #     return max(lo, min(hi, v))

        # for ielem_f in range(nelems_f):
        #     ex = ielem_f % nxe_f
        #     ey = ielem_f // nxe_f

        #     x = 0.5 * (ex + 0.5) - 0.5
        #     y = 0.5 * (ey + 0.5) - 0.5

        #     i0 = int(np.floor(x))
        #     j0 = int(np.floor(y))
        #     tx = x - i0
        #     ty = y - j0

        #     i0 = clamp(i0, 0, nxe_c - 1)
        #     j0 = clamp(j0, 0, nxe_c - 1)
        #     i1 = clamp(i0 + 1, 0, nxe_c - 1)
        #     j1 = clamp(j0 + 1, 0, nxe_c - 1)

        #     if i1 == i0: tx = 0.0
        #     if j1 == j0: ty = 0.0

        #     w00 = (1.0 - tx) * (1.0 - ty)
        #     w10 = (tx)       * (1.0 - ty)
        #     w01 = (1.0 - tx) * (ty)
        #     w11 = (tx)       * (ty)

        #     e00 = i0 + j0 * nxe_c
        #     e10 = i1 + j0 * nxe_c
        #     e01 = i0 + j1 * nxe_c
        #     e11 = i1 + j1 * nxe_c

        #     rf = 4 * ielem_f
        #     for comp in range(4):
        #         P_gam[rf + comp, 4 * e00 + comp] += w00
        #         P_gam[rf + comp, 4 * e10 + comp] += w10
        #         P_gam[rf + comp, 4 * e01 + comp] += w01
        #         P_gam[rf + comp, 4 * e11 + comp] += w11

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

        RHS_full = P_gam @ G_c  # dense (4*nelems_f, 3*nnodes_c)

        # baseline P0 (csr) already has BCs applied correctly
        P_0 = self._build_P2_uncoupled3(nxe_c)
        P_0 = self._apply_bcs_to_P(P_0, nxe_c)

        lam = float(self.lam)
        constrained_dofs = (0, 1, 2) if self.clamped else (0,)

        # coarse constrained cols (still computed, but NOT eliminated from solve)
        fixed_cols_c = []
        for inode in range(nnodes_c):
            i = inode % nx_c
            j = inode // nx_c
            if (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1):
                base = 3 * inode
                for a in constrained_dofs:
                    fixed_cols_c.append(base + a)
        fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=int)

        # fine constrained rows
        fixed_rows_f = []
        for inode in range(nnodes_f):
            i = inode % nx_f
            j = inode // nx_f
            if (i == 0) or (i == nx_f - 1) or (j == 0) or (j == nx_f - 1):
                base = 3 * inode
                for a in constrained_dofs:
                    fixed_rows_f.append(base + a)
        fixed_rows_f = np.array(sorted(set(fixed_rows_f)), dtype=int)

        solve_rows_f = np.arange(N_f, dtype=int)  # keep all rows

        # -----------------------------
        # v5: impose coarse BCs by zeroing constrained coarse columns
        # in BOTH RHS term and P0 (so system has N_c cols but BC cols are forced to 0)
        # -----------------------------
        if fixed_cols_c.size > 0:
            RHS_full[:, fixed_cols_c] = 0.0
            # also guarantee P0 has those cols zero (in case _apply_bcs_to_P doesn't)
            P_0 = P_0.tolil(copy=True)
            P_0[:, fixed_cols_c] = 0.0
            P_0 = P_0.tocsr()

        # Build normal equations without augmentation (FULL columns)
        A = G_f[:, solve_rows_f]           # (4*nelems_f, N_f)
        B = RHS_full                       # (4*nelems_f, N_c)
        P0_all = P_0[solve_rows_f, :].toarray()  # (N_f, N_c), cols multiple of 3

        M   = A.T @ A + lam * np.eye(solve_rows_f.size)
        rhs = A.T @ B + lam * P0_all

        # -----------------------------
        # enforce fine BCs by row/col elimination on M, set rhs row to P0 row (ALL cols)
        # -----------------------------
        if fixed_rows_f.size > 0:
            M[fixed_rows_f, :] = 0.0
            M[:, fixed_rows_f] = 0.0
            for d in fixed_rows_f:
                M[d, d] = 1.0
            rhs[fixed_rows_f, :] = P0_all[fixed_rows_f, :]

        # -----------------------------
        # SpMM Jacobi solve (same structure, but X/RHS have N_c cols)
        # -----------------------------
        LHS = sp.csr_matrix(M)
        RHS_csr = sp.csr_matrix(rhs)
        X = sp.csr_matrix(P0_all)  # initial guess

        mask_mode = 1
        # mask_mode = 2

        if mask_mode == 1:
            mask = None
            if (not with_fillin) and use_mask:
                S_lock = (LHS @ RHS_csr)
                tol = 1e-24
                mask = (abs(S_lock) > tol).astype(np.int8)

        elif mask_mode == 2:
            def _structural_mask(A: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
                """Structural pattern of A@B (no numeric cancellation)."""
                A1 = A.copy()
                if A1.nnz:
                    A1.data[:] = 1.0
                B1 = B.copy()
                if B1.nnz:
                    B1.data[:] = 1.0
                S = A1 @ B1
                S.eliminate_zeros()
                return (S != 0).astype(np.int8)

            mask = None
            if (not with_fillin) and use_mask:
                P0_pat = (sp.csr_matrix(P0_all) != 0).astype(np.int8)
                MR_pat = _structural_mask(LHS, RHS_csr)
                mask = (P0_pat + MR_pat)
                mask.data[:] = 1
                mask.eliminate_zeros()

        def sparse_control(Z):
            if mask is None:
                return Z
            if not sp.issparse(Z):
                Z = sp.csr_matrix(Z)
            return Z.multiply(mask)

        block_size = 3
        n_unknown = X.shape[0]
        assert n_unknown % block_size == 0
        n_blk = n_unknown // block_size

        Mb = LHS.tobsr(blocksize=(block_size, block_size))

        diag_blocks = np.zeros((n_blk, block_size, block_size), dtype=float)
        for i in range(n_blk):
            start, end = Mb.indptr[i], Mb.indptr[i + 1]
            cols = Mb.indices[start:end]
            data = Mb.data[start:end]
            matches = np.where(cols == i)[0]
            if matches.size == 0:
                raise RuntimeError("Missing diagonal block in M (unexpected).")
            Db = data[matches[0]]
            Db = 0.5 * (Db + Db.T)
            diag_blocks[i, :, :] = np.linalg.inv(Db)

        Dinv_indptr  = np.arange(n_blk + 1, dtype=np.int32)
        Dinv_indices = np.arange(n_blk,     dtype=np.int32)
        Dinv_data    = diag_blocks

        Dinv_op = sp.bsr_matrix(
            (Dinv_data, Dinv_indices, Dinv_indptr),
            shape=(n_unknown, n_unknown),
            blocksize=(block_size, block_size),
        ).tocsr()

        X = sparse_control(X)

        if with_fillin or (mask is None):
            for _ in range(int(n_sweeps)):
                MX  = LHS @ X
                RES = RHS_csr - MX
                X   = X + float(omega) * (Dinv_op @ RES)
        else:
            for _ in range(int(n_sweeps)):
                # LHS /= 64.0; RHS_csr /= 32.0; Dinv_op *= 64.0 # DEBUG
                MX  = sparse_control(LHS @ X) 
                RES = sparse_control(RHS_csr - MX)
                X   = sparse_control(X + float(omega) * (Dinv_op @ RES))

        P_sol = X.toarray()  # (N_f, N_c)

        # -----------------------------
        # Assemble full P (dense)
        # -----------------------------
        P = P_0.toarray()                 # already has coarse BC columns zeroed (we enforced above)
        if fixed_cols_c.size > 0:
            P[:, fixed_cols_c] = 0.0      # safety
        P[solve_rows_f, :] = P_sol        # overwrite all rows/cols (including coarse BC cols which remain 0)

        # stash debug
        self.G_f = G_f
        self.G_c = G_c
        self.P_gam = P_gam
        self.P_0 = P_0
        self.M = M
        self.RHS = rhs
        self.fixed_cols_c = fixed_cols_c
        self.solve_rows_f = solve_rows_f
        self.fixed_rows_f = fixed_rows_f
        self.P = P

        self._lock_P_cache[cache_key] = P.copy()
        return P


    def _energy_smooth_jacobi_v1(
        self,
        nxe_c: int,
        length: float = 1.0,
        n_sweeps: int = 10,
        omega: float = 0.7,
        with_fillin: bool = True,
        use_mask: bool = True,
    ):
        """
        Standard K-matrix energy smoothing in Jacobi-preconditioned space:
            P <- P - omega * D^{-1} (K P)

        - K is taken from: self._kmat_cache[nxe_c]   (you provide it)
        - Optional fixed sparsity: mask = pattern(K@P) computed once initially.
        - Cache key is ONLY nxe_c (simple).
        """

        import numpy as np
        import scipy.sparse as sp

        # simple cache: ONLY keyed by nxe_c
        cache_key = ("energy_smooth_jacobi_v1", int(nxe_c))
        if cache_key in self._lock_P_cache:
            return self._lock_P_cache[cache_key]

        # baseline P0
        P0 = self._build_P2_uncoupled3(nxe_c)
        P0 = self._apply_bcs_to_P(P0, nxe_c)
        P = P0.tocsr()

        # kmat from cache
        if not hasattr(self, "_kmat_cache") or (int(nxe_c) not in self._kmat_cache):
            raise RuntimeError("Expected self._kmat_cache[nxe_c] to exist for energy smoothing.")
        K = self._kmat_cache[int(nxe_c)]
        K = K.tocsr() if sp.isspmatrix(K) else sp.csr_matrix(K)

        # Jacobi block inverse (3x3 nodal blocks)
        bs = 3
        N = K.shape[0]
        if (N % bs) != 0 or K.shape[1] != N:
            raise ValueError(f"kmat must be square with size multiple of 3, got {K.shape}")
        nblk = N // bs

        Kb = K.tobsr(blocksize=(bs, bs))
        diag_inv = np.zeros((nblk, bs, bs), dtype=float)
        for i in range(nblk):
            s, e = Kb.indptr[i], Kb.indptr[i + 1]
            cols = Kb.indices[s:e]
            data = Kb.data[s:e]  # (nblocks_in_row, bs, bs)
            k = np.searchsorted(cols, i)
            if k >= cols.size or cols[k] != i:
                raise RuntimeError("Missing diagonal 3x3 block in kmat.")
            Db = data[k]
            Db = 0.5 * (Db + Db.T)
            diag_inv[i] = np.linalg.inv(Db)

        Dinv = sp.bsr_matrix(
            (diag_inv, np.arange(nblk, dtype=np.int32), np.arange(nblk + 1, dtype=np.int32)),
            shape=(N, N),
            blocksize=(bs, bs),
        ).tocsr()

        # fixed sparsity mask from initial K@P
        mask = None
        if (not with_fillin) and use_mask:
            mask = ((K @ P) != 0).astype(np.int8)

        def control(Z):
            if mask is None:
                return Z
            if not sp.issparse(Z):
                Z = sp.csr_matrix(Z)
            return Z.multiply(mask)

        P = control(P)

        # Jacobi-preconditioned energy smoothing: P <- P - omega * Dinv * (K P)
        if with_fillin or (mask is None):
            for _ in range(int(n_sweeps)):
                KP = K @ P
                P = P - float(omega) * (Dinv @ KP)
        else:
            for _ in range(int(n_sweeps)):
                KP = control(K @ P)
                P = control(P - float(omega) * (Dinv @ KP))

        P_out = P.toarray()
        self._lock_P_cache[cache_key] = P_out
        return P_out


    def prolongate(self, coarse_u: np.ndarray, nxe_coarse: int):
        dpn = self.dof_per_node
        nxc = nxe_coarse + 1
        Nc = nxc * nxc
        assert coarse_u.size == dpn * Nc

        nxe_f = 2 * nxe_coarse
        nxf = nxe_f + 1
        Nf = nxf * nxf

        # NOTE : make sure with_fillin = True for all energy / locking smoothed methods
        if self.prolong_mode == "locking-global":
            method = self._locking_aware_prolong_global_mitc_v1
            # method = self._locking_aware_prolong_global_mitc_v2
            P = method(nxe_coarse, length=1.0)
        elif self.prolong_mode == 'locking-local':
            # P = self._locking_aware_prolong_local_mitc_v1(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
            # P = self._locking_aware_prolong_local_mitc_v2_jacobi(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
            # P = self._locking_aware_prolong_local_mitc_v3_jacobi(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
            # test more GPU-friendly version with v4 (BCs enforced on fine and coarse BCs instead of extra eqns)
            # P = self._locking_aware_prolong_local_mitc_v4_jacobi(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
            P = self._locking_aware_prolong_local_mitc_v5_jacobi(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
        elif self.prolong_mode == "standard":
            P = self._build_P2_uncoupled3(nxe_coarse)
            P = self._apply_bcs_to_P(P, nxe_coarse)
        elif self.prolong_mode == "energy-jacobi":
            P = self._energy_smooth_jacobi_v1(nxe_coarse, n_sweeps=self.n_lock_sweeps, omega=self.omega)
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

        # NOTE : make sure with_fillin = True for all energy / locking smoothed methods
        if self.prolong_mode == "locking-global":
            method = self._locking_aware_prolong_global_mitc_v1
            # method = self._locking_aware_prolong_global_mitc_v2
            P = method(nxe_coarse, length=1.0)
        elif self.prolong_mode == 'locking-local':
            # P = self._locking_aware_prolong_local_mitc_v1(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
            # P = self._locking_aware_prolong_local_mitc_v2_jacobi(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
            # P = self._locking_aware_prolong_local_mitc_v3_jacobi(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
            # test more GPU-friendly version with v4 (BCs enforced on fine and coarse BCs instead of extra eqns)
            # P = self._locking_aware_prolong_local_mitc_v4_jacobi(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
            P = self._locking_aware_prolong_local_mitc_v5_jacobi(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
        elif self.prolong_mode == "standard":
            P = self._build_P2_uncoupled3(nxe_coarse)
            P = self._apply_bcs_to_P(P, nxe_coarse)
        elif self.prolong_mode == "energy-jacobi":
            P = self._energy_smooth_jacobi_v1(nxe_coarse, n_sweeps=self.n_lock_sweeps, omega=self.omega)
        else:
            raise NotImplementedError("locking-local not implemented in this prototype")

        R = P.T

        fine_r = fine_r.copy()
        self.apply_bcs_2d(fine_r, nxe_fine)

        coarse_r = R @ fine_r

        return coarse_r
