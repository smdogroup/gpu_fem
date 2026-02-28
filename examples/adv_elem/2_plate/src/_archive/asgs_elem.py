import numpy as np
from .basis import second_order_quadrature, zero_order_quadrature, first_order_quadrature
from .basis import interp_lagrange, interp_lagrange_transpose
from .basis import get_lagrange_basis_2d_all

import scipy.sparse as sp

class AlgebraicSubGridScaleElement:
    # reissner-mindlin C0-continuous element 
    # with ASGS method (algebraic sub-grid scale)
    # A variational multiscale stabilized finite element formulation for Reissner–Mindlin plates and Timoshenko beams
    # https://upcommons.upc.edu/server/api/core/bitstreams/73fce476-07ab-4ea8-84b0-3552f852f9e7/content

    def __init__(self):              
        self.dof_per_node = 3
        self.nodes_per_elem = 4
        self.ndof = self.dof_per_node * self.nodes_per_elem
        self.clamped = False

        # stabilization constants for A
        # this is copied from TS beam, does this still work for RM plate? idk..
        self.c1 = 12.0

        # sweep in c3 values (should be 12 but a bit better with c3 = 6 (c3 = 6 makes t/L = 1e-2 worse though)?)
        self.c3 = 12.0
        # self.c3 = 24.0
        # self.c3 = 10.0
        # self.c3 = 6.0
        # self.c3 = 3.0
        # self.c3 = 20.0
        # self.c3 = 2.0
        self.c2 = self.c4 = 1.0

        # cache for prolong/restrict operators
        self._P1_cache = {}   # key: nxe_coarse -> P1 (csr)
        self._P2_cache = {}   # key: nxe_coarse -> P2 (csr)

    def get_kelem(self, E: float, nu: float, thick: float, elem_xpts: np.ndarray):

        pts, wts = second_order_quadrature()

        kelem = np.zeros((self.ndof, self.ndof))

        k1 = E * thick**3 / (24.0 * (1.0 + nu))
        k2 = E * thick**3 / (24.0 * (1.0 - nu))

        ks = 5.0 / 6.0
        G = E / (2.0 * (1.0 + nu))
        ksGh = ks * G * thick
        Ds = ksGh * np.eye(2)

        x = elem_xpts[0::3]
        y = elem_xpts[1::3]

        dx = abs(np.max(x) - np.min(x))
        dy = abs(np.max(y) - np.min(y))
        h = np.sqrt(dx * dy)     # or h = max(dx, dy)

        eps = 1.0 / ksGh
        kstab = k1 + k2
        # tau_w = 0.0
        tau_th = 1.0 / (self.c1 * kstab / h**2 + self.c2 / eps)
        tau_w = 1.0 / (self.c3 / eps / h**2 + self.c4 / eps**2 / kstab)

        # tau_w goes to zero
        # ks_final = k2 - tau_w * (ksGh**2)
        # ksGh_final = 1.0 - tau_th * ksGh
        # print(f"{ks_final=:.4e} {ksGh_final=:.4e}")

        for ii, xi in enumerate(pts):
            for jj, eta in enumerate(pts):
                wt = wts[ii] * wts[jj]

                N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)

                x_xi = np.dot(Nxi, x);  x_eta = np.dot(Neta, x)
                y_xi = np.dot(Nxi, y);  y_eta = np.dot(Neta, y)

                J = x_xi * y_eta - x_eta * y_xi
                invJ = 1.0 / J
                xi_x  =  y_eta * invJ
                xi_y  = -x_eta * invJ
                eta_x = -y_xi  * invJ
                eta_y =  x_xi  * invJ

                Nx = Nxi * xi_x + Neta * eta_x
                Ny = Nxi * xi_y + Neta * eta_y

                Bdiv = np.zeros((1, self.ndof))
                Bdiv[0, 1::3] = Nx
                Bdiv[0, 2::3] = Ny

                # ================
                # paper elem formulation

                Bgrad = np.zeros((4, self.ndof))
                Bgrad[0, 1::3] = Nx
                Bgrad[1, 1::3] = Ny
                Bgrad[2, 2::3] = Nx
                Bgrad[3, 2::3] = Ny

                kelem += k1 * (Bgrad.T @ Bgrad) * (wt * J)
                kelem += k2 * (Bdiv.T @ Bdiv) * (wt * J)

                # # proper plate bending
                # # NOTE : not sure how to get consistent results with paper yet (this doesn't seem to match)
                # # if use above bending => not mesh converging but great GMG performance
                # Bk = np.zeros((3, self.ndof))
                # Bk[0, 1::3] = Nx          # thx_x
                # Bk[1, 2::3] = Ny          # thy_y
                # Bk[2, 1::3] = Ny          # thx_y
                # Bk[2, 2::3] = Nx          # thy_x

                # Db = (E*thick**3)/(12.0*(1.0-nu**2)) * np.array([
                #     [1.0, nu, 0.0],
                #     [nu, 1.0, 0.0],
                #     [0.0, 0.0, 0.5*(1.0-nu)]
                # ])

                # kelem += (Bk.T @ Db @ Bk) * (wt*J)

                Bs = np.zeros((2, self.ndof))
                Bs[0, 0::3] = Nx
                Bs[0, 1::3] -= N
                Bs[1, 0::3] = Ny
                Bs[1, 2::3] -= N

                kelem += (Bs.T @ Ds @ Bs) * (wt * J)

                kelem -= tau_w * (ksGh**2) * (Bdiv.T @ Bdiv) * (wt * J)
                kelem -= tau_th * ksGh * (Bs.T @ Ds @ Bs) * (wt * J)

        return kelem

    # def get_kelem(self, E: float, nu: float, thick: float, elem_xpts: np.ndarray):
    #     pts, wts = second_order_quadrature()
    #     kelem = np.zeros((self.ndof, self.ndof))

    #     # material
    #     ks = 5.0 / 6.0
    #     G = E / (2.0 * (1.0 + nu))
    #     ksGh = ks * G * thick
    #     Ds = ksGh * np.eye(2)

    #     x = elem_xpts[0::3]
    #     y = elem_xpts[1::3]
    #     dx = abs(np.max(x) - np.min(x))
    #     dy = abs(np.max(y) - np.min(y))
    #     h = max(dx, dy)  # recommend for robustness

    #     eps = 1.0 / ksGh
    #     # if you keep your tau formulas:
    #     k1 = E * thick**3 / (24.0 * (1.0 + nu))
    #     k2 = E * thick**3 / (24.0 * (1.0 - nu))
    #     kstab = k1 + k2
    #     tau_th = 1.0 / (self.c1 * kstab / h**2 + self.c2 / eps)
    #     tau_w  = 1.0 / (self.c3 / eps / h**2 + self.c4 / eps**2 / kstab)

    #     # --- accumulators for projected residual operators ---
    #     # area = 0.0
    #     # Bs_int   = np.zeros((2, self.ndof))  # ∫ Bs0 dA
    #     # Bdiv_int = np.zeros((1, self.ndof))  # ∫ Bdiv dA
    #     # accumulators
    #     b2_int = 0.0
    #     Bs_b_int   = np.zeros((2, self.ndof))  # ∫ b Bs0 dA
    #     Bdiv_b_int = np.zeros((1, self.ndof))  # ∫ b Bdiv dA

    #     for ii, xi in enumerate(pts):
    #         for jj, eta in enumerate(pts):
    #             wt = wts[ii] * wts[jj]
    #             N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)

    #             x_xi = np.dot(Nxi, x);  x_eta = np.dot(Neta, x)
    #             y_xi = np.dot(Nxi, y);  y_eta = np.dot(Neta, y)
    #             J = x_xi * y_eta - x_eta * y_xi

    #             invJ = 1.0 / J
    #             xi_x  =  y_eta * invJ
    #             xi_y  = -x_eta * invJ
    #             eta_x = -y_xi  * invJ
    #             eta_y =  x_xi  * invJ

    #             Nx = Nxi * xi_x + Neta * eta_x
    #             Ny = Nxi * xi_y + Neta * eta_y

    #             dA = wt * J

    #             # --- your bending (whatever you use) ---
    #             Bgrad = np.zeros((4, self.ndof))
    #             Bgrad[0, 1::3] = Nx
    #             Bgrad[1, 1::3] = Ny
    #             Bgrad[2, 2::3] = Nx
    #             Bgrad[3, 2::3] = Ny
    #             kelem += k1 * (Bgrad.T @ Bgrad) * dA

    #             Bdiv = np.zeros((1, self.ndof))
    #             Bdiv[0, 1::3] = Nx
    #             Bdiv[0, 2::3] = Ny
    #             kelem += k2 * (Bdiv.T @ Bdiv) * dA

    #             # --- kinematic shear operator Bs0 ---
    #             Bs0 = np.zeros((2, self.ndof))
    #             Bs0[0, 0::3] = Nx
    #             Bs0[0, 1::3] -= N
    #             Bs0[1, 0::3] = Ny
    #             Bs0[1, 2::3] -= N

    #             # physical shear energy (full integration, MG-friendly)
    #             kelem += (Bs0.T @ Ds @ Bs0) * dA

    #             # --- accumulate integrals for projected-ASGS ---
    #             # area += dA
    #             # Bs_int   += Bs0 * dA
    #             # Bdiv_int += Bdiv * dA

    #             # in quad loop, after you have xi, eta, dA, Bs0, Bdiv:
    #             b = (1.0 - xi*xi) * (1.0 - eta*eta)

    #             b2_int     += (b*b) * dA
    #             Bs_b_int   += (b)   * Bs0  * dA
    #             Bdiv_b_int += (b)   * Bdiv * dA

    #     # # --- projected operators (P0 L2 projection) ---
    #     # Bs_bar   = Bs_int / area          # 2 x ndof
    #     # Bdiv_bar = Bdiv_int / area        # 1 x ndof

    #     # # --- ASGS stabilization using projected residuals ---
    #     # # paper uses tau/eps^2; since eps=1/ksGh => 1/eps^2 = ksGh^2
    #     # coef = 1.0 / (eps**2)

    #     # kelem -= (tau_th * coef) * (Bs_bar.T @ Bs_bar) * area
    #     # kelem -= (tau_w  * coef) * (Bdiv_bar.T @ Bdiv_bar) * area

    #     coef = 1.0 / (eps**2)  # = ksGh**2

    #     # bubble-projected stabilization (scalar “mass” b2_int)
    #     kelem -= (tau_th * coef) * (Bs_b_int.T @ Bs_b_int) / b2_int
    #     kelem -= (tau_w  * coef) * (Bdiv_b_int.T @ Bdiv_b_int) / b2_int

    #     return kelem

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

        P2 = self._build_P2_scalar(nxe_coarse)

        fine_u = np.zeros(dpn * Nf, dtype=float)

        # apply prolongation per dof-slice
        for a in range(dpn):
            fine_u[a::dpn] = P2 @ coarse_u[a::dpn]

        # enforce BCs the same way you currently do
        self.apply_bcs_2d(fine_u, nxe_f)
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

        P2 = self._build_P2_scalar(nxe_coarse)
        R2 = P2.T  # per your requirement

        fine_r = fine_r.copy()
        self.apply_bcs_2d(fine_r, nxe_fine)

        coarse_r = np.zeros(dpn * Nc, dtype=float)
        for a in range(dpn):
            coarse_r[a::dpn] = R2 @ fine_r[a::dpn]

        self.apply_bcs_2d(coarse_r, nxe_coarse)
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
