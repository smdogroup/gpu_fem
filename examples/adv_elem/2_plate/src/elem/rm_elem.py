import numpy as np
from .basis import second_order_quadrature, zero_order_quadrature, first_order_quadrature
from .basis import interp_lagrange, interp_lagrange_transpose
from .basis import get_lagrange_basis_2d_all

import scipy.sparse as sp

class ReissnerMindlinPlateElement:
    # reissner-mindlin C0-continuous element

    def __init__(self, reduced_integrated:bool=False):              
        self.dof_per_node = 3
        self.nodes_per_elem = 4
        self.ndof = self.dof_per_node * self.nodes_per_elem
        self.reduced_integrated = reduced_integrated
        self.clamped = False

        # cache for prolong/restrict operators
        self._P1_cache = {}   # key: nxe_coarse -> P1 (csr)
        self._P2_cache = {}   # key: nxe_coarse -> P2 (csr)

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
