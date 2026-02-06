import numpy as np
import scipy.sparse as sp

# You already have these in your codebase (based on your snippets)
from .basis import second_order_quadrature, zero_order_quadrature, first_order_quadrature
from .basis import get_iga2_basis, get_lagrange_basis_01


# =============================================================================
# 2D DeRham-compatible Reissner–Mindlin plate element (p=2 for w, p=(1,2)/(2,1) for theta)
# =============================================================================
class DeRhamIsogeometricPlateElement:
    """
    De Rham IGA plate element (Reissner–Mindlin):
      - w     : H1   (0-form), tensor p=2 x p=2   => 3x3 dofs per elem (9)
      - thx   : H(curl) component, p=1 in x, p=2 in y => 2x3 dofs per elem (6)
      - thy   : H(curl) component, p=2 in x, p=1 in y => 3x2 dofs per elem (6)

    Local ordering:
      w   : (ix,iy) over {0..2}x{0..2}  -> 9
      thx : (ix,iy) over {0..1}x{0..2}  -> 6
      thy : (ix,iy) over {0..2}x{0..1}  -> 6
    """

    def __init__(self, reduced_integrated:bool=False, clamped: bool = False):
        self.dof_per_node = 1  # you store fields in separate blocks (block-CSR style)
        self.reduced_integrated = reduced_integrated
        self.clamped = bool(clamped)

    # ---- tensor helpers ------------------------------------------------------
    @staticmethod
    def _tensor_product_basis_22(xi, eta, bx, by):
        # bx=(Nx, dNx/dxi), by=(Ny, dNy/deta)
        Nx, dNx = bx
        Ny, dNy = by
        # tensor basis size = (len(Nx)*len(Ny),)
        N = np.kron(Ny, Nx)
        Nxi = np.kron(Ny, dNx)
        Neta = np.kron(dNy, Nx)
        return N, Nxi, Neta

    @staticmethod
    def _iga2_1d(x, left, right):
        return get_iga2_basis(x, left, right)  # (N(3,), dN(3,))

    @staticmethod
    def _p1_1d(x):
        return get_lagrange_basis_01(x)  # (N(2,), dN(2,))

    def get_kelem(
        self,
        E: float,
        nu: float,
        thick: float,
        dx: float,
        dy: float,
        left_bndry: bool,
        right_bndry: bool,
        bot_bndry: bool,
        top_bndry: bool,
    ):
        """
        Returns 3x3 block stiffness (dense) for local dofs:
            [ w(9), thx(6), thy(6) ]

        We assume a *diagonal/affine* geometry map per element:
            x = x0 + xi * dx,   y = y0 + eta * dy,  with xi,eta in [0,1]
        """
        pts, wts = second_order_quadrature()  # length 3
        if self.reduced_integrated:
            # shear_pts, shear_wts = zero_order_quadrature()
            shear_pts, shear_wts = first_order_quadrature()
        else:
            shear_pts, shear_wts = second_order_quadrature()

        n_w = 9
        n_tx = 6
        n_ty = 6

        Kww   = np.zeros((n_w,  n_w))
        Kwtx  = np.zeros((n_w,  n_tx))
        Kwty  = np.zeros((n_w,  n_ty))

        Ktxw  = np.zeros((n_tx, n_w))
        Ktxtx = np.zeros((n_tx, n_tx))
        Ktxty = np.zeros((n_tx, n_ty))

        Ktyw  = np.zeros((n_ty, n_w))
        Ktytx = np.zeros((n_ty, n_tx))
        Ktyty = np.zeros((n_ty, n_ty))

        # material
        EI = E * thick**3 / 12.0 / (1.0 - nu**2)
        ks = 5.0 / 6.0
        G = E / (2.0 * (1.0 + nu))
        ksGA = ks * G * thick

        Db = EI * np.array([
            [1.0, nu, 0.0],
            [nu,  1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])

        # affine map
        J = dx * dy
        xi_x = 1.0 / dx
        eta_y = 1.0 / dy

        for jj in range(3):
            _eta = pts[jj]
            w_eta = wts[jj]
            eta = 0.5 * (_eta + 1.0)
            w_eta *= 0.5

            # 1D y-bases
            Ny2, dNy2 = self._iga2_1d(eta, bot_bndry, top_bndry)  # (3,)
            Ny1, dNy1 = self._p1_1d(eta)                          # (2,)

            for ii in range(3):
                _xi = pts[ii]
                w_xi = wts[ii]
                xi = 0.5 * (_xi + 1.0)
                w_xi *= 0.5

                wt = (w_xi * w_eta) * J  # already includes mapping from [-1,1]^2 to [0,1]^2 via 0.5 factors above

                # 1D x-bases
                Nx2, dNx2 = self._iga2_1d(xi, left_bndry, right_bndry)  # (3,)
                Nx1, dNx1 = self._p1_1d(xi)                              # (2,)

                # --- w basis: (2,2) => 3x3 = 9
                Nw, Nw_xi, Nw_eta = self._tensor_product_basis_22(
                    xi, eta, (Nx2, dNx2), (Ny2, dNy2)
                )
                # physical derivs
                Nw_x = Nw_xi * xi_x
                Nw_y = Nw_eta * eta_y

                # --- thx basis: (1,2) => 2x3 = 6
                Ntx, Ntx_xi, Ntx_eta = self._tensor_product_basis_22(
                    xi, eta, (Nx1, dNx1), (Ny2, dNy2)
                )
                Ntx_x = Ntx_xi * xi_x
                Ntx_y = Ntx_eta * eta_y

                # --- thy basis: (2,1) => 3x2 = 6
                Nty, Nty_xi, Nty_eta = self._tensor_product_basis_22(
                    xi, eta, (Nx2, dNx2), (Ny1, dNy1)
                )
                Nty_x = Nty_xi * xi_x
                Nty_y = Nty_eta * eta_y

                # =========================
                # Bending: (kappa^T Db kappa)
                # kappa = [thx_x, thy_y, thx_y + thy_x]
                # =========================
                # Build Bb blocks via outer products:
                # k1 = thx_x
                # k2 = thy_y
                # k3 = thx_y + thy_x
                #
                # => energy = [k1,k2,k3] Db [k1,k2,k3]^T
                #
                cB = wt

                # Contributions:
                # thx-thx: k1 and k3(thx_y)
                # thy-thy: k2 and k3(thy_x)
                # thx-thy: k3 cross terms, and Db coupling terms

                # For convenience define:
                k1_tx = Ntx_x
                k3_tx = Ntx_y
                k2_ty = Nty_y
                k3_ty = Nty_x

                # Assemble with Db components:
                D11, D12, D33 = Db[0,0], Db[0,1], Db[2,2]
                D22 = Db[1,1]

                # thx-thx from k1*D11*k1 + k3*D33*k3
                Ktxtx += cB * (D11 * np.outer(k1_tx, k1_tx) + D33 * np.outer(k3_tx, k3_tx))

                # thy-thy from k2*D22*k2 + k3*D33*k3
                Ktyty += cB * (D22 * np.outer(k2_ty, k2_ty) + D33 * np.outer(k3_ty, k3_ty))

                # coupling thx-thy:
                # k1*D12*k2 terms:
                Ktxty += cB * (D12 * np.outer(k1_tx, k2_ty))
                Ktytx += cB * (D12 * np.outer(k2_ty, k1_tx))

                # k3 coupling: D33 * (thx_y + thy_x)^2 => cross term 2*D33*(thx_y)*(thy_x)
                Ktxty += cB * (D33 * np.outer(k3_tx, k3_ty))
                Ktytx += cB * (D33 * np.outer(k3_ty, k3_tx))

        nshear = len(shear_pts)
        for jj in range(nshear):
            _eta = shear_pts[jj]
            w_eta = shear_wts[jj]
            eta = 0.5 * (_eta + 1.0)
            w_eta *= 0.5

            # 1D y-bases
            Ny2, dNy2 = self._iga2_1d(eta, bot_bndry, top_bndry)  # (3,)
            Ny1, dNy1 = self._p1_1d(eta)                          # (2,)

            for ii in range(nshear):
                _xi = shear_pts[ii]
                w_xi = shear_wts[ii]
                xi = 0.5 * (_xi + 1.0)
                w_xi *= 0.5

                wt = (w_xi * w_eta) * J  # already includes mapping from [-1,1]^2 to [0,1]^2 via 0.5 factors above

                # 1D x-bases
                Nx2, dNx2 = self._iga2_1d(xi, left_bndry, right_bndry)  # (3,)
                Nx1, dNx1 = self._p1_1d(xi)                              # (2,)

                # --- w basis: (2,2) => 3x3 = 9
                Nw, Nw_xi, Nw_eta = self._tensor_product_basis_22(
                    xi, eta, (Nx2, dNx2), (Ny2, dNy2)
                )
                # physical derivs
                Nw_x = Nw_xi * xi_x
                Nw_y = Nw_eta * eta_y

                # --- thx basis: (1,2) => 2x3 = 6
                Ntx, Ntx_xi, Ntx_eta = self._tensor_product_basis_22(
                    xi, eta, (Nx1, dNx1), (Ny2, dNy2)
                )
                Ntx_x = Ntx_xi * xi_x
                Ntx_y = Ntx_eta * eta_y

                # --- thy basis: (2,1) => 3x2 = 6
                Nty, Nty_xi, Nty_eta = self._tensor_product_basis_22(
                    xi, eta, (Nx2, dNx2), (Ny1, dNy1)
                )
                Nty_x = Nty_xi * xi_x
                Nty_y = Nty_eta * eta_y

                # =========================
                # Shear: ksGA * [ (w_x - thx)^2 + (w_y - thy)^2 ]
                # =========================
                cS = ksGA * wt

                # (w_x - thx)^2
                Kww   += cS * np.outer(Nw_x, Nw_x)
                Kwtx  -= cS * np.outer(Nw_x, Ntx)
                Ktxw  -= cS * np.outer(Ntx,  Nw_x)
                Ktxtx += cS * np.outer(Ntx,  Ntx)

                # (w_y - thy)^2
                Kww   += cS * np.outer(Nw_y, Nw_y)
                Kwty  -= cS * np.outer(Nw_y, Nty)
                Ktyw  -= cS * np.outer(Nty,  Nw_y)
                Ktyty += cS * np.outer(Nty,  Nty)


        # pack as full 3x3 block
        return (Kww, Kwtx, Kwty,
                Ktxw, Ktxtx, Ktxty,
                Ktyw, Ktytx, Ktyty)

    def get_felem(
        self,
        load_fcn,     # callable q(x,y)
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        left_bndry: bool,
        right_bndry: bool,
        bot_bndry: bool,
        top_bndry: bool,
    ):
        """
        Consistent load vector for q(x,y) acting on w only:
            f_w = ∫ Nw q dA
        """
        pts, wts = second_order_quadrature()
        fw = np.zeros(9)
        ftx = np.zeros(6)
        fty = np.zeros(6)

        J = dx * dy

        for jj in range(3):
            _eta = pts[jj]
            w_eta = 0.5 * wts[jj]
            eta = 0.5 * (_eta + 1.0)

            Ny2, dNy2 = self._iga2_1d(eta, bot_bndry, top_bndry)

            for ii in range(3):
                _xi = pts[ii]
                w_xi = 0.5 * wts[ii]
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * J

                Nx2, dNx2 = self._iga2_1d(xi, left_bndry, right_bndry)

                # w basis
                Nw, _, _ = self._tensor_product_basis_22(xi, eta, (Nx2, dNx2), (Ny2, dNy2))

                xq = x0 + xi * dx
                yq = y0 + eta * dy
                q = float(load_fcn(xq, yq))

                fw += (q * Nw) * wt

        return fw, ftx, fty

    # -------------------------------------------------------------------------
    # Multigrid transfers (dyadic refinement) for [w, thx, thy]
    # -------------------------------------------------------------------------
    def _build_R_p2(self, nxe_c: int) -> np.ndarray:
        """
        Your quadratic (p=2) restriction on a 1D line with n = nxe + 2 dofs.
        Copied conceptually from your 1D DeRham element.
        """
        n_w_c = nxe_c + 2
        nxe_f = 2 * nxe_c
        n_w_f = nxe_f + 2

        R = np.zeros((n_w_c, n_w_f))
        counts = 1e-14 * np.ones((n_w_c, n_w_f))

        for ielem_c in range(nxe_c):
            left_felem = 2 * ielem_c
            l_mat = np.array([
                [0.75, 0.25, 0.0],
                [0.25, 0.75, 0.75],
                [0.0,  0.0,  0.25],
            ])
            if ielem_c == 0:
                l_mat[0, :2] = np.array([1.0, 0.5])
                l_mat[1, :2] = np.array([0.0, 0.5])
            if ielem_c == nxe_c - 1:
                l_mat[1, 2] = 0.5
                l_mat[2, 2] = 0.5

            l_nz = l_mat / (l_mat + 1e-14)
            R[ielem_c:ielem_c+3, left_felem:left_felem+3] += l_mat
            counts[ielem_c:ielem_c+3, left_felem:left_felem+3] += l_nz

            right_felem = 2 * ielem_c + 1
            r_mat = np.array([
                [0.25, 0.0,  0.0],
                [0.75, 0.75, 0.25],
                [0.0,  0.25, 0.75],
            ])
            if ielem_c == 0:
                r_mat[0, 0] = 0.5
                r_mat[1, 0] = 0.5
            if ielem_c == nxe_c - 1:
                r_mat[1, 1:] = np.array([0.5, 0.0])
                r_mat[2, 1:] = np.array([0.5, 1.0])

            r_nz = r_mat / (r_mat + 1e-14)
            R[ielem_c:ielem_c+3, right_felem:right_felem+3] += r_mat
            counts[ielem_c:ielem_c+3, right_felem:right_felem+3] += r_nz

        R /= counts
        return R
    
    def _build_P_p1(self, nxe_c: int) -> np.ndarray:
        # """
        # 1D P1 prolongation on nodal line (dyadic refinement).
        # Coarse n_c = nxe_c + 1
        # Fine   n_f = 2*nxe_c + 1

        # Constructed as P = 2*R^T (Galerkin-friendly), with a small boundary patch
        # so it matches geometric linear interpolation at the first/last midpoints.
        # """
        # R = self._build_R_p1(nxe_c)          # (n_c, n_f)
        # P = 2.0 * R.T                        # (n_f, n_c)

        # n_c = nxe_c + 1
        # n_f = 2 * nxe_c + 1

        # # Patch the two boundary-adjacent odd points to be true linear interpolation
        # # th_f[1]     = 0.5*(th_c[0] + th_c[1])
        # # th_f[n_f-2] = 0.5*(th_c[n_c-2] + th_c[n_c-1])
        # if n_c >= 2:
        #     P[1, :] = 0.0
        #     P[1, 0] = 0.5
        #     P[1, 1] = 0.5

        #     P[n_f - 2, :] = 0.0
        #     P[n_f - 2, n_c - 2] = 0.5
        #     P[n_f - 2, n_c - 1] = 0.5

        # print(f"{P=}")

        # return P
    
        """
        Geometric 1D P1 prolongation matrix (dyadic refinement).
        Coarse n_c = nxe_c + 1
        Fine   n_f = 2*nxe_c + 1

        th_f[2i]   = th_c[i]
        th_f[2i+1] = 0.5*(th_c[i] + th_c[i+1])
        """
        n_c = nxe_c + 1
        n_f = 2 * nxe_c + 1
        P = np.zeros((n_f, n_c))

        for i in range(nxe_c):
            P[2*i,   i]   = 1.0
            P[2*i+1, i]   = 0.5
            P[2*i+1, i+1] = 0.5
        P[2*nxe_c, nxe_c] = 1.0
        return P


    def _build_R_p1(self, nxe_c: int, is_prolong:bool=False) -> np.ndarray:
        """
        1D P1 restriction (full-weighting) on nodal line with n = nxe + 1 dofs.
        Fine nxe_f = 2*nxe_c => n_f = 2*nxe_c + 1
        """
        n_c = nxe_c + 1
        n_f = 2 * nxe_c + 1
        R = np.zeros((n_c, n_f))

        # injection at ends
        R[0, 0] = 1.0
        R[-1, -1] = 1.0
        for i in range(1, n_c - 1):
            R[i, 2*i - 1] = 0.25
            R[i, 2*i]     = 0.50
            R[i, 2*i + 1] = 0.25
        if is_prolong:
            # divide by col sums
            for j in range(n_f):
                R[:, j] /= np.sum(R[:, j])
            
        # print(f"{R=}")
        # exit()
        return R

    @staticmethod
    def _kron2(Ry: np.ndarray, Rx: np.ndarray) -> np.ndarray:
        # (nyc,n yf) kron (nxc,n xf) -> (nxc*nyc, nxf*nyf)
        return np.kron(Ry, Rx)

    # boundary application on vectors [w, thx, thy]
    def apply_bcs_2d(self, u: np.ndarray, nxw: int, nyw: int, nxtx: int, nytx: int, nxty: int, nyty: int, mode: str):
        """
        Strong essential BC projector/substitution:
          - Simply supported (default): w = 0 on boundary
          - Clamped: w=0 and theta=(0,0) on boundary

        mode:
          - "prolong": enforce by overwriting dofs
          - "restrict": enforce by zeroing rows (defect) (same effect here: set to 0)
        """
        u = np.asarray(u)
        nw = nxw * nyw
        ntx = nxtx * nytx
        nty = nxty * nyty

        w = u[:nw]
        tx = u[nw:nw+ntx]
        ty = u[nw+ntx:]

        def on_bndry(i, j, nx, ny):
            return (i == 0) or (i == nx-1) or (j == 0) or (j == ny-1)

        # w boundary
        for j in range(nyw):
            for i in range(nxw):
                if on_bndry(i, j, nxw, nyw):
                    w[i + nxw*j] = 0.0

        if self.clamped:
            # thx boundary on its grid
            for j in range(nytx):
                for i in range(nxtx):
                    if on_bndry(i, j, nxtx, nytx):
                        tx[i + nxtx*j] = 0.0
            # thy boundary on its grid
            for j in range(nyty):
                for i in range(nxty):
                    if on_bndry(i, j, nxty, nyty):
                        ty[i + nxty*j] = 0.0

        out = np.concatenate([w, tx, ty])
        return out

    def prolongate(self, u_c: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        u = [w, thx, thy]
        sizes:
          w   : (nxe+2) x (nye+2)
          thx : (nxe+1) x (nye+2)
          thy : (nxe+2) x (nye+1)
        """
        u_c = np.asarray(u_c)

        nxw_c, nyw_c = nxe_c + 2, nye_c + 2
        nxtx_c, nytx_c = nxe_c + 1, nye_c + 2
        nxty_c, nyty_c = nxe_c + 2, nye_c + 1

        nw_c = nxw_c * nyw_c
        ntx_c = nxtx_c * nytx_c
        nty_c = nxty_c * nyty_c
        assert u_c.size == nw_c + ntx_c + nty_c

        w_c = u_c[:nw_c]
        tx_c = u_c[nw_c:nw_c+ntx_c]
        ty_c = u_c[nw_c+ntx_c:]

        # Build restriction operators and take P = R^T
        Rx2 = self._build_R_p2(nxe_c)
        Ry2 = self._build_R_p2(nye_c)
        Rx1 = self._build_P_p1(nxe_c).T
        Ry1 = self._build_P_p1(nye_c).T

        # w: (p2,p2)
        Rw = self._kron2(Ry2, Rx2)
        Pw = Rw.T

        # thx: (p1 in x, p2 in y)
        Rtx = self._kron2(Ry2, Rx1)
        Ptx = Rtx.T

        # thy: (p2 in x, p1 in y)
        Rty = self._kron2(Ry1, Rx2)
        Pty = Rty.T

        w_f  = Pw @ w_c
        tx_f = Ptx @ tx_c
        ty_f = Pty @ ty_c

        # sizes fine
        nxe_f, nye_f = 2*nxe_c, 2*nye_c
        nxw_f, nyw_f = nxe_f + 2, nye_f + 2
        nxtx_f, nytx_f = nxe_f + 1, nye_f + 2
        nxty_f, nyty_f = nxe_f + 2, nye_f + 1

        u_f = np.concatenate([w_f, tx_f, ty_f])
        u_f = self.apply_bcs_2d(u_f, nxw_f, nyw_f, nxtx_f, nytx_f, nxty_f, nyty_f, mode="prolong")
        return u_f

    def restrict_defect(self, r_f: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        Restrict fine defect -> coarse defect for dyadic refinement.
        Fine has nxe_f=2*nxe_c, nye_f=2*nye_c.
        """
        r_f = np.asarray(r_f)

        nxe_f, nye_f = 2*nxe_c, 2*nye_c

        nxw_f, nyw_f = nxe_f + 2, nye_f + 2
        nxtx_f, nytx_f = nxe_f + 1, nye_f + 2
        nxty_f, nyty_f = nxe_f + 2, nye_f + 1

        nw_f = nxw_f * nyw_f
        ntx_f = nxtx_f * nytx_f
        nty_f = nxty_f * nyty_f
        assert r_f.size == nw_f + ntx_f + nty_f

        w_f  = r_f[:nw_f]
        tx_f = r_f[nw_f:nw_f+ntx_f]
        ty_f = r_f[nw_f+ntx_f:]

        # restriction ops
        Rx2 = self._build_R_p2(nxe_c)
        Ry2 = self._build_R_p2(nye_c)
        Rx1 = self._build_R_p1(nxe_c, is_prolong=False)
        Ry1 = self._build_R_p1(nye_c, is_prolong=False)

        Rw  = self._kron2(Ry2, Rx2)
        Rtx = self._kron2(Ry2, Rx1)
        Rty = self._kron2(Ry1, Rx2)

        w_c  = Rw  @ w_f
        tx_c = Rtx @ tx_f
        ty_c = Rty @ ty_f

        nxw_c, nyw_c = nxe_c + 2, nye_c + 2
        nxtx_c, nytx_c = nxe_c + 1, nye_c + 2
        nxty_c, nyty_c = nxe_c + 2, nye_c + 1

        r_c = np.concatenate([w_c, tx_c, ty_c])
        r_c = self.apply_bcs_2d(r_c, nxw_c, nyw_c, nxtx_c, nytx_c, nxty_c, nyty_c, mode="restrict")
        return r_c