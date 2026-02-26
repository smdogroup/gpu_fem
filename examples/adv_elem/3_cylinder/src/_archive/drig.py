import numpy as np
import sys
from .basis import second_order_quadrature, first_order_quadrature, zero_order_quadrature
from .basis import get_iga2_basis, get_lagrange_basis_01


class DeRhamIsogeometricCylinderElement:
    """
    De Rham IGA cylinder element on (x,y) param domain:
      x = axial
      y = hoop arc-length s = r*phi   (your derived formulas assume this)

    Unknowns (field blocks):
      w   : radial displacement (H1, p=2x2)            -> 9 dofs  (p2,p2)
      u   : axial displacement  (x-edge space)         -> 6 dofs  (p1,p2)
      v   : hoop displacement   (y-edge space)         -> 6 dofs  (p2,p1)
      thx : rotation about hoop? (y-edge space)        -> 6 dofs  (p2,p1)
      thy : rotation about axial (x-edge space)        -> 6 dofs  (p1,p2)

    This choice preserves exact discrete shear cancellation:
      2*e13 = w_x + thy
      2*e23 = w_y - v/r - thx   (y = s so w_y is w_s)

    NOTE : this element experiences some membrane locking..

    Your derived strains implemented:

    Bending:
      k11      = thy_x
      k22      = -thx_y + w/r^2 + v_y/r
      2*k12    = v_x/r + thy_y - thx_x

    Membrane:
      e11      = u_x
      e22      = v_y + w/r
      2*e12    = v_x + u_y

    Transverse shear:
      2*e13    = w_x + thy
      2*e23    = w_y - v/r - thx

    NOTE:
      - Here y is assumed to be arc-length s. If you ever switch to y=phi,
        you MUST insert 1/r factors into y-derivatives accordingly.
      - “opposite support” for u vs v is BC/assembler logic, not element stiffness.
        The element just provides K; apply different essential BCs in your projector.
    """

    def __init__(self, r: float, reduced_integrated: bool = False, clamped: bool = False, axial_factor:float=0.0, curvature_on:bool=True):
        self.dof_per_node = 1  # separate field blocks
        self.r = float(r)
        self.reduced_integrated = bool(reduced_integrated)
        self.clamped = bool(clamped)
        self.axial_factor = float(axial_factor)
        self.curvature_on = bool(curvature_on)

    # ---- tensor helpers ------------------------------------------------------
    @staticmethod
    def _tensor_product_basis(xi, eta, bx, by):
        Nx, dNx = bx
        Ny, dNy = by
        N    = np.kron(Ny, Nx)
        Nxi  = np.kron(Ny, dNx)
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
        Returns 5x5 block stiffness (dense) for local dofs ordered:
          [ w(9), u(9), v(6), thx(6), thy(6) ]
        """
        pts, wts = second_order_quadrature()
        if self.reduced_integrated:
            shear_pts, shear_wts = first_order_quadrature()
            # shear_pts, shear_wts = zero_order_quadrature()
        else:
            shear_pts, shear_wts = second_order_quadrature()

        r = self.r

        # sizes
        n_w   = 9   # (p2,p2)
        n_u   = 6   # (p1,p2)
        n_v   = 6   # (p2,p1)
        n_thx = 6   # (p2,p1)
        n_thy = 6   # (p1,p2)

        # allocate 5x5 blocks: (w,u,v,thx,thy)
        Kww   = np.zeros((n_w,   n_w))
        Kwu   = np.zeros((n_w,   n_u))
        Kwv   = np.zeros((n_w,   n_v))
        Kwtx  = np.zeros((n_w,   n_thx))
        Kwty  = np.zeros((n_w,   n_thy))

        Kuw   = np.zeros((n_u,   n_w))
        Kuu   = np.zeros((n_u,   n_u))
        Kuv   = np.zeros((n_u,   n_v))
        Kutx  = np.zeros((n_u,   n_thx))
        Kuty  = np.zeros((n_u,   n_thy))

        Kvw   = np.zeros((n_v,   n_w))
        Kvu   = np.zeros((n_v,   n_u))
        Kvv   = np.zeros((n_v,   n_v))
        Kvtx  = np.zeros((n_v,   n_thx))
        Kvty  = np.zeros((n_v,   n_thy))

        Ktxw  = np.zeros((n_thx, n_w))
        Ktxu  = np.zeros((n_thx, n_u))
        Ktxv  = np.zeros((n_thx, n_v))
        Ktxtx = np.zeros((n_thx, n_thx))
        Ktxty = np.zeros((n_thx, n_thy))

        Ktyw  = np.zeros((n_thy, n_w))
        Ktyu  = np.zeros((n_thy, n_u))
        Ktyv  = np.zeros((n_thy, n_v))
        Ktytx = np.zeros((n_thy, n_thx))
        Ktyty = np.zeros((n_thy, n_thy))

        # -------------------
        # material matrices
        # -------------------
        # bending (plane stress-like) matrix
        EI = E * thick**3 / (12.0 * (1.0 - nu**2))
        Db = EI * np.array([
            [1.0, nu, 0.0],
            [nu,  1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])

        # membrane matrix (same form, with A = E t /(1-nu^2))
        A0 = E * thick / (1.0 - nu**2)
        Dm = A0 * np.array([
            [1.0, nu, 0.0],
            [nu,  1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])

        # transverse shear (isotropic)
        ks = 5.0 / 6.0
        G  = E / (2.0 * (1.0 + nu))
        Ds = (ks * G * thick) * np.eye(2)

        # affine map scaling
        J = dx * dy
        xi_x  = 1.0 / dx
        eta_y = 1.0 / dy

        # turns off curvature terms so it should solve cylinder as plate..
        # debug_mem_off = True
        debug_mem_off = False

        # # debug_curv_off = True
        # debug_curv_off = False
        debug_curv_off = not self.curvature_on

        # half_eng_strains = True
        half_eng_strains = False

        # load = "bend"
        # load = "mem"
        # load = "both"


        # ==========================================================
        # BENDING: kappa^T Db kappa
        # kappa = [k11, k22, 2*k12]  (engineering shear curvature)
        #
        # k11     = thy_x
        # k22     = -thx_y + w/r^2 + v_y/r
        # 2*k12   = v_x/r + thy_y - thx_x
        # ==========================================================
        for jj in range(3):
            _eta = pts[jj]
            w_eta = wts[jj] * 0.5
            eta = 0.5 * (_eta + 1.0)

            Ny2, dNy2 = self._iga2_1d(eta, bot_bndry, top_bndry)  # p2
            Ny1, dNy1 = self._p1_1d(eta)                          # p1

            for ii in range(3):
                _xi = pts[ii]
                w_xi = wts[ii] * 0.5
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * J

                Nx2, dNx2 = self._iga2_1d(xi, left_bndry, right_bndry)  # p2
                Nx1, dNx1 = self._p1_1d(xi)                              # p1

                # w, u : (p2,p2)
                Nw, Nw_xi, Nw_eta = self._tensor_product_basis(xi, eta, (Nx2, dNx2), (Ny2, dNy2))
                Nw_x = Nw_xi * xi_x
                Nw_y = Nw_eta * eta_y

                # v, thx : (p2,p1)
                Nv, Nv_xi, Nv_eta = self._tensor_product_basis(xi, eta, (Nx2, dNx2), (Ny1, dNy1))
                Nv_x = Nv_xi * xi_x
                Nv_y = Nv_eta * eta_y

                Ntx, Ntx_xi, Ntx_eta = Nv, Nv_xi, Nv_eta
                Ntx_x = Nv_x
                Ntx_y = Nv_y

                # u, thy : (p1,p2)
                Nty, Nty_xi, Nty_eta = self._tensor_product_basis(xi, eta, (Nx1, dNx1), (Ny2, dNy2))
                Nty_x = Nty_xi * xi_x
                Nty_y = Nty_eta * eta_y

                Nu, Nu_xi, Nu_eta = Nty, Nty_xi, Nty_eta
                Nu_x = Nty_x
                Nu_y = Nty_y

                # k11 = thy_x
                k11_thy = Nty_x

                # k22 = -thx_y + w/r^2 + v_y/r
                k22_w   = (1.0 / (r * r)) * Nw
                # k22_w = 0.0 # ignore 2nd order term?
                k22_thx = -Ntx_y
                k22_v   = (1.0 / r) * Nv_y

                # 2*k12 = v_x/r + thy_y - thx_x
                k12_v   = (1.0 / r) * Nv_x
                k12_thy = Nty_y
                k12_thx = -Ntx_x

                if debug_curv_off:
                    # turns off all radial curvature terms so cylinder becomes like plate
                    k22_w = 0.0
                    k22_v = 0.0
                    k12_v = 0.0

                # DEBUG should be 1/2 here..
                if half_eng_strains:
                    k12_v *= 0.5
                    k12_thy *= 0.5
                    k12_thx *= 0.5

                # assemble with Db, using outer-products
                D11, D12, D22, D33 = Db[0, 0], Db[0, 1], Db[1, 1], Db[2, 2]
                cB = wt

                # k11-k11
                Ktyty += cB * (D11 * np.outer(k11_thy, k11_thy))

                # k22-k22 (w,v,thx)
                Kww   += cB * (D22 * np.outer(k22_w,   k22_w))
                Kvv   += cB * (D22 * np.outer(k22_v,   k22_v))
                Ktxtx += cB * (D22 * np.outer(k22_thx, k22_thx))

                Kwv   += cB * (D22 * np.outer(k22_w,   k22_v))
                Kvw   += cB * (D22 * np.outer(k22_v,   k22_w))
                Kwtx  += cB * (D22 * np.outer(k22_w,   k22_thx))
                Ktxw  += cB * (D22 * np.outer(k22_thx, k22_w))
                Kvtx  += cB * (D22 * np.outer(k22_v,   k22_thx))
                Ktxv  += cB * (D22 * np.outer(k22_thx, k22_v))

                # k12-k12 (v,thx,thy)
                Kvv   += cB * (D33 * np.outer(k12_v,   k12_v))
                Ktxtx += cB * (D33 * np.outer(k12_thx, k12_thx))
                Ktyty += cB * (D33 * np.outer(k12_thy, k12_thy))

                Kvtx  += cB * (D33 * np.outer(k12_v,   k12_thx))
                Ktxv  += cB * (D33 * np.outer(k12_thx, k12_v))
                Kvty  += cB * (D33 * np.outer(k12_v,   k12_thy))
                Ktyv  += cB * (D33 * np.outer(k12_thy, k12_v))
                Ktxty += cB * (D33 * np.outer(k12_thx, k12_thy))
                Ktytx += cB * (D33 * np.outer(k12_thy, k12_thx))

                # k11-k22 coupling via D12 (thy with w,v,thx)
                Kwty  += cB * (D12 * np.outer(k22_w,   k11_thy))
                Ktyw  += cB * (D12 * np.outer(k11_thy, k22_w))

                Kvty  += cB * (D12 * np.outer(k22_v,   k11_thy))
                Ktyv  += cB * (D12 * np.outer(k11_thy, k22_v))

                Ktxty += cB * (D12 * np.outer(k22_thx, k11_thy))
                Ktytx += cB * (D12 * np.outer(k11_thy, k22_thx))

                # (Optionally) you could also include D12 coupling between k11 and k12
                # if your kappa ordering is [k11,k22,2k12] and Db uses (1,3) = 0 for isotropic,
                # so there is no coupling: Db[0,2]=Db[1,2]=0.  Nothing to add.

        # ==========================================================
        # SHEAR: gamma^T Ds gamma, where gamma = [2e13, 2e23]
        #
        # 2e13 = w_x + thy
        # 2e23 = w_y - v/r - thx
        # ==========================================================
        ns = len(shear_pts)
        for jj in range(ns):
            _eta = shear_pts[jj]
            w_eta = shear_wts[jj] * 0.5
            eta = 0.5 * (_eta + 1.0)

            Ny2, dNy2 = self._iga2_1d(eta, bot_bndry, top_bndry)
            Ny1, dNy1 = self._p1_1d(eta)

            for ii in range(ns):
                _xi = shear_pts[ii]
                w_xi = shear_wts[ii] * 0.5
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * J
                cS = wt  # Ds handled below

                Nx2, dNx2 = self._iga2_1d(xi, left_bndry, right_bndry)
                Nx1, dNx1 = self._p1_1d(xi)

                # bases
                Nw, Nw_xi, Nw_eta = self._tensor_product_basis(xi, eta, (Nx2, dNx2), (Ny2, dNy2))
                Nw_x = Nw_xi * xi_x
                Nw_y = Nw_eta * eta_y

                Nv, Nv_xi, Nv_eta = self._tensor_product_basis(xi, eta, (Nx2, dNx2), (Ny1, dNy1))
                Ntx = Nv
                Nty, Nty_xi, Nty_eta = self._tensor_product_basis(xi, eta, (Nx1, dNx1), (Ny2, dNy2))

                # gamma1 = w_x + thy
                g1_w   = Nw_x
                g1_thy = Nty

                # gamma2 = w_y - v/r - thx
                g2_w   = Nw_y
                g2_v   = -(1.0 / r) * Nv
                g2_thx = -Ntx

                if debug_curv_off:
                    # turns off all radial curvature terms so cylinder becomes like plate
                    g2_v = 0.0

                # Assemble: [g1,g2]^T Ds [g1,g2]
                # Ds is diagonal isotropic here
                Ds11 = Ds[0, 0]
                Ds22 = Ds[1, 1]

                # g1
                Kww   += cS * (Ds11 * np.outer(g1_w,   g1_w))
                Kwty  += cS * (Ds11 * np.outer(g1_w,   g1_thy))
                Ktyw  += cS * (Ds11 * np.outer(g1_thy, g1_w))
                Ktyty += cS * (Ds11 * np.outer(g1_thy, g1_thy))

                # g2
                Kww   += cS * (Ds22 * np.outer(g2_w,   g2_w))
                Kvv   += cS * (Ds22 * np.outer(g2_v,   g2_v))
                Ktxtx += cS * (Ds22 * np.outer(g2_thx, g2_thx))

                Kwv   += cS * (Ds22 * np.outer(g2_w,   g2_v))
                Kvw   += cS * (Ds22 * np.outer(g2_v,   g2_w))

                Kwtx  += cS * (Ds22 * np.outer(g2_w,   g2_thx))
                Ktxw  += cS * (Ds22 * np.outer(g2_thx, g2_w))

                Kvtx  += cS * (Ds22 * np.outer(g2_v,   g2_thx))
                Ktxv  += cS * (Ds22 * np.outer(g2_thx, g2_v))

        # ==========================================================
        # MEMBRANE: eps^T Dm eps, with eps = [e11, e22, 2e12]
        #
        # e11   = u_x
        # e22   = v_y + w/r
        # 2e12  = v_x + u_y
        # ==========================================================
        for jj in range(3):
            _eta = pts[jj]
            w_eta = wts[jj] * 0.5
            eta = 0.5 * (_eta + 1.0)

            Ny2, dNy2 = self._iga2_1d(eta, bot_bndry, top_bndry)
            Ny1, dNy1 = self._p1_1d(eta)

            for ii in range(3):
                _xi = pts[ii]
                w_xi = wts[ii] * 0.5
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * J
                cM = wt

                Nx2, dNx2 = self._iga2_1d(xi, left_bndry, right_bndry)
                Nx1, dNx1 = self._p1_1d(xi)

                # w (p2,p2)
                Nw, Nw_xi, Nw_eta = self._tensor_product_basis(xi, eta, (Nx2, dNx2), (Ny2, dNy2))
                Nw_x = Nw_xi * xi_x
                Nw_y = Nw_eta * eta_y

                # reduced integrated w like MITC? oof this may not be as good..
                # TEMP DEBUG
                # Ny0 = np.array([1.0/3.0]*3)
                # Nw = np.kron(Ny0, Nx2)

                # v (p2,p1)
                Nv, Nv_xi, Nv_eta = self._tensor_product_basis(xi, eta, (Nx2, dNx2), (Ny1, dNy1))
                Nv_x = Nv_xi * xi_x
                Nv_y = Nv_eta * eta_y

                # u (p1, p2)
                Nu, Nu_xi, Nu_eta = self._tensor_product_basis(xi, eta, (Nx1, dNx1), (Ny2, dNy2))
                Nu_x = Nu_xi * xi_x
                Nu_y = Nu_eta * eta_y

                # e11 = u_x
                e11_u = Nu_x

                # e22 = v_y + w/r
                e22_v = Nv_y
                e22_w = (1.0 / r) * Nw

                # 2e12 = v_x + u_y
                e12_v = Nv_x
                e12_u = Nu_y

                # DEBUG
                if half_eng_strains:
                    e12_v *= 0.5
                    e12_u *= 0.5

                if debug_mem_off:
                    # turn off curvature and membrane strains to debug as plate
                    e11_u = 0.0
                    e22_v = 0.0
                    e12_v = 0.0
                    e12_u = 0.0

                if debug_curv_off:
                    # curvature term
                    e22_w = 0.0
                    
                # Assemble eps^T Dm eps with eps = [e11, e22, 2e12]
                D11, D12, D22, D33 = Dm[0, 0], Dm[0, 1], Dm[1, 1], Dm[2, 2]

                # e11-e11
                Kuu += cM * (D11 * np.outer(e11_u, e11_u))

                # e22-e22
                Kvv += cM * (D22 * np.outer(e22_v, e22_v))
                Kww += cM * (D22 * np.outer(e22_w, e22_w))
                Kvw += cM * (D22 * np.outer(e22_v, e22_w))
                Kwv += cM * (D22 * np.outer(e22_w, e22_v))

                # (2e12)-(2e12)
                Kvv += cM * (D33 * np.outer(e12_v, e12_v))
                Kuu += cM * (D33 * np.outer(e12_u, e12_u))
                Kvu += cM * (D33 * np.outer(e12_v, e12_u))
                Kuv += cM * (D33 * np.outer(e12_u, e12_v))

                # e11-e22 coupling (nu terms): between u_x and (v_y + w/r)
                Kvu += cM * (D12 * np.outer(e22_v, e11_u))
                Kuv += cM * (D12 * np.outer(e11_u, e22_v))

                Kwu += cM * (D12 * np.outer(e22_w, e11_u))
                Kuw += cM * (D12 * np.outer(e11_u, e22_w))

                # Note: Dm has no coupling between normal and shear for isotropic (Dm[0,2]=Dm[1,2]=0).

        # Pack 5x5 block (w, u, v, thx, thy)
        return (
            Kww,  Kwu,  Kwv,  Kwtx,  Kwty,
            Kuw,  Kuu,  Kuv,  Kutx,  Kuty,
            Kvw,  Kvu,  Kvv,  Kvtx,  Kvty,
            Ktxw, Ktxu, Ktxv, Ktxtx, Ktxty,
            Ktyw, Ktyu, Ktyv, Ktytx, Ktyty
        )
    

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
        fu = np.zeros(6)
        fv = np.zeros(6)

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
                Nx1, dNx1 = self._p1_1d(xi)

                # w basis
                # Nw, _, _ = self._tensor_product_basis_22(xi, eta, (Nx2, dNx2), (Ny2, dNy2))

                Nw, _, _ = self._tensor_product_basis(xi, eta, (Nx2, dNx2), (Ny2, dNy2))

                Nu, _, _ = self._tensor_product_basis(xi, eta, (Nx1, dNx1), (Ny2, dNy2))

                xq = x0 + xi * dx
                yq = y0 + eta * dy
                q = float(load_fcn(xq, yq))

                fw += (q * Nw) * wt
                fu += (q * Nu) * wt * self.axial_factor

        return fw, fu, fv, ftx, fty

    # -------------------------------------------------------------------------
    # Multigrid transfers (dyadic refinement) for [w, u, v, thx, thy]
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

    # boundary application on vectors [w, u, v, thx, thy]
    def apply_bcs_2d(self, u: np.ndarray, nxw: int, nyw: int, nxtx: int, nytx: int, nxty: int, nyty: int):
        """
        Strong essential BC projector/substitution:
          - Simply supported (default): w = 0 on boundary
          - Clamped: w=0, (u,v)=(0,0) and theta=(0,0) on boundary
        """
        u = np.asarray(u)
        nw = nxw * nyw
        ntx = nxtx * nytx
        nty = nxty * nyty
        nu, nv = nty, ntx

        # reference to array data (so no need to copy over, can set into subarrays)
        w = u[:nw]
        U = u[nw:(nw+nu)]
        V = u[(nw+nu):(nw+nu+nv)]
        tx = u[(nw+nu+nv):(nw+nu+nv+ntx)]
        ty = u[(nw+nu+nv+ntx):]

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
                        V[i + nxtx*j] = 0.0
            # thy boundary on its grid
            for j in range(nyty):
                for i in range(nxty):
                    if on_bndry(i, j, nxty, nyty):
                        ty[i + nxty*j] = 0.0
                        U[i + nxty*j] = 0.0
        
        else:
            # u boundary: on x=0 edge
            for j in range(nyty):
                for i in range(nxty):
                    if i == 0:
                        U[i + nxty*j] = 0.0

            # v boundary: on y=0 edge
            for j in range(nytx):
                for i in range(nxtx):
                    if j == 0:
                        V[i + nxtx*j] = 0.0

        out = np.concatenate([w, U, V, tx, ty])
        return out

    def prolongate(self, u_c: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        u = [w, u, v, thx, thy]
        sizes:
          w   : (nxe+2) x (nye+2)
          u   : (nxe+1) x (nye+2)
          v   : (nxe+2) x (nxe+1)
          thx : (nxe+2) x (nye+1)
          thy : (nxe+1) x (nye+2)
        """
        u_c = np.asarray(u_c)

        nxw_c, nyw_c = nxe_c + 2, nye_c + 2
        nxtx_c, nytx_c = nxe_c + 2, nye_c + 1
        nxty_c, nyty_c = nxe_c + 1, nye_c + 2

        nw_c = nxw_c * nyw_c
        ntx_c = nxtx_c * nytx_c
        nty_c = nxty_c * nyty_c
        nu_c = nty_c
        nv_c = ntx_c
        assert u_c.size == nw_c + nu_c + nv_c + ntx_c + nty_c

        w_c = u_c[:nw_c]
        U_c = u_c[nw_c:(nw_c+nu_c)]
        V_c = u_c[(nw_c+nu_c):(nw_c+nu_c+nv_c)]
        tx_c = u_c[(nw_c+nu_c+nv_c):(nw_c+nu_c+nv_c+ntx_c)]
        ty_c = u_c[(nw_c+nu_c+nv_c+ntx_c):]

        # Build restriction operators and take P = R^T
        Rx2 = self._build_R_p2(nxe_c)
        Ry2 = self._build_R_p2(nye_c)
        Rx1 = self._build_P_p1(nxe_c).T
        Ry1 = self._build_P_p1(nye_c).T

        # w: (p2,p2)
        Rw = self._kron2(Ry2, Rx2)
        Pw = Rw.T

        # thx: (p2 in x, p1 in y)
        Rtx = self._kron2(Ry1, Rx2)
        Ptx = Rtx.T

        # thy: (p1 in x, p2 in y)
        Rty = self._kron2(Ry2, Rx1)
        Pty = Rty.T

        w_f  = Pw @ w_c
        U_f  = Pty @ U_c
        V_f  = Ptx @ V_c
        tx_f = Ptx @ tx_c
        ty_f = Pty @ ty_c

        # sizes fine
        nxe_f, nye_f = 2*nxe_c, 2*nye_c
        nxw_f, nyw_f = nxe_f + 2, nye_f + 2
        nxtx_f, nytx_f = nxe_f + 2, nye_f + 1
        nxty_f, nyty_f = nxe_f + 1, nye_f + 2

        u_f = np.concatenate([w_f, U_f, V_f, tx_f, ty_f])
        u_f = self.apply_bcs_2d(u_f, nxw_f, nyw_f, nxtx_f, nytx_f, nxty_f, nyty_f) #, mode="prolong")
        return u_f

    def restrict_defect(self, r_f: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        Restrict fine defect -> coarse defect for dyadic refinement.
        Fine has nxe_f=2*nxe_c, nye_f=2*nye_c.
        """
        r_f = np.asarray(r_f)

        nxe_f, nye_f = 2*nxe_c, 2*nye_c

        nxw_f, nyw_f = nxe_f + 2, nye_f + 2
        nxtx_f, nytx_f = nxe_f + 2, nye_f + 1
        nxty_f, nyty_f = nxe_f + 1, nye_f + 2

        nw_f = nxw_f * nyw_f
        ntx_f = nxtx_f * nytx_f
        nty_f = nxty_f * nyty_f
        nu_f = nty_f
        nv_f = ntx_f
        assert r_f.size == nw_f + nu_f + nv_f + ntx_f + nty_f

        w_f = r_f[:nw_f]
        U_f = r_f[nw_f:(nw_f+nu_f)]
        V_f = r_f[(nw_f+nu_f):(nw_f+nu_f+nv_f)]
        tx_f = r_f[(nw_f+nu_f+nv_f):(nw_f+nu_f+nv_f+ntx_f)]
        ty_f = r_f[(nw_f+nu_f+nv_f+ntx_f):]

        # restriction ops
        Rx2 = self._build_R_p2(nxe_c)
        Ry2 = self._build_R_p2(nye_c)
        Rx1 = self._build_R_p1(nxe_c, is_prolong=False)
        Ry1 = self._build_R_p1(nye_c, is_prolong=False)

        Rw  = self._kron2(Ry2, Rx2)
        Rtx = self._kron2(Ry2, Rx1)
        Rty = self._kron2(Ry1, Rx2)

        w_c  = Rw  @ w_f
        U_c  = Rty @ U_f
        V_c  = Rtx @ V_f
        tx_c = Rtx @ tx_f
        ty_c = Rty @ ty_f

        nxw_c, nyw_c = nxe_c + 2, nye_c + 2
        nxtx_c, nytx_c = nxe_c + 2, nye_c + 1
        nxty_c, nyty_c = nxe_c + 1, nye_c + 2

        r_c = np.concatenate([w_c, U_c, V_c, tx_c, ty_c])
        r_c = self.apply_bcs_2d(r_c, nxw_c, nyw_c, nxtx_c, nytx_c, nxty_c, nyty_c) #, mode="restrict")
        return r_c