# make a new mixed finite element drig element with different order IGA 

import numpy as np
from .basis import (
    second_order_quadrature,
    first_order_quadrature,
    zero_order_quadrature,
    get_iga2_basis,
    get_lagrange_basis_01,
    third_order_quadrature,
    get_iga3_basis,
)

class MixedDeRhamIGACylinderElement:
    """
    p=3 cylinder element with aux w2 in (p3,p1) and a *separate* weak constraint equation
    living in the *w2 row* (no SPD penalty energy, no extra lambda DOF).

    Unknown ordering (local block order):
      Ue = [ w(16), w2(8), u(12), v(12), thx(12), thy(12) ]
    
    NOTE : this element was derived to fix mem locking in the cylinder..
    See this ref for another derivation of cylinder mem locking and the energy
        https://link.springer.com/article/10.1007/BF01385524
        The problem of membrane locking in finite element analysis of cylindrical shells

    Spaces:
      w    : (p3,p3)  -> 16
      w2   : (p3,p1)  ->  8   (matches v_y which is (p3,p1) since v is (p3,p2))
      u    : (p2,p3)  -> 12
      v    : (p3,p2)  -> 12
      thx  : (p3,p2)  -> 12
      thy  : (p2,p3)  -> 12

    Strains:
      same as your cylinder, except you can switch e22 to use w2:
        e22 = v_y + w2/r

    Constraint equation (assembled ONLY into the w2 row):
      ∫ Nw2 * (w2 - w) dA = 0

    Implementation: add to stiffness blocks
      Kw2w2 += alpha_c * ∫ Nw2^T Nw2 dA
      Kw2w  += -alpha_c * ∫ Nw2^T (Iw_to_w2 * Nw) dA

    where Iw_to_w2 maps w's (p3,p3) shape functions down to the w2 test space in y.
    Practically: evaluate Nw at quad, then *restrict in y* with a 1D operator Ry31 (p3->p1).
    I leave that operator as a stub for you to fill.
    """

    def __init__(self, r: float, reduced_integrated: bool=False, clamped: bool=False,
                 axial_factor: float=0.0, curvature_on: bool=True,
                 use_w2_in_e22: bool=True, alpha_c: float=1.0):
        self.r = float(r)
        self.reduced_integrated = bool(reduced_integrated)
        self.clamped = bool(clamped)
        self.axial_factor = float(axial_factor)
        self.curvature_on = bool(curvature_on)
        self.use_w2_in_e22 = bool(use_w2_in_e22)
        self.alpha_c = float(alpha_c)

    # ---- tensor helpers ------------------------------------------------------
    @staticmethod
    def _tensor_product_basis(xi, eta, bx, by):
        Nx, dNx = bx
        Ny, dNy = by
        N    = np.kron(Ny, Nx)
        Nxi  = np.kron(Ny, dNx)
        Neta = np.kron(dNy, Nx)
        return N, Nxi, Neta

    # ---- 1D bases (YOU fill iga3 + quad) ------------------------------------
    @staticmethod
    def _iga3_1d(x, ixe, nxe):
        return get_iga3_basis(x, ixe, nxe)

    @staticmethod
    def _iga2_1d(x, ixe, nxe):
        left = ixe == 0; right = ixe == nxe-1
        return get_iga2_basis(x, left, right)

    @staticmethod
    def _p1_1d(x):
        return get_lagrange_basis_01(x)

    def get_kelem(self, E, nu, thick, dx, dy, ixe, nxe, iye, nye):
        """
        Returns 6x6 block stiffness (dense) for local dofs ordered:
          [ w(16), w2(8), u(12), v(12), thx(12), thy(12) ]
        """
        # pts, wts = third_order_quadrature()
        pts, wts = third_order_quadrature()

        r = self.r

        # sizes
        n_w   = 16  # (p3,p3)
        n_w2  = 8   # (p3,p1)
        n_u   = 12  # (p2,p3)
        n_v   = 12  # (p3,p2)
        n_thx = 12  # (p3,p2)
        n_thy = 12  # (p2,p3)

        def Z(a,b): return np.zeros((a,b))

        # blocks: (w, w2, u, v, thx, thy)
        Kww   = Z(n_w,   n_w)
        Kww2  = Z(n_w,   n_w2)
        Kwu   = Z(n_w,   n_u)
        Kwv   = Z(n_w,   n_v)
        Kwtx  = Z(n_w,   n_thx)
        Kwty  = Z(n_w,   n_thy)

        Kw2w  = Z(n_w2,  n_w)
        Kw2w2 = Z(n_w2,  n_w2)
        Kw2u  = Z(n_w2,  n_u)
        Kw2v  = Z(n_w2,  n_v)
        Kw2tx = Z(n_w2,  n_thx)
        Kw2ty = Z(n_w2,  n_thy)

        Kuw   = Z(n_u,   n_w)
        Kuw2  = Z(n_u,   n_w2)
        Kuu   = Z(n_u,   n_u)
        Kuv   = Z(n_u,   n_v)
        Kutx  = Z(n_u,   n_thx)
        Kuty  = Z(n_u,   n_thy)

        Kvw   = Z(n_v,   n_w)
        Kvw2  = Z(n_v,   n_w2)
        Kvu   = Z(n_v,   n_u)
        Kvv   = Z(n_v,   n_v)
        Kvtx  = Z(n_v,   n_thx)
        Kvty  = Z(n_v,   n_thy)

        Ktxw  = Z(n_thx, n_w)
        Ktxw2 = Z(n_thx, n_w2)
        Ktxu  = Z(n_thx, n_u)
        Ktxv  = Z(n_thx, n_v)
        Ktxtx = Z(n_thx, n_thx)
        Ktxty = Z(n_thx, n_thy)

        Ktyw  = Z(n_thy, n_w)
        Ktyw2 = Z(n_thy, n_w2)
        Ktyu  = Z(n_thy, n_u)
        Ktyv  = Z(n_thy, n_v)
        Ktytx = Z(n_thy, n_thx)
        Ktyty = Z(n_thy, n_thy)

        # ------------------- material matrices -------------------
        EI = E * thick**3 / (12.0 * (1.0 - nu**2))
        Db = EI * np.array([
            [1.0, nu, 0.0],
            [nu,  1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])

        A0 = E * thick / (1.0 - nu**2)
        Dm = A0 * np.array([
            [1.0, nu, 0.0],
            [nu,  1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])

        ks = 5.0 / 6.0
        G  = E / (2.0 * (1.0 + nu))
        Ds = (ks * G * thick) * np.eye(2)

        Jdet = dx * dy
        xi_x  = 1.0 / dx
        eta_y = 1.0 / dy

        debug_curv_off = not self.curvature_on
        half_eng_strains = False

        # ==========================================================
        # BENDING (same structure as your code; just bases are p3/p2)
        # ==========================================================
        for _eta, w_eta_raw in zip(pts, wts):
            w_eta = 0.5 * w_eta_raw
            eta = 0.5 * (_eta + 1.0)

            Ny3, dNy3 = self._iga3_1d(eta, iye, nye)  # p3
            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)  # p2

            for _xi, w_xi_raw in zip(pts, wts):
                w_xi = 0.5 * w_xi_raw
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * Jdet
                cB = wt

                Nx3, dNx3 = self._iga3_1d(xi, ixe, nxe)  # p3
                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)  # p2

                # w: (p3,p3)
                Nw, Nw_xi, Nw_eta = self._tensor_product_basis(xi, eta, (Nx3, dNx3), (Ny3, dNy3))
                Nw_x = Nw_xi * xi_x
                Nw_y = Nw_eta * eta_y

                # v, thx: (p3,p2)
                Nv, Nv_xi, Nv_eta = self._tensor_product_basis(xi, eta, (Nx3, dNx3), (Ny2, dNy2))
                Nv_x = Nv_xi * xi_x
                Nv_y = Nv_eta * eta_y
                Ntx, Ntx_x, Ntx_y = Nv, Nv_x, Nv_y

                # u, thy: (p2,p3)
                Nty, Nty_xi, Nty_eta = self._tensor_product_basis(xi, eta, (Nx2, dNx2), (Ny3, dNy3))
                Nty_x = Nty_xi * xi_x
                Nty_y = Nty_eta * eta_y

                # k11 = thy_x
                k11_thy = Nty_x

                # k22 = -thx_y + w/r^2 + v_y/r
                k22_w   = (1.0/(r*r)) * Nw
                k22_thx = -Ntx_y
                k22_v   = (1.0/r) * Nv_y

                # 2*k12 = v_x/r + thy_y - thx_x
                k12_v   = (1.0/r) * Nv_x
                k12_thy = Nty_y
                k12_thx = -Ntx_x

                if debug_curv_off:
                    k22_w = 0.0
                    k22_v = 0.0
                    k12_v = 0.0

                if half_eng_strains:
                    k12_v *= 0.5
                    k12_thy *= 0.5
                    k12_thx *= 0.5

                D11, D12, D22, D33 = Db[0,0], Db[0,1], Db[1,1], Db[2,2]

                Ktyty += cB * (D11 * np.outer(k11_thy, k11_thy))

                Kww   += cB * (D22 * np.outer(k22_w,   k22_w))
                Kvv   += cB * (D22 * np.outer(k22_v,   k22_v))
                Ktxtx += cB * (D22 * np.outer(k22_thx, k22_thx))

                Kwv   += cB * (D22 * np.outer(k22_w,   k22_v))
                Kvw   += cB * (D22 * np.outer(k22_v,   k22_w))
                Kwtx  += cB * (D22 * np.outer(k22_w,   k22_thx))
                Ktxw  += cB * (D22 * np.outer(k22_thx, k22_w))
                Kvtx  += cB * (D22 * np.outer(k22_v,   k22_thx))
                Ktxv  += cB * (D22 * np.outer(k22_thx, k22_v))

                Kvv   += cB * (D33 * np.outer(k12_v,   k12_v))
                Ktxtx += cB * (D33 * np.outer(k12_thx, k12_thx))
                Ktyty += cB * (D33 * np.outer(k12_thy, k12_thy))

                Kvtx  += cB * (D33 * np.outer(k12_v,   k12_thx))
                Ktxv  += cB * (D33 * np.outer(k12_thx, k12_v))
                Kvty  += cB * (D33 * np.outer(k12_v,   k12_thy))
                Ktyv  += cB * (D33 * np.outer(k12_thy, k12_v))
                Ktxty += cB * (D33 * np.outer(k12_thx, k12_thy))
                Ktytx += cB * (D33 * np.outer(k12_thy, k12_thx))

                Kwty  += cB * (D12 * np.outer(k22_w,   k11_thy))
                Ktyw  += cB * (D12 * np.outer(k11_thy, k22_w))
                Kvty  += cB * (D12 * np.outer(k22_v,   k11_thy))
                Ktyv  += cB * (D12 * np.outer(k11_thy, k22_v))
                Ktxty += cB * (D12 * np.outer(k22_thx, k11_thy))
                Ktytx += cB * (D12 * np.outer(k11_thy, k22_thx))

        # ==========================================================
        # SHEAR (same form, new bases)
        # ==========================================================
        ns = len(pts)
        for jj in range(ns):
            _eta = pts[jj]
            w_eta = wts[jj] * 0.5
            eta = 0.5 * (_eta + 1.0)

            Ny3, dNy3 = self._iga3_1d(eta, iye, nye)  # w uses p3
            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)  # v/thx uses p2 in y
            # u/thy uses p3 in y (handled below)

            for ii in range(ns):
                _xi = pts[ii]
                w_xi = wts[ii] * 0.5
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * Jdet

                Nx3, dNx3 = self._iga3_1d(xi, ixe, nxe)
                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)

                # w: (p3,p3)
                Nw, Nw_xi, Nw_eta = self._tensor_product_basis(xi, eta, (Nx3, dNx3), (Ny3, dNy3))
                Nw_x = Nw_xi * xi_x
                Nw_y = Nw_eta * eta_y

                # v, thx: (p3,p2)
                Nv, _, _ = self._tensor_product_basis(xi, eta, (Nx3, dNx3), (Ny2, dNy2))
                Ntx = Nv

                # thy: (p2,p3)
                Nty, _, _ = self._tensor_product_basis(xi, eta, (Nx2, dNx2), (Ny3, dNy3))

                # gamma1 = w_x + thy
                g1_w   = Nw_x
                g1_thy = Nty

                # gamma2 = w_y - v/r - thx
                g2_w   = Nw_y
                g2_v   = -(1.0 / r) * Nv
                g2_thx = -Ntx

                if debug_curv_off:
                    g2_v = 0.0

                Ds11 = Ds[0, 0]
                Ds22 = Ds[1, 1]

                # g1
                Kww   += wt * (Ds11 * np.outer(g1_w,   g1_w))
                Kwty  += wt * (Ds11 * np.outer(g1_w,   g1_thy))
                Ktyw  += wt * (Ds11 * np.outer(g1_thy, g1_w))
                Ktyty += wt * (Ds11 * np.outer(g1_thy, g1_thy))

                # g2
                Kww   += wt * (Ds22 * np.outer(g2_w,   g2_w))
                Kvv   += wt * (Ds22 * np.outer(g2_v,   g2_v))
                Ktxtx += wt * (Ds22 * np.outer(g2_thx, g2_thx))

                Kwv   += wt * (Ds22 * np.outer(g2_w,   g2_v))
                Kvw   += wt * (Ds22 * np.outer(g2_v,   g2_w))
                Kwtx  += wt * (Ds22 * np.outer(g2_w,   g2_thx))
                Ktxw  += wt * (Ds22 * np.outer(g2_thx, g2_w))
                Kvtx  += wt * (Ds22 * np.outer(g2_v,   g2_thx))
                Ktxv  += wt * (Ds22 * np.outer(g2_thx, g2_v))

        # ==========================================================
        # MEMBRANE (unchanged energy form, but optionally use w2 in e22)
        # ==========================================================
        for _eta, w_eta_raw in zip(pts, wts):
            w_eta = 0.5 * w_eta_raw
            eta = 0.5 * (_eta + 1.0)

            Ny3, dNy3 = self._iga3_1d(eta, iye, nye)  # p3
            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)  # p2
            Ny1, dNy1 = self._p1_1d(eta)                          # p1 (for w2)

            for _xi, w_xi_raw in zip(pts, wts):
                w_xi = 0.5 * w_xi_raw
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * Jdet
                cM = wt

                Nx3, dNx3 = self._iga3_1d(xi, ixe, nxe)
                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)

                # w: (p3,p3)
                Nw, Nw_xi, Nw_eta = self._tensor_product_basis(xi, eta, (Nx3, dNx3), (Ny3, dNy3))
                Nw_x = Nw_xi * xi_x
                Nw_y = Nw_eta * eta_y

                # w2: (p3,p1)
                Nw2, _, _ = self._tensor_product_basis(xi, eta, (Nx3, dNx3), (Ny1, dNy1))

                # v: (p3,p2)
                Nv, Nv_xi, Nv_eta = self._tensor_product_basis(xi, eta, (Nx3, dNx3), (Ny2, dNy2))
                Nv_x = Nv_xi * xi_x
                Nv_y = Nv_eta * eta_y

                # u, thy: (p2,p3)
                Nu, Nu_xi, Nu_eta = self._tensor_product_basis(xi, eta, (Nx2, dNx2), (Ny3, dNy3))
                Nu_x = Nu_xi * xi_x
                Nu_y = Nu_eta * eta_y

                # e11 = u_x
                e11_u = Nu_x

                # e22 = v_y + (w or w2)/r
                e22_v = Nv_y
                if self.use_w2_in_e22:
                    e22_wlike = (1.0 / r) * Nw2     # goes into w2 column/row
                else:
                    e22_wlike = (1.0 / r) * Nw      # original

                # 2e12 = v_x + u_y
                e12_v = Nv_x
                e12_u = Nu_y

                if debug_curv_off:
                    e22_wlike = 0.0

                D11, D12, D22, D33 = Dm[0,0], Dm[0,1], Dm[1,1], Dm[2,2]

                # e11-e11
                Kuu += cM * (D11 * np.outer(e11_u, e11_u))

                # e22-e22
                Kvv += cM * (D22 * np.outer(e22_v, e22_v))

                if self.use_w2_in_e22:
                    # contribute to w2 blocks instead of w blocks
                    Kw2w2 += cM * (D22 * np.outer(e22_wlike, e22_wlike))
                    Kvw2  += cM * (D22 * np.outer(e22_v,     e22_wlike))
                    Kw2v  += cM * (D22 * np.outer(e22_wlike, e22_v))
                else:
                    Kww += cM * (D22 * np.outer(e22_wlike, e22_wlike))
                    Kvw += cM * (D22 * np.outer(e22_v,     e22_wlike))
                    Kwv += cM * (D22 * np.outer(e22_wlike, e22_v))

                # (2e12)-(2e12)
                Kvv += cM * (D33 * np.outer(e12_v, e12_v))
                Kuu += cM * (D33 * np.outer(e12_u, e12_u))
                Kvu += cM * (D33 * np.outer(e12_v, e12_u))
                Kuv += cM * (D33 * np.outer(e12_u, e12_v))

                # e11-e22 coupling (nu terms)
                Kvu += cM * (D12 * np.outer(e22_v, e11_u))
                Kuv += cM * (D12 * np.outer(e11_u, e22_v))

                if self.use_w2_in_e22:
                    Kw2u += cM * (D12 * np.outer(e22_wlike, e11_u))
                    Kuw2 += cM * (D12 * np.outer(e11_u,     e22_wlike))
                else:
                    Kwu += cM * (D12 * np.outer(e22_wlike, e11_u))
                    Kuw += cM * (D12 * np.outer(e11_u,     e22_wlike))

        # ==========================================================
        # CONSTRAINT ROW (ONLY w2 row):  ∫ Nw2 * (w2 - w) dA = 0
        # ==========================================================
        alpha_c = self.alpha_c
        for _eta, w_eta_raw in zip(pts, wts):
            w_eta = 0.5 * w_eta_raw
            eta = 0.5 * (_eta + 1.0)

            Ny3, dNy3 = self._iga3_1d(eta, iye, nye)
            Ny1, dNy1 = self._p1_1d(eta)

            for _xi, w_xi_raw in zip(pts, wts):
                w_xi = 0.5 * w_xi_raw
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * Jdet

                Nx3, dNx3 = self._iga3_1d(xi, ixe, nxe)

                # Nw2: (p3,p1) -> len 8
                Nw2, _, _ = self._tensor_product_basis(xi, eta, (Nx3, dNx3), (Ny1, dNy1))

                # Nw: (p3,p3) -> len 16
                Nw, _, _  = self._tensor_product_basis(xi, eta, (Nx3, dNx3), (Ny3, dNy3))

                # row-only equation:
                # Kw2w2 += α ∫ Nw2^T Nw2
                Kw2w2 += alpha_c * wt * np.outer(Nw2, Nw2)

                # Kw2w  += -α ∫ Nw2^T Nw
                Kw2w  += -alpha_c * wt * np.outer(Nw2, Nw)

                # DO NOT add transpose block Kww2 (keep row-only / non-symmetric)


        # Return blocks in order (w, w2, u, v, thx, thy)
        return (
            Kww,  Kww2, Kwu,  Kwv,  Kwtx,  Kwty,
            Kw2w, Kw2w2, Kw2u, Kw2v, Kw2tx, Kw2ty,
            Kuw,  Kuw2, Kuu,  Kuv,  Kutx,  Kuty,
            Kvw,  Kvw2, Kvu,  Kvv,  Kvtx,  Kvty,
            Ktxw, Ktxw2, Ktxu, Ktxv, Ktxtx, Ktxty,
            Ktyw, Ktyw2, Ktyu, Ktyv, Ktytx, Ktyty
        )

        # ==========================================================
    # Loads
    # ==========================================================
    def get_felem(
        self,
        load_fcn,     # callable q(x,y) acting on w only
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        ixe:int, nxe:int,
        iye:int, nye:int,
    ):
        """
        Consistent load vector for q(x,y) acting on w only (and optional axial_factor on u):

        Unknown ordering (local):
          [ w(16), w2(8), u(12), v(12), thx(12), thy(12) ]

        Returns:
          fw(16), fw2(8), fu(12), fv(12), ftx(12), fty(12)
        """
        # pts, wts = third_order_quadrature()
        pts, wts = third_order_quadrature()

        fw  = np.zeros(16)
        fw2 = np.zeros(8)    # typically zero (no direct load on w2)
        fu  = np.zeros(12)
        fv  = np.zeros(12)
        ftx = np.zeros(12)
        fty = np.zeros(12)

        Jdet = dx * dy

        for _eta, w_eta_raw in zip(pts, wts):
            w_eta = 0.5 * w_eta_raw
            eta = 0.5 * (_eta + 1.0)

            Ny3, dNy3 = self._iga3_1d(eta, iye, nye)  # p3
            # For u/thy (p2,p3) -> Ny3 in y, Nx2 in x
            # For v/thx (p3,p2) -> Ny2 in y, Nx3 in x
            # For w2 (p3,p1) -> Ny1 in y, Nx3 in x (but no load)

            for _xi, w_xi_raw in zip(pts, wts):
                w_xi = 0.5 * w_xi_raw
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * Jdet

                Nx3, dNx3 = self._iga3_1d(xi, ixe, nxe)  # p3
                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)  # p2

                # w basis: (p3,p3) -> 16
                Nw, _, _ = self._tensor_product_basis(xi, eta, (Nx3, dNx3), (Ny3, dNy3))

                # u basis (if you want axial_factor forcing): (p2,p3) -> 12
                Nu, _, _ = self._tensor_product_basis(xi, eta, (Nx2, dNx2), (Ny3, dNy3))

                xq = x0 + xi * dx
                yq = y0 + eta * dy
                q = float(load_fcn(xq, yq))

                fw += (q * Nw) * wt
                if self.axial_factor != 0.0:
                    fu += (q * Nu) * wt * self.axial_factor

        return fw, fw2, fu, fv, ftx, fty

    # -------------------------------------------------------------------------
    # Multigrid transfers (dyadic refinement) for [w, w2, u, v, thx, thy]
    # -------------------------------------------------------------------------

    @staticmethod
    def _kron2(Ry: np.ndarray, Rx: np.ndarray) -> np.ndarray:
        return np.kron(Ry, Rx)

    def apply_bcs_2d(self, u: np.ndarray,
                     nxw: int, nyw: int,
                     nxw2: int, nyw2: int,
                     nxu: int, nyu: int,
                     nxv: int, nyv: int,
                     nxtx: int, nytx: int,
                     nxty: int, nyty: int):
        """
        Strong essential BC projector/substitution for the mixed element.

        Global vector layout (fine or coarse, consistent with caller):
          [ w, w2, u, v, thx, thy ]

        Simply supported (default):
          - enforce w = 0 on boundary (on w grid)
          - keep others free except "opposite supports" like before:
              u = 0 on x=0 edge (on u grid)
              v = 0 on y=0 edge (on v grid)

        Clamped:
          - w=0 on boundary
          - u=v=0 on boundary (on their grids)
          - thx=thy=0 on boundary (on their grids)
          - (optional) also w2=0 on boundary if you want; I leave it OFF by default
            because w2 is an auxiliary constrained by weak equation.
        """
        u = np.asarray(u)

        nw   = nxw  * nyw
        nw2  = nxw2 * nyw2
        nu   = nxu  * nyu
        nv   = nxv  * nyv
        ntx  = nxtx * nytx
        nty  = nxty * nyty

        assert u.size == (nw + nw2 + nu + nv + ntx + nty)

        w  = u[:nw]
        w2 = u[nw:(nw+nw2)]
        U  = u[(nw+nw2):(nw+nw2+nu)]
        V  = u[(nw+nw2+nu):(nw+nw2+nu+nv)]
        tx = u[(nw+nw2+nu+nv):(nw+nw2+nu+nv+ntx)]
        ty = u[(nw+nw2+nu+nv+ntx):]

        def on_bndry(i, j, nx, ny):
            return (i == 0) or (i == nx-1) or (j == 0) or (j == ny-1)

        # w boundary
        for j in range(nyw):
            for i in range(nxw):
                if on_bndry(i, j, nxw, nyw):
                    w[i + nxw*j] = 0.0

        if self.clamped:
            # u boundary on its grid
            for j in range(nyu):
                for i in range(nxu):
                    if on_bndry(i, j, nxu, nyu):
                        U[i + nxu*j] = 0.0

            # v boundary on its grid
            for j in range(nyv):
                for i in range(nxv):
                    if on_bndry(i, j, nxv, nyv):
                        V[i + nxv*j] = 0.0

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
        else:
            # u boundary: x=0 edge on u grid
            for j in range(nyu):
                for i in range(nxu):
                    if i == 0:
                        U[i + nxu*j] = 0.0

            # v boundary: y=0 edge on v grid
            for j in range(nyv):
                for i in range(nxv):
                    if j == 0:
                        V[i + nxv*j] = 0.0

        return np.concatenate([w, w2, U, V, tx, ty])
    
    # -------------------------------------------------------------------------
    # Multigrid transfers (dyadic refinement) for [w, u, v, thx, thy]
    # -------------------------------------------------------------------------
    def _build_R_p3(self, nxe_c: int) -> np.ndarray:
        """
        1D restriction for p=3 IGA line.

        Coarse dofs: n_c = nxe_c + 3
        Fine   dofs: n_f = 2*nxe_c + 3

        YOU fill this in later (could be full-weighting / variational / knot-insertion).
        """

        n_w_c = nxe_c + 3
        nxe_f = 2 * nxe_c
        n_w_f = nxe_f + 3

        R = np.zeros((n_w_c, n_w_f))

        for ielem in range(nxe_c):

            # choose local restriction based on coarse element index
            if ielem == 0:
                R_loc = np.array([
                    [1.0   , 0.5   , 0.0   , 0.0   , 0.0],
                    [0.0   , 0.5   , 0.75  , 0.1875, 0.0],
                    [0.0   , 0.0   , 0.25  , 0.6875, 0.5],
                    [0.0   , 0.0   , 0.0   , 0.125 , 0.5]
                ])

            elif ielem == 1:
                R_loc = np.array([
                    [0.75  , 0.1875, 0.0   , 0.0   , 0.0],
                    [0.25  , 0.6875, 0.5   , 0.125 , 0.0],
                    [0.0   , 0.125 , 0.5   , 0.75  , 0.5],
                    [0.0   , 0.0   , 0.0   , 0.125 , 0.5]
                ])

            elif ielem == nxe_c - 2:  # second to last
                R_loc = np.array([
                    [0.5   , 0.125 , 0.0   , 0.0   , 0.0],
                    [0.5   , 0.75  , 0.5   , 0.125 , 0.0],
                    [0.0   , 0.125 , 0.5   , 0.6875, 0.25],
                    [0.0   , 0.0   , 0.0   , 0.1875, 0.75]
                ])

            elif ielem == nxe_c - 1:  # last element
                R_loc = np.array([
                    [0.5   , 0.125 , 0.0   , 0.0   , 0.0],
                    [0.5   , 0.6875, 0.25  , 0.0   , 0.0],
                    [0.0   , 0.1875, 0.75  , 0.5   , 0.0],
                    [0.0   , 0.0   , 0.0   , 0.5   , 1.0]
                ])

            else:  # interior
                R_loc = np.array([
                    [0.5   , 0.125 , 0.0   , 0.0   , 0.0],
                    [0.5   , 0.75  , 0.5   , 0.125 , 0.0],
                    [0.0   , 0.125 , 0.5   , 0.75  , 0.5],
                    [0.0   , 0.0   , 0.0   , 0.125 , 0.5]
                ])

            # add in coarse elem restriction
            R[ielem:(ielem+4), 2*ielem:(2*ielem+5)] += R_loc

        R /= np.sum(R, axis=0) # normalize if prolongated, results in right values usually
        return R

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

    def prolongate(self, u_c: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        Dyadic prolongation for the mixed p=3 element.

        Coarse element counts: (nxe_c, nye_c)
        Fine   element counts: (2*nxe_c, 2*nye_c)

        Grid sizes:
          w   : (nxe+3) x (nye+3)   (p3,p3)
          w2  : (nxe+3) x (nye+1)   (p3,p1)
          u   : (nxe+2) x (nye+3)   (p2,p3)
          v   : (nxe+3) x (nye+2)   (p3,p2)
          thx : (nxe+3) x (nye+2)   (p3,p2)
          thy : (nxe+2) x (nye+3)   (p2,p3)
        """
        u_c = np.asarray(u_c)

        # coarse sizes
        nxw_c,  nyw_c  = nxe_c + 3, nye_c + 3
        nxw2_c, nyw2_c = nxe_c + 3, nye_c + 1
        nxu_c,  nyu_c  = nxe_c + 2, nye_c + 3
        nxv_c,  nyv_c  = nxe_c + 3, nye_c + 2
        nxtx_c, nytx_c = nxv_c,     nyv_c
        nxty_c, nyty_c = nxu_c,     nyu_c

        nw_c  = nxw_c  * nyw_c
        nw2_c = nxw2_c * nyw2_c
        nu_c  = nxu_c  * nyu_c
        nv_c  = nxv_c  * nyv_c
        ntx_c = nxtx_c * nytx_c
        nty_c = nxty_c * nyty_c

        assert u_c.size == (nw_c + nw2_c + nu_c + nv_c + ntx_c + nty_c)

        off = 0
        w_c  = u_c[off:off+nw_c];   off += nw_c
        w2_c = u_c[off:off+nw2_c];  off += nw2_c
        U_c  = u_c[off:off+nu_c];   off += nu_c
        V_c  = u_c[off:off+nv_c];   off += nv_c
        tx_c = u_c[off:off+ntx_c];  off += ntx_c
        ty_c = u_c[off:off+nty_c];  off += nty_c

        # --- build 1D prolongations ---
        # p3 (you fill):
        Rx3 = self._build_R_p3(nxe_c)
        Ry3 = self._build_R_p3(nye_c)
        Px3 = Rx3.T
        Py3 = Ry3.T

        # p2 (reuse your p2 restriction then transpose, OR implement a true p2 prolongation)
        # Here: use your existing p2 restriction builder but treat P = R^T.
        Rx2 = self._build_R_p2(nxe_c)
        Ry2 = self._build_R_p2(nye_c)
        Px2 = Rx2.T
        Py2 = Ry2.T

        # p1 prolongation exists:
        Px1 = self._build_P_p1(nxe_c)
        Py1 = self._build_P_p1(nye_c)

        # --- 2D prolongations (kron) ---
        # w: (p3,p3)
        Pw  = self._kron2(Py3, Px3)

        # w2: (p3,p1)
        Pw2 = self._kron2(Py1, Px3)

        # u, thy: (p2,p3)
        Pu  = self._kron2(Py3, Px2)
        Pty = Pu

        # v, thx: (p3,p2)
        Pv  = self._kron2(Py2, Px3)
        Ptx = Pv

        w_f  = Pw  @ w_c
        w2_f = Pw2 @ w2_c
        U_f  = Pu  @ U_c
        V_f  = Pv  @ V_c
        tx_f = Ptx @ tx_c
        ty_f = Pty @ ty_c

        # fine sizes
        nxe_f, nye_f = 2*nxe_c, 2*nye_c

        nxw_f,  nyw_f  = nxe_f + 3, nye_f + 3
        nxw2_f, nyw2_f = nxe_f + 3, nye_f + 1
        nxu_f,  nyu_f  = nxe_f + 2, nye_f + 3
        nxv_f,  nyv_f  = nxe_f + 3, nye_f + 2
        nxtx_f, nytx_f = nxv_f,     nyv_f
        nxty_f, nyty_f = nxu_f,     nyu_f

        u_f = np.concatenate([w_f, w2_f, U_f, V_f, tx_f, ty_f])
        u_f = self.apply_bcs_2d(
            u_f,
            nxw_f, nyw_f,
            nxw2_f, nyw2_f,
            nxu_f, nyu_f,
            nxv_f, nyv_f,
            nxtx_f, nytx_f,
            nxty_f, nyty_f,
        )
        return u_f

    def restrict_defect(self, r_f: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        Restrict fine defect -> coarse defect for dyadic refinement
        for the mixed p=3 element.

        Fine has (nxe_f, nye_f) = (2*nxe_c, 2*nye_c).
        """
        r_f = np.asarray(r_f)

        nxe_f, nye_f = 2*nxe_c, 2*nye_c

        # fine sizes
        nxw_f,  nyw_f  = nxe_f + 3, nye_f + 3
        nxw2_f, nyw2_f = nxe_f + 3, nye_f + 1
        nxu_f,  nyu_f  = nxe_f + 2, nye_f + 3
        nxv_f,  nyv_f  = nxe_f + 3, nye_f + 2
        nxtx_f, nytx_f = nxv_f,     nyv_f
        nxty_f, nyty_f = nxu_f,     nyu_f

        nw_f  = nxw_f  * nyw_f
        nw2_f = nxw2_f * nyw2_f
        nu_f  = nxu_f  * nyu_f
        nv_f  = nxv_f  * nyv_f
        ntx_f = nxtx_f * nytx_f
        nty_f = nxty_f * nyty_f

        assert r_f.size == (nw_f + nw2_f + nu_f + nv_f + ntx_f + nty_f)

        off = 0
        w_f  = r_f[off:off+nw_f];   off += nw_f
        w2_f = r_f[off:off+nw2_f];  off += nw2_f
        U_f  = r_f[off:off+nu_f];   off += nu_f
        V_f  = r_f[off:off+nv_f];   off += nv_f
        tx_f = r_f[off:off+ntx_f];  off += ntx_f
        ty_f = r_f[off:off+nty_f];  off += nty_f

        # 1D restrictions
        Rx3 = self._build_R_p3(nxe_c)
        Ry3 = self._build_R_p3(nye_c)

        Rx2 = self._build_R_p2(nxe_c)
        Ry2 = self._build_R_p2(nye_c)

        Rx1 = self._build_R_p1(nxe_c, is_prolong=False)
        Ry1 = self._build_R_p1(nye_c, is_prolong=False)

        # 2D restrictions
        Rw  = self._kron2(Ry3, Rx3)   # w: (p3,p3)
        Rw2 = self._kron2(Ry1, Rx3)   # w2: (p3,p1)
        Ru  = self._kron2(Ry3, Rx2)   # u:  (p2,p3)
        Rv  = self._kron2(Ry2, Rx3)   # v:  (p3,p2)
        Rtx = Rv
        Rty = Ru

        w_c  = Rw  @ w_f
        w2_c = Rw2 @ w2_f
        U_c  = Ru  @ U_f
        V_c  = Rv  @ V_f
        tx_c = Rtx @ tx_f
        ty_c = Rty @ ty_f

        # coarse sizes
        nxw_c,  nyw_c  = nxe_c + 3, nye_c + 3
        nxw2_c, nyw2_c = nxe_c + 3, nye_c + 1
        nxu_c,  nyu_c  = nxe_c + 2, nye_c + 3
        nxv_c,  nyv_c  = nxe_c + 3, nye_c + 2
        nxtx_c, nytx_c = nxv_c,     nyv_c
        nxty_c, nyty_c = nxu_c,     nyu_c

        r_c = np.concatenate([w_c, w2_c, U_c, V_c, tx_c, ty_c])
        r_c = self.apply_bcs_2d(
            r_c,
            nxw_c, nyw_c,
            nxw2_c, nyw2_c,
            nxu_c, nyu_c,
            nxv_c, nyv_c,
            nxtx_c, nytx_c,
            nxty_c, nyty_c,
        )
        return r_c
