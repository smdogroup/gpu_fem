# New mixed IGA cylinder element (no extra w2, no mixed constraint)
# DOF spaces:
#   w   : (p2,p2)  ->  9
#   thx : (p1,p2)  ->  6
#   thy : (p2,p1)  ->  6
#   u   : (p3,p2)  -> 12
#   v   : (p2,p3)  -> 12
#
# Strains:
#   exx = u_x + w/Rx   (Rx -> ∞ for cylinder axis => term = 0)
#   eyy = v_y + w/Ry   (Ry = r)
#   exy = u_y + v_x    (REDUCED INTEGRATION only)
#
# Bending:
#   kxx = thx_x
#   kyy = thy_y
#   kxy = thy_x + thx_y
#
# Shear:
#   gamx = w_x + thx
#   gamy = w_y + thy
#
# Locking handling:
#   - all locking-sensitive strains (exx, eyy, gamx, gamy) are evaluated consistently with IGA
#   - exy is reduced-integrated (1-point) to avoid membrane shear locking

import numpy as np
from .basis import (
    third_order_quadrature,
    second_order_quadrature,
    zero_order_quadrature,
    get_iga2_basis,
    get_iga3_basis,
    get_lagrange_basis_01,
)

class DeRhamMITC_IGACylinderElement:
    """
    Mixed IGA cylinder element with different-order spaces:

      Ue = [ w(9), u(12), v(12), thx(6), thy(6) ]

    Spaces:
      w   : (p2,p2)
      u   : (p3,p2)
      v   : (p2,p3)
      thx : (p1,p2)
      thy : (p2,p1)
    """

    def __init__(self, r: float, clamped: bool = False, curvature_on: bool = True,
                 reduced_integrate_exy: bool = True, rax:float=None):
        self.r = float(r)
        self.rax = float(rax) if rax is not None else None
        self.clamped = bool(clamped)
        self.curvature_on = bool(curvature_on)
        self.reduced_integrate_exy = bool(reduced_integrate_exy)
        self._P_cache = {}
        self.dof_per_node = 5

    # ---- tensor helpers ------------------------------------------------------
    @staticmethod
    def _tensor_product_basis(bx, by):
        Nx, dNx = bx
        Ny, dNy = by
        N    = np.kron(Ny, Nx)
        Nxi  = np.kron(Ny, dNx)
        Neta = np.kron(dNy, Nx)
        return N, Nxi, Neta

    # ---- 1D bases ------------------------------------------------------------
    @staticmethod
    def _iga3_1d(x, ixe, nxe):
        return get_iga3_basis(x, ixe, nxe)

    @staticmethod
    def _iga2_1d(x, ixe, nxe):
        left = (ixe == 0)
        right = (ixe == nxe - 1)
        return get_iga2_basis(x, left, right)

    @staticmethod
    def _p1_1d(x):
        return get_lagrange_basis_01(x)

    def get_kelem(self, E, nu, thick, dx, dy, ixe, nxe, iye, nye):
        """
        Returns dense block stiffness for local dofs ordered:
          [ w(9), u(12), v(12), thx(6), thy(6) ]

        Output is a tuple of 25 blocks in (w,u,v,thx,thy) x (w,u,v,thx,thy) order:
          Kww, Kwu, Kwv, Kwtx, Kwty,
          Kuw, Kuu, Kuv, Kutx, Kuty,
          Kvw, Kvu, Kvv, Kvtx, Kvty,
          Ktxw, Ktxu, Ktxv, Ktxtx, Ktxty,
          Ktyw, Ktyu, Ktyv, Ktytx, Ktyty
        """
        # full quadrature for most terms
        pts, wts = third_order_quadrature()

        # reduced quadrature (1-point) for exy only
        pts_r, wts_r = zero_order_quadrature()

        r = self.r
        Jdet = dx * dy
        xi_x  = 1.0 / dx
        eta_y = 1.0 / dy

        # sizes
        n_w   = 9   # (p2,p2)
        n_u   = 12  # (p3,p2)
        n_v   = 12  # (p2,p3)
        n_thx = 6   # (p1,p2)
        n_thy = 6   # (p2,p1)

        def Z(a, b): return np.zeros((a, b))

        # blocks: (w, u, v, thx, thy)
        Kww   = Z(n_w,   n_w)
        Kwu   = Z(n_w,   n_u)
        Kwv   = Z(n_w,   n_v)
        Kwtx  = Z(n_w,   n_thx)
        Kwty  = Z(n_w,   n_thy)

        Kuw   = Z(n_u,   n_w)
        Kuu   = Z(n_u,   n_u)
        Kuv   = Z(n_u,   n_v)
        Kutx  = Z(n_u,   n_thx)
        Kuty  = Z(n_u,   n_thy)

        Kvw   = Z(n_v,   n_w)
        Kvu   = Z(n_v,   n_u)
        Kvv   = Z(n_v,   n_v)
        Kvtx  = Z(n_v,   n_thx)
        Kvty  = Z(n_v,   n_thy)

        Ktxw  = Z(n_thx, n_w)
        Ktxu  = Z(n_thx, n_u)
        Ktxv  = Z(n_thx, n_v)
        Ktxtx = Z(n_thx, n_thx)
        Ktxty = Z(n_thx, n_thy)

        Ktyw  = Z(n_thy, n_w)
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
            [0.0, 0.0, (1.0 - nu) / 2.0],  # multiplies engineering shear exy = u_y + v_x
        ])

        ks = 5.0 / 6.0
        G  = E / (2.0 * (1.0 + nu))
        Ds = (ks * G * thick) * np.eye(2)

        curv_on = self.curvature_on
        invRx = (1.0 / self.rax) if (curv_on and self.rax is not None) else 0.0
        invRy = (1.0 / r) if (curv_on and r != 0.0) else 0.0

        # ==========================================================
        # BENDING: [kxx, kyy, kxy] = [thx_x, thy_y, thy_x + thx_y]
        # ==========================================================
        for _eta, w_eta_raw in zip(pts, wts):
            w_eta = 0.5 * w_eta_raw
            eta = 0.5 * (_eta + 1.0)

            # y-bases
            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)  # p2
            Ny3, dNy3 = self._iga3_1d(eta, iye, nye)  # p3
            Ny1, dNy1 = self._p1_1d(eta)              # p1

            for _xi, w_xi_raw in zip(pts, wts):
                w_xi = 0.5 * w_xi_raw
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * Jdet

                # x-bases
                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)  # p2
                Nx3, dNx3 = self._iga3_1d(xi, ixe, nxe)  # p3
                Nx1, dNx1 = self._p1_1d(xi)              # p1

                # thx: (p1,p2)
                Ntx, Ntx_xi, Ntx_eta = self._tensor_product_basis((Nx1, dNx1), (Ny2, dNy2))
                Ntx_x = Ntx_xi * xi_x
                Ntx_y = Ntx_eta * eta_y

                # thy: (p2,p1)
                Nty, Nty_xi, Nty_eta = self._tensor_product_basis((Nx2, dNx2), (Ny1, dNy1))
                Nty_x = Nty_xi * xi_x
                Nty_y = Nty_eta * eta_y

                kxx_thx = Ntx_x
                kyy_thy = Nty_y
                kxy_thy = Nty_x
                kxy_thx = Ntx_y

                D11, D12, D22, D33 = Db[0, 0], Db[0, 1], Db[1, 1], Db[2, 2]

                # kxx-kxx
                Ktxtx += wt * (D11 * np.outer(kxx_thx, kxx_thx))

                # kyy-kyy
                Ktyty += wt * (D22 * np.outer(kyy_thy, kyy_thy))

                # coupling kxx-kyy
                Ktxty += wt * (D12 * np.outer(kxx_thx, kyy_thy))
                Ktytx += wt * (D12 * np.outer(kyy_thy, kxx_thx))

                # kxy = thy_x + thx_y
                Ktyty += wt * (D33 * np.outer(kxy_thy, kxy_thy))
                Ktxtx += wt * (D33 * np.outer(kxy_thx, kxy_thx))
                Ktxty += wt * (D33 * np.outer(kxy_thx, kxy_thy))
                Ktytx += wt * (D33 * np.outer(kxy_thy, kxy_thx))

        # ==========================================================
        # SHEAR: gamx = w_x + thx, gamy = w_y + thy
        # ==========================================================
        for _eta, w_eta_raw in zip(pts, wts):
            w_eta = 0.5 * w_eta_raw
            eta = 0.5 * (_eta + 1.0)

            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)  # p2
            Ny1, dNy1 = self._p1_1d(eta)              # p1

            for _xi, w_xi_raw in zip(pts, wts):
                w_xi = 0.5 * w_xi_raw
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * Jdet

                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)  # p2
                Nx1, dNx1 = self._p1_1d(xi)              # p1

                # w: (p2,p2)
                Nw, Nw_xi, Nw_eta = self._tensor_product_basis((Nx2, dNx2), (Ny2, dNy2))
                Nw_x = Nw_xi * xi_x
                Nw_y = Nw_eta * eta_y

                # thx: (p1,p2)
                Ntx, _, _ = self._tensor_product_basis((Nx1, dNx1), (Ny2, dNy2))

                # thy: (p2,p1)
                Nty, _, _ = self._tensor_product_basis((Nx2, dNx2), (Ny1, dNy1))

                g1_w   = Nw_x
                g1_thx = Ntx
                g2_w   = Nw_y
                g2_thy = Nty

                Ds11 = Ds[0, 0]
                Ds22 = Ds[1, 1]

                # gamx
                Kww   += wt * (Ds11 * np.outer(g1_w,   g1_w))
                Kwtx  += wt * (Ds11 * np.outer(g1_w,   g1_thx))
                Ktxw  += wt * (Ds11 * np.outer(g1_thx, g1_w))
                Ktxtx += wt * (Ds11 * np.outer(g1_thx, g1_thx))

                # gamy
                Kww   += wt * (Ds22 * np.outer(g2_w,   g2_w))
                Kwty  += wt * (Ds22 * np.outer(g2_w,   g2_thy))
                Ktyw  += wt * (Ds22 * np.outer(g2_thy, g2_w))
                Ktyty += wt * (Ds22 * np.outer(g2_thy, g2_thy))

        # ==========================================================
        # MEMBRANE (full for exx/eyy + coupling; reduced for exy)
        #
        # exx = u_x + w/Rx  (Rx term off for cylinder axis => 0)
        # eyy = v_y + w/Ry
        # exy = u_y + v_x   (reduced integration)
        # ==========================================================

        # ---- full integration for exx/eyy and coupling (NO exy here) ----
        for _eta, w_eta_raw in zip(pts, wts):
            w_eta = 0.5 * w_eta_raw
            eta = 0.5 * (_eta + 1.0)

            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)  # p2
            Ny3, dNy3 = self._iga3_1d(eta, iye, nye)  # p3

            for _xi, w_xi_raw in zip(pts, wts):
                w_xi = 0.5 * w_xi_raw
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * Jdet

                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)  # p2
                Nx3, dNx3 = self._iga3_1d(xi, ixe, nxe)  # p3

                # w: (p2,p2)
                Nw, _, _ = self._tensor_product_basis((Nx2, dNx2), (Ny2, dNy2))

                # u: (p3,p2)
                Nu, Nu_xi, Nu_eta = self._tensor_product_basis((Nx3, dNx3), (Ny2, dNy2))
                Nu_x = Nu_xi * xi_x
                # Nu_y = Nu_eta * eta_y  (only needed for exy reduced)

                # v: (p2,p3)
                Nv, Nv_xi, Nv_eta = self._tensor_product_basis((Nx2, dNx2), (Ny3, dNy3))
                # Nv_x = Nv_xi * xi_x    (only needed for exy reduced)
                Nv_y = Nv_eta * eta_y

                exx_u = Nu_x                       # + 0*w
                eyy_v = Nv_y
                exx_w = invRx * Nw
                eyy_w = invRy * Nw                  # curvature term (w/Ry)

                D11, D12, D22 = Dm[0, 0], Dm[0, 1], Dm[1, 1]

                # exx-exx
                Kuu += wt * (D11 * np.outer(exx_u, exx_u))
                Kuw += wt * (D11 * np.outer(exx_u, exx_w))
                Kwu += wt * (D11 * np.outer(exx_w, exx_u))
                Kww += wt * (D11 * np.outer(exx_w, exx_w))

                # eyy-eyy pieces
                Kvv += wt * (D22 * np.outer(eyy_v, eyy_v))
                Kww += wt * (D22 * np.outer(eyy_w, eyy_w))
                Kvw += wt * (D22 * np.outer(eyy_v, eyy_w))
                Kwv += wt * (D22 * np.outer(eyy_w, eyy_v))

                # coupling exx-eyy via nu term
                Kuv += wt * (D12 * np.outer(exx_u, eyy_v))
                Kvu += wt * (D12 * np.outer(eyy_v, exx_u))

                Kwu += wt * (D12 * np.outer(eyy_w, exx_u))
                Kuw += wt * (D12 * np.outer(exx_u, eyy_w))

                Kwv += wt * (D12 * np.outer(exx_w, eyy_v))
                Kvw += wt * (D12 * np.outer(eyy_v, exx_w))

        # ---- reduced integration for exy only ----
        if self.reduced_integrate_exy:
            for _eta, w_eta_raw in zip(pts_r, wts_r):
                w_eta = 0.5 * w_eta_raw
                eta = 0.5 * (_eta + 1.0)

                Ny2, dNy2 = self._iga2_1d(eta, iye, nye)  # p2
                Ny3, dNy3 = self._iga3_1d(eta, iye, nye)  # p3

                for _xi, w_xi_raw in zip(pts_r, wts_r):
                    w_xi = 0.5 * w_xi_raw
                    xi = 0.5 * (_xi + 1.0)

                    wt = (w_xi * w_eta) * Jdet

                    Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)  # p2
                    Nx3, dNx3 = self._iga3_1d(xi, ixe, nxe)  # p3

                    # u: (p3,p2)
                    Nu, Nu_xi, Nu_eta = self._tensor_product_basis((Nx3, dNx3), (Ny2, dNy2))
                    Nu_y = Nu_eta * eta_y

                    # v: (p2,p3)
                    Nv, Nv_xi, Nv_eta = self._tensor_product_basis((Nx2, dNx2), (Ny3, dNy3))
                    Nv_x = Nv_xi * xi_x

                    exy_u = Nu_y
                    exy_v = Nv_x

                    D33 = Dm[2, 2]  # engineering shear modulus part

                    # exy-exy contribution
                    Kuu += wt * (D33 * np.outer(exy_u, exy_u))
                    Kvv += wt * (D33 * np.outer(exy_v, exy_v))
                    Kuv += wt * (D33 * np.outer(exy_u, exy_v))
                    Kvu += wt * (D33 * np.outer(exy_v, exy_u))
        else:
            # if you ever want to full-integrate exy, do it here (same as reduced but with pts,wts)
            pass

        return (
            Kww,  Kwu,  Kwv,  Kwtx,  Kwty,
            Kuw,  Kuu,  Kuv,  Kutx,  Kuty,
            Kvw,  Kvu,  Kvv,  Kvtx,  Kvty,
            Ktxw, Ktxu, Ktxv, Ktxtx, Ktxty,
            Ktyw, Ktyu, Ktyv, Ktytx, Ktyty
        )
    
    def get_felem(
        self,
        load_fcn,     # callable q(x,y) acting on w only
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        ixe: int, nxe: int,
        iye: int, nye: int,
    ):
        """
        Consistent load vector for q(x,y) acting on w only.

        Local unknown ordering (this class):
        [ w(9), u(12), v(12), thx(6), thy(6) ]

        Returns:
        fw(9), fu(12), fv(12), ftx(6), fty(6)
        """
        pts, wts = third_order_quadrature()

        fw  = np.zeros(9)    # w   : (p2,p2)
        fu  = np.zeros(12)   # u   : (p3,p2)  (no load here)
        fv  = np.zeros(12)   # v   : (p2,p3)  (no load here)
        ftx = np.zeros(6)    # thx : (p1,p2)  (no load here)
        fty = np.zeros(6)    # thy : (p2,p1)  (no load here)

        Jdet = dx * dy

        for _eta, w_eta_raw in zip(pts, wts):
            w_eta = 0.5 * w_eta_raw
            eta = 0.5 * (_eta + 1.0)

            # y-basis for w is p2
            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)

            for _xi, w_xi_raw in zip(pts, wts):
                w_xi = 0.5 * w_xi_raw
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * Jdet

                # x-basis for w is p2
                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)

                # w basis: (p2,p2) -> 9
                Nw, _, _ = self._tensor_product_basis((Nx2, dNx2), (Ny2, dNy2))

                xq = x0 + xi * dx
                yq = y0 + eta * dy
                q = float(load_fcn(xq, yq))

                fw += (q * Nw) * wt

        return fw, fu, fv, ftx, fty

    # -------------------------------------------------------------------------
    # BCs for global vector layout: [w, u, v, thx, thy]
    # -------------------------------------------------------------------------
    def apply_bcs_2d(self, u: np.ndarray,
                     nxw: int, nyw: int,
                     nxu: int, nyu: int,
                     nxv: int, nyv: int,
                     nxtx: int, nytx: int,
                     nxty: int, nyty: int):
        u = np.asarray(u)

        nw   = nxw  * nyw
        nu   = nxu  * nyu
        nv   = nxv  * nyv
        ntx  = nxtx * nytx
        nty  = nxty * nyty

        assert u.size == (nw + nu + nv + ntx + nty)

        off = 0
        w  = u[off:off+nw];   off += nw
        U  = u[off:off+nu];   off += nu
        V  = u[off:off+nv];   off += nv
        tx = u[off:off+ntx];  off += ntx
        ty = u[off:off+nty];  off += nty

        def on_bndry(i, j, nx, ny):
            return (i == 0) or (i == nx-1) or (j == 0) or (j == ny-1)

        # w boundary always
        for j in range(nyw):
            for i in range(nxw):
                if on_bndry(i, j, nxw, nyw):
                    w[i + nxw*j] = 0.0

        if self.clamped:
            for j in range(nyu):
                for i in range(nxu):
                    if on_bndry(i, j, nxu, nyu):
                        U[i + nxu*j] = 0.0

            for j in range(nyv):
                for i in range(nxv):
                    if on_bndry(i, j, nxv, nyv):
                        V[i + nxv*j] = 0.0

            for j in range(nytx):
                for i in range(nxtx):
                    if on_bndry(i, j, nxtx, nytx):
                        tx[i + nxtx*j] = 0.0

            for j in range(nyty):
                for i in range(nxty):
                    if on_bndry(i, j, nxty, nyty):
                        ty[i + nxty*j] = 0.0
        else:
            # SS-ish: u=0 on x=0 edge (on u grid), v=0 on min-theta edge (y=0) (on v grid)
            for j in range(nyu):
                for i in range(nxu):
                    if i == 0:
                        U[i + nxu*j] = 0.0
            for j in range(nyv):
                for i in range(nxv):
                    if j == 0:
                        V[i + nxv*j] = 0.0

        return np.concatenate([w, U, V, tx, ty])

    # -------------------------------------------------------------------------
    # Multigrid transfers (dyadic refinement) for [w, u, v, thx, thy]
    # -------------------------------------------------------------------------
    @staticmethod
    def _kron2(Ry: np.ndarray, Rx: np.ndarray) -> np.ndarray:
        return np.kron(Ry, Rx)

    def _build_R_p3(self, nxe_c: int) -> np.ndarray:
        # ---- paste your existing p3 restriction here (unchanged) ----
        n_w_c = nxe_c + 3
        nxe_f = 2 * nxe_c
        n_w_f = nxe_f + 3
        R = np.zeros((n_w_c, n_w_f))
        for ielem in range(nxe_c):
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
            elif ielem == nxe_c - 2:
                R_loc = np.array([
                    [0.5   , 0.125 , 0.0   , 0.0   , 0.0],
                    [0.5   , 0.75  , 0.5   , 0.125 , 0.0],
                    [0.0   , 0.125 , 0.5   , 0.6875, 0.25],
                    [0.0   , 0.0   , 0.0   , 0.1875, 0.75]
                ])
            elif ielem == nxe_c - 1:
                R_loc = np.array([
                    [0.5   , 0.125 , 0.0   , 0.0   , 0.0],
                    [0.5   , 0.6875, 0.25  , 0.0   , 0.0],
                    [0.0   , 0.1875, 0.75  , 0.5   , 0.0],
                    [0.0   , 0.0   , 0.0   , 0.5   , 1.0]
                ])
            else:
                R_loc = np.array([
                    [0.5   , 0.125 , 0.0   , 0.0   , 0.0],
                    [0.5   , 0.75  , 0.5   , 0.125 , 0.0],
                    [0.0   , 0.125 , 0.5   , 0.75  , 0.5],
                    [0.0   , 0.0   , 0.0   , 0.125 , 0.5]
                ])
            R[ielem:(ielem+4), 2*ielem:(2*ielem+5)] += R_loc
        R /= np.sum(R, axis=0)
        return R

    def _build_R_p2(self, nxe_c: int) -> np.ndarray:
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
        n_c = nxe_c + 1
        n_f = 2 * nxe_c + 1
        P = np.zeros((n_f, n_c))
        for i in range(nxe_c):
            P[2*i,   i]   = 1.0
            P[2*i+1, i]   = 0.5
            P[2*i+1, i+1] = 0.5
        P[2*nxe_c, nxe_c] = 1.0
        return P

    def _build_R_p1(self, nxe_c: int) -> np.ndarray:
        n_c = nxe_c + 1
        n_f = 2 * nxe_c + 1
        R = np.zeros((n_c, n_f))
        R[0, 0] = 1.0
        R[-1, -1] = 1.0
        for i in range(1, n_c - 1):
            R[i, 2*i - 1] = 0.25
            R[i, 2*i]     = 0.50
            R[i, 2*i + 1] = 0.25
        return R

    def prolongate(self, u_c: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        Dyadic prolongation for layout [w, u, v, thx, thy].

        Coarse grids:
          w   : (nxe+2) x (nye+2)   (p2,p2)
          u   : (nxe+3) x (nye+2)   (p3,p2)
          v   : (nxe+2) x (nye+3)   (p2,p3)
          thx : (nxe+1) x (nye+2)   (p1,p2)
          thy : (nxe+2) x (nye+1)   (p2,p1)
        """
        u_c = np.asarray(u_c)

        # coarse sizes
        nxw_c,  nyw_c  = nxe_c + 2, nye_c + 2
        nxu_c,  nyu_c  = nxe_c + 3, nye_c + 2
        nxv_c,  nyv_c  = nxe_c + 2, nye_c + 3
        nxtx_c, nytx_c = nxe_c + 1, nye_c + 2
        nxty_c, nyty_c = nxe_c + 2, nye_c + 1

        nw_c  = nxw_c  * nyw_c
        nu_c  = nxu_c  * nyu_c
        nv_c  = nxv_c  * nyv_c
        ntx_c = nxtx_c * nytx_c
        nty_c = nxty_c * nyty_c

        assert u_c.size == (nw_c + nu_c + nv_c + ntx_c + nty_c)

        off = 0
        w_c  = u_c[off:off+nw_c];   off += nw_c
        U_c  = u_c[off:off+nu_c];   off += nu_c
        V_c  = u_c[off:off+nv_c];   off += nv_c
        tx_c = u_c[off:off+ntx_c];  off += ntx_c
        ty_c = u_c[off:off+nty_c];  off += nty_c

        # 1D prolongations
        Rx3 = self._build_R_p3(nxe_c); Px3 = Rx3.T
        Ry3 = self._build_R_p3(nye_c); Py3 = Ry3.T

        Rx2 = self._build_R_p2(nxe_c); Px2 = Rx2.T
        Ry2 = self._build_R_p2(nye_c); Py2 = Ry2.T

        Px1 = self._build_P_p1(nxe_c)
        Py1 = self._build_P_p1(nye_c)

        # 2D prolongations (kron)
        Pw   = self._kron2(Py2, Px2)   # w  : (p2,p2)
        Pu   = self._kron2(Py2, Px3)   # u  : (p3,p2)
        Pv   = self._kron2(Py3, Px2)   # v  : (p2,p3)
        Ptx  = self._kron2(Py2, Px1)   # thx: (p1,p2)
        Pty  = self._kron2(Py1, Px2)   # thy: (p2,p1)

        w_f  = Pw  @ w_c
        U_f  = Pu  @ U_c
        V_f  = Pv  @ V_c
        tx_f = Ptx @ tx_c
        ty_f = Pty @ ty_c

        # fine sizes
        nxe_f, nye_f = 2*nxe_c, 2*nye_c
        nxw_f,  nyw_f  = nxe_f + 2, nye_f + 2
        nxu_f,  nyu_f  = nxe_f + 3, nye_f + 2
        nxv_f,  nyv_f  = nxe_f + 2, nye_f + 3
        nxtx_f, nytx_f = nxe_f + 1, nye_f + 2
        nxty_f, nyty_f = nxe_f + 2, nye_f + 1

        u_f = np.concatenate([w_f, U_f, V_f, tx_f, ty_f])
        u_f = self.apply_bcs_2d(
            u_f,
            nxw_f, nyw_f,
            nxu_f, nyu_f,
            nxv_f, nyv_f,
            nxtx_f, nytx_f,
            nxty_f, nyty_f,
        )
        return u_f

    def restrict_defect(self, r_f: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        Restrict fine defect -> coarse defect for layout [w, u, v, thx, thy].
        """
        r_f = np.asarray(r_f)
        nxe_f, nye_f = 2*nxe_c, 2*nye_c

        # fine sizes
        nxw_f,  nyw_f  = nxe_f + 2, nye_f + 2
        nxu_f,  nyu_f  = nxe_f + 3, nye_f + 2
        nxv_f,  nyv_f  = nxe_f + 2, nye_f + 3
        nxtx_f, nytx_f = nxe_f + 1, nye_f + 2
        nxty_f, nyty_f = nxe_f + 2, nye_f + 1

        nw_f  = nxw_f  * nyw_f
        nu_f  = nxu_f  * nyu_f
        nv_f  = nxv_f  * nyv_f
        ntx_f = nxtx_f * nytx_f
        nty_f = nxty_f * nyty_f

        assert r_f.size == (nw_f + nu_f + nv_f + ntx_f + nty_f)

        off = 0
        w_f  = r_f[off:off+nw_f];   off += nw_f
        U_f  = r_f[off:off+nu_f];   off += nu_f
        V_f  = r_f[off:off+nv_f];   off += nv_f
        tx_f = r_f[off:off+ntx_f];  off += ntx_f
        ty_f = r_f[off:off+nty_f];  off += nty_f

        # 1D restrictions
        Rx3 = self._build_R_p3(nxe_c)
        Ry3 = self._build_R_p3(nye_c)
        Rx2 = self._build_R_p2(nxe_c)
        Ry2 = self._build_R_p2(nye_c)
        Rx1 = self._build_R_p1(nxe_c)
        Ry1 = self._build_R_p1(nye_c)

        # 2D restrictions
        Rw   = self._kron2(Ry2, Rx2)   # w
        Ru   = self._kron2(Ry2, Rx3)   # u
        Rv   = self._kron2(Ry3, Rx2)   # v
        Rtx  = self._kron2(Ry2, Rx1)   # thx
        Rty  = self._kron2(Ry1, Rx2)   # thy

        w_c  = Rw  @ w_f
        U_c  = Ru  @ U_f
        V_c  = Rv  @ V_f
        tx_c = Rtx @ tx_f
        ty_c = Rty @ ty_f

        # coarse sizes
        nxw_c,  nyw_c  = nxe_c + 2, nye_c + 2
        nxu_c,  nyu_c  = nxe_c + 3, nye_c + 2
        nxv_c,  nyv_c  = nxe_c + 2, nye_c + 3
        nxtx_c, nytx_c = nxe_c + 1, nye_c + 2
        nxty_c, nyty_c = nxe_c + 2, nye_c + 1

        r_c = np.concatenate([w_c, U_c, V_c, tx_c, ty_c])
        r_c = self.apply_bcs_2d(
            r_c,
            nxw_c, nyw_c,
            nxu_c, nyu_c,
            nxv_c, nyv_c,
            nxtx_c, nytx_c,
            nxty_c, nyty_c,
        )
        return r_c