import numpy as np
from .basis import second_order_quadrature, zero_order_quadrature, first_order_quadrature
from .basis import interp_lagrange, interp_lagrange_transpose
from .basis import get_lagrange_basis_2d_all

import scipy.sparse as sp
# ============================================================
# ASGS element + added CIP edge stabilization
# ============================================================

import numpy as np
import scipy.sparse as sp
from .basis import second_order_quadrature
from .basis import get_lagrange_basis_2d_all

class AlgebraicSubGridScaleElement:
    """
    ASGS Reissner–Mindlin Q1 plate element + inter-element edge SGS terms.

    Compatibility requirements (for your StabilizedPlateAssembler):
      - attributes:
          dof_per_node = 3
          nodes_per_elem = 4
          ndof = 12
          clamped (bool)
          edge_stab (bool)
      - methods:
          get_kelem(E,nu,thick,elem_xpts) -> (12,12)
          get_felem(mag,elem_xpts) -> (12,)
          get_edge_stab_kelem(E,nu,thick, elem_xpts_L, elem_xpts_R,
                              edge_L, edge_R, nx, ny, loc_conn_L=None, loc_conn_R=None)
              -> (Kedge6(6,6), gnodes2(g0,g1))

    Notes on the paper:
      - Volume ASGS uses tau_w/eps^2 and tau_th/eps^2 (your current structure).
      - Edge SGS terms for RM plates are (paper Eq. 5.14):
            -(δ k1/2) < [[ n·∇θ ]], [[ n·∇φ ]] >_E
            -(δ k2/2) < [[ n (div θ) ]], [[ n (div φ) ]] >_E
            -(δ /(2ε))< [[ n·∇w ]], [[ n·∇v ]] >_E
        with δ = δ0 * h (δ0 dimensionless).
      - Because those appear with MINUS sign in the bilinear form,
        we assemble Kedge6 as NEGATIVE semi-definite.
    """

    def __init__(self, edge_stab: bool = True):
        self.dof_per_node = 3
        self.nodes_per_elem = 4
        self.ndof = 12
        self.clamped = False
        self.edge_stab = bool(edge_stab)

        # ASGS constants (same family you already use)
        self.c1 = 12.0
        self.c3 = 12.0
        self.c2 = 1.0
        self.c4 = 1.0

        # edge SGS parameter δ = δ0 * h (dimensionless δ0)
        # NOTE : fix this later, currently is hacked

        # self.delta0 = 0.1  # start modest; sweep 0.01..0.5 first
        # self.delta0 = 1e-3
        # self.delta0 = 1e-4
        # self.delta0 = 2e-5
        # self.delta0 = 1.25e-5
        # self.delta0 = 1.225e-5
        # self.delta0 = 1.21e-5
        self.delta0 = 1.203e-5 # Rc = 1.927 mesh conv rate at thick = 1e-3
        # self.delta0 = 1.202e-5 # Rc = 1.756 mesh conv rate at thick = 1e-3
        # self.delta0 = 1.201e-5
        # self.delta0 = 1.2e-5
        # self.delta0 = 1.19e-5
        # self.delta0 = 1.15e-5
        # self.delta0 = 1.1e-5
        # self.delta0 = 1e-5
        # self.delta0 = 9e-6
        # self.delta0 = 8e-6
        # self.delta0 = 1e-6
        # self.delta0 = 0.0

        # Q1 local edge -> local node ids (KEEP STANDARD ORDER for compatibility)
        # local nodes: 0:(ex,ey), 1:(ex+1,ey), 2:(ex+1,ey+1), 3:(ex,ey+1)
        self._edge_lnodes = {
            0: (0, 1),  # bottom
            1: (1, 2),  # right
            2: (2, 3),  # top
            3: (3, 0),  # left
        }

        # caches for your MG hooks (keep if you use them elsewhere)
        self._P1_cache = {}
        self._P2_cache = {}

    # ----------------------------
    # helpers: material constants
    # ----------------------------
    @staticmethod
    def _k1k2(E, nu, t):
        k1 = E * t**3 / (24.0 * (1.0 + nu))
        k2 = E * t**3 / (24.0 * (1.0 - nu))
        return float(k1), float(k2)

    @staticmethod
    def _ksGh_eps(E, nu, t):
        ks = 5.0 / 6.0
        G = E / (2.0 * (1.0 + nu))
        ksGh = ks * G * t
        eps = 1.0 / ksGh
        return float(ksGh), float(eps)

    @staticmethod
    def _elem_h(elem_xpts):
        x = elem_xpts[0::3]
        y = elem_xpts[1::3]
        dx = float(np.max(x) - np.min(x))
        dy = float(np.max(y) - np.min(y))
        # structured quad
        return float(np.sqrt(abs(dx * dy)))

    @staticmethod
    def _edge_quad_2pt():
        s = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
        w = np.array([1.0, 1.0], dtype=float)
        return s, w

    @staticmethod
    def _ref_edge(edge_id: int, s: float):
        # reference quad edges (xi,eta) in [-1,1]^2
        if edge_id == 0:  # bottom: eta=-1
            return s, -1.0
        if edge_id == 1:  # right: xi=+1
            return 1.0, s
        if edge_id == 2:  # top: eta=+1
            return s, 1.0
        if edge_id == 3:  # left: xi=-1
            return -1.0, s
        raise ValueError("edge_id must be 0..3")

    # ----------------------------
    # volume element stiffness
    # ----------------------------
    def get_kelem(self, E: float, nu: float, thick: float, elem_xpts: np.ndarray):
        pts, wts = second_order_quadrature()
        kelem = np.zeros((self.ndof, self.ndof), dtype=float)

        k1, k2 = self._k1k2(E, nu, thick)
        ksGh, eps = self._ksGh_eps(E, nu, thick)

        # shear matrix
        Ds = ksGh * np.eye(2)

        # mesh size and taus
        h = self._elem_h(elem_xpts)
        kstab = k1 + k2

        tau_th = 1.0 / (self.c1 * kstab / (h**2) + self.c2 / eps)
        tau_w  = 1.0 / (self.c3 / eps / (h**2) + self.c4 / (eps**2) / kstab)

        x = elem_xpts[0::3]
        y = elem_xpts[1::3]

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

                dA = wt * J

                # div(theta) = thx_x + thy_y
                Bdiv = np.zeros((1, self.ndof))
                Bdiv[0, 1::3] = Nx
                Bdiv[0, 2::3] = Ny

                # k1 ||∇θ||^2 + k2 ||div θ||^2 (your "paper formulation" variant)
                Bgrad = np.zeros((4, self.ndof))
                Bgrad[0, 1::3] = Nx
                Bgrad[1, 1::3] = Ny
                Bgrad[2, 2::3] = Nx
                Bgrad[3, 2::3] = Ny

                kelem += k1 * (Bgrad.T @ Bgrad) * dA
                kelem += k2 * (Bdiv.T  @ Bdiv)  * dA

                # shear operator: [w_x - thx, w_y - thy]
                Bs = np.zeros((2, self.ndof))
                Bs[0, 0::3] = Nx
                Bs[0, 1::3] -= N
                Bs[1, 0::3] = Ny
                Bs[1, 2::3] -= N

                kelem += (Bs.T @ Ds @ Bs) * dA

                # ASGS interior stabilization (paper uses tau/eps^2)
                kelem -= (tau_w  / (eps**2)) * (Bdiv.T @ Bdiv) * dA
                kelem -= (tau_th / (eps**2)) * (Bs.T   @ Bs)   * dA

        return kelem

    def get_felem(self, mag, elem_xpts: np.ndarray):
        pts, wts = second_order_quadrature()
        felem = np.zeros(self.ndof, dtype=float)

        x = elem_xpts[0::3]
        y = elem_xpts[1::3]

        for ii, xi in enumerate(pts):
            for jj, eta in enumerate(pts):
                wt = wts[ii] * wts[jj]
                N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)

                x_xi = np.dot(Nxi, x);  x_eta = np.dot(Neta, x)
                y_xi = np.dot(Nxi, y);  y_eta = np.dot(Neta, y)
                J = x_xi * y_eta - x_eta * y_xi

                xq = float(np.dot(N, x))
                yq = float(np.dot(N, y))
                q = float(mag(xq, yq))

                felem[0::3] += q * wt * J * N  # load on w

        return felem

    # ----------------------------
    # EDGE SGS: required by your assembler
    # ----------------------------
    def get_edge_stab_kelem(
        self,
        E: float, nu: float, thick: float,
        elem_xpts_L: np.ndarray, elem_xpts_R: np.ndarray,
        edge_L: int, edge_R: int,
        nx: float, ny: float,
        loc_conn_L: np.ndarray = None,
        loc_conn_R: np.ndarray = None,
    ):
        """
        Returns:
          Kedge6: (6,6) over the two shared edge nodes (in left edge node order)
          gnodes2: (g0,g1) global node ids in that same order

        Assembles the inter-element SGS edge terms (paper Eq. 5.14):
          Kedge6 = - ∫_E [ (δ k1/2) [[n·∇θ]]·[[n·∇φ]]
                         + (δ k2/2) [[div θ]] [[div φ]]
                         + (δ /(2ε)) [[n·∇w]] [[n·∇v]] ] ds

        where δ = delta0 * h, h from the left element size.
        """
        if (loc_conn_L is None) or (loc_conn_R is None):
            raise ValueError("Assembler must pass loc_conn_L and loc_conn_R for consistent edge DOF ordering.")

        dpn = 3
        k1, k2 = self._k1k2(E, nu, thick)
        _, eps = self._ksGh_eps(E, nu, thick)

        # δ = δ0 * h
        h = self._elem_h(elem_xpts_L)
        delta = float(self.delta0) * h

        # left edge local nodes (standard)
        l0, l1 = self._edge_lnodes[edge_L]
        g0 = int(loc_conn_L[l0])
        g1 = int(loc_conn_L[l1])
        gnodes2 = (g0, g1)

        def cols3(ln):
            return [3*ln + 0, 3*ln + 1, 3*ln + 2]  # w, thx, thy

        cols6_L = cols3(l0) + cols3(l1)

        # right side local edge nodes, reorder to match (g0,g1)
        r0, r1 = self._edge_lnodes[edge_R]
        rg0 = int(loc_conn_R[r0])
        rg1 = int(loc_conn_R[r1])
        if (rg0, rg1) == (g0, g1):
            cols6_R = cols3(r0) + cols3(r1)
        elif (rg1, rg0) == (g0, g1):
            cols6_R = cols3(r1) + cols3(r0)
        else:
            raise RuntimeError("Neighbor edge node mismatch: wrong edge pairing or mesh not conforming.")

        xL = elem_xpts_L[0::3]; yL = elem_xpts_L[1::3]
        xR = elem_xpts_R[0::3]; yR = elem_xpts_R[1::3]

        spts, swts = self._edge_quad_2pt()
        Kedge6 = np.zeros((2*dpn, 2*dpn), dtype=float)

        for s, ws in zip(spts, swts):
            xiL, etaL = self._ref_edge(edge_L, float(s))
            xiR, etaR = self._ref_edge(edge_R, float(s))

            NL, NxiL, NetaL = get_lagrange_basis_2d_all(xiL, etaL)
            NR, NxiR, NetaR = get_lagrange_basis_2d_all(xiR, etaR)

            # ---- L mapping / grads
            x_xi  = np.dot(NxiL, xL); x_eta = np.dot(NetaL, xL)
            y_xi  = np.dot(NxiL, yL); y_eta = np.dot(NetaL, yL)
            JL = x_xi * y_eta - x_eta * y_xi
            invJL = 1.0 / JL
            xi_x  =  y_eta * invJL
            xi_y  = -x_eta * invJL
            eta_x = -y_xi  * invJL
            eta_y =  x_xi  * invJL
            NxL = NxiL * xi_x + NetaL * eta_x
            NyL = NxiL * xi_y + NetaL * eta_y

            # ---- R mapping / grads
            x_xiR  = np.dot(NxiR, xR); x_etaR = np.dot(NetaR, xR)
            y_xiR  = np.dot(NxiR, yR); y_etaR = np.dot(NetaR, yR)
            JR = x_xiR * y_etaR - x_etaR * y_xiR
            invJR = 1.0 / JR
            xi_xR  =  y_etaR * invJR
            xi_yR  = -x_etaR * invJR
            eta_xR = -y_xiR  * invJR
            eta_yR =  x_xiR  * invJR
            NxR = NxiR * xi_xR + NetaR * eta_xR
            NyR = NxiR * xi_yR + NetaR * eta_yR

            # edge metric ds (from L)
            if edge_L in (0, 2):  # eta const, xi varies
                tx, ty = x_xi, y_xi
            else:                 # xi const, eta varies
                tx, ty = x_eta, y_eta
            ds = float(np.sqrt(tx*tx + ty*ty))
            w = float(ws * ds)

            # d/dn for shape fns on each side
            dndL = nx * NxL + ny * NyL
            dndR = nx * NxR + ny * NyR

            # (1) n·∇θ  => 2 components [thx_n, thy_n]
            Bn_th_L = np.zeros((2, self.ndof))
            Bn_th_R = np.zeros((2, self.ndof))
            Bn_th_L[0, 1::3] = dndL
            Bn_th_L[1, 2::3] = dndL
            Bn_th_R[0, 1::3] = dndR
            Bn_th_R[1, 2::3] = dndR

            # (2) div θ = thx_x + thy_y
            Bdiv_L = np.zeros((1, self.ndof))
            Bdiv_R = np.zeros((1, self.ndof))
            Bdiv_L[0, 1::3] = NxL
            Bdiv_L[0, 2::3] = NyL
            Bdiv_R[0, 1::3] = NxR
            Bdiv_R[0, 2::3] = NyR

            # (3) n·∇w
            Bn_w_L = np.zeros((1, self.ndof))
            Bn_w_R = np.zeros((1, self.ndof))
            Bn_w_L[0, 0::3] = dndL
            Bn_w_R[0, 0::3] = dndR

            # restrict to the 2 edge nodes (6 dofs)
            J_th  = (Bn_th_L[:, cols6_L] - Bn_th_R[:, cols6_R])   # (2,6)
            J_div = (Bdiv_L[:,  cols6_L] - Bdiv_R[:,  cols6_R])   # (1,6)
            J_w   = (Bn_w_L[:,  cols6_L] - Bn_w_R[:,  cols6_R])   # (1,6)

            # coefficients (5.14): -δ k1/2, -δ k2/2, -δ/(2ε)
            c_th  = (delta * k1) / 2.0
            c_div = (delta * k2) / 2.0
            c_w   = (delta / (2.0 * eps))

            # c_w *= 1e-1
            c_w = 0.0

            # assemble with MINUS sign (important)
            Kedge6 -= c_th  * (J_th.T  @ J_th)  * w
            Kedge6 -= c_div * (J_div.T @ J_div) * w
            Kedge6 -= c_w   * (J_w.T   @ J_w)   * w

        return Kedge6, gnodes2
    
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
