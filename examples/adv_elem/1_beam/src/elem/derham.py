import numpy as np
from .basis import second_order_quadrature
from .basis import get_iga2_basis, get_lagrange_basis_01

class DeRhamIsogeometricElement:
    # NOTE : has its own custom assembler since [w, th] stored separately in block-CSR form
    # not to be confused with BSR form as w and th have different num nodes
    # w is num vertices and th is num edges (leads to compatible grad w and th so that locking-free with IGA)

    # uses 0-form and 1-forms in IGA for Timoshenko beam for DeRham diagram
    # like Nedelec elements with vertices for w and edges for theta
    # so fully compatible

    # see refs: # De Rham (cohomology) IGA plate element
    # based on paper: https://www.sciencedirect.com/science/article/abs/pii/S0045782511003215
    # "An isogeometric method for the Reissner-Mindlin plate bending problem" by Veiga

    # see also the thickness-independent Schwarz multigrid smoothers for it in Benzaken et al.,
    # https://grandmaster.colorado.edu/copper/2016/StudentCompetition/Benzaken_Isogeometric_Multigrid.pdf
    # "Multigrid Methods for Isogeometric Thin Plate Discretizations" 

    def __init__(self):
        self.dof_per_node = 1 # block-CSR style (not BSR style)
        self.clamped = False

    def get_kelem(self, E, nu, thick, elem_length, left_bndry: bool, right_bndry: bool):
        pts, weights = second_order_quadrature()

        # Return dense blocks for speed; assemble into CSR blocks.
        Kww   = np.zeros((3, 3))
        Kwth  = np.zeros((3, 2))
        Kthw  = np.zeros((2, 3))
        Kthth = np.zeros((2, 2))

        EI = (thick**3) * E / 12.0
        ks = 5.0 / 6.0
        G  = E / (2.0 * (1.0 + nu))
        ksGA = ks * G * thick
        J = elem_length

        for _xi, _wt in zip(pts, weights):
            xi = 0.5 * (_xi + 1.0)
            wt = 0.5 * _wt

            Nv, dNv = get_iga2_basis(xi, left_bndry, right_bndry)  # (3,)
            # Ne, dNe = get_lagrange_basis(xi)                       # (2,)
            Ne, dNe = get_lagrange_basis_01(xi)

            dNv = dNv / J
            dNe = dNe / J

            # shear: ksGA * (w_x - th)^2
            c_shear = ksGA * wt * J
            Kww  += c_shear * np.outer(dNv, dNv)
            Kwth -= c_shear * np.outer(dNv, Ne)
            Kthw -= c_shear * np.outer(Ne, dNv)
            Kthth += c_shear * np.outer(Ne, Ne)

            # bending: EI * (th_x)^2
            c_bend = EI * wt * J
            Kthth += c_bend * np.outer(dNe, dNe)

        # Pack as 2x2 block for your calling code, but dense is easiest here
        return Kww, Kwth, Kthw, Kthth

    def get_felem(self, load_fcn, elem_length: float, left_bndry: bool, right_bndry: bool, x0: float):
        """
        Simple consistent load vector for w only:
          ∫ Nw(x) * q(x) dx
        where q(x) = load_fcn(x)
        """
        pts, weights = second_order_quadrature()
        fw = np.zeros(3)
        fth = np.zeros(2)

        J = elem_length
        for _xi, _wt in zip(pts, weights):
            xi = 0.5 * (_xi + 1.0)
            wt = 0.5 * _wt

            Nv, _dNv = get_iga2_basis(xi, left_bndry, right_bndry)

            # map xi in [0,1] to physical x in [x0, x0+L_e]
            x = x0 + xi * elem_length
            q = float(load_fcn(x))

            fw += (q * Nv) * (wt * J)

        return fw, fth
    
    # -------------------------------------------------------------------------
    # w-transfer (quadratic IGA / your existing operator, just wrapped & cleaned)
    # -------------------------------------------------------------------------
    def _build_Rw(self, nxe_c: int) -> np.ndarray:
        """
        Restriction for w only: (n_w_c x n_w_f) mapping w_f -> w_c
        where n_w = nxe + 2.

        This is your existing operator (normalized by counts). Keep it if it works
        with your custom quadratic basis.
        """
        n_w_c = nxe_c + 2
        nxe_f = 2 * nxe_c
        n_w_f = nxe_f + 2

        R = np.zeros((n_w_c, n_w_f))
        counts = 1e-14 * np.ones((n_w_c, n_w_f))

        for ielem_c in range(nxe_c):
            # left half element on fine grid
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

            # right half element on fine grid
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

    def _Pw_from_Rw(self, Rw: np.ndarray) -> np.ndarray:
        """
        Prolongation for w only.

        For a Galerkin hierarchy on SPD problems, restriction is often chosen
        as (scaled) P^T. Since your Rw is already normalized, use Pw = Rw.T.

        If you ever need strict partition-of-unity on Pw rows, you can row-normalize.
        """
        Pw = Rw.T.copy()
        # Optional row normalization (usually not needed if your R is already consistent):
        # rs = Pw.sum(axis=1)
        # rs[rs == 0.0] = 1.0
        # Pw = (Pw.T / rs).T
        return Pw

    # -------------------------------------------------------------------------
    # theta-transfer (linear P1 on "edges": n_th = nxe + 1)
    # -------------------------------------------------------------------------
    def _prolong_th(self, th_c: np.ndarray) -> np.ndarray:
        """
        Dyadic prolongation for linear nodal unknowns (P1):
          th_f[2i]   = th_c[i]
          th_f[2i+1] = 0.5*(th_c[i] + th_c[i+1])
          th_f[end]  = th_c[end]
        """
        nth_c = th_c.shape[0]
        nxe_c = nth_c - 1
        nth_f = 2 * nxe_c + 1

        th_f = np.zeros(nth_f)
        for i in range(nxe_c):
            th_f[2*i]   = th_c[i]
            th_f[2*i+1] = 0.5 * (th_c[i] + th_c[i+1])
        th_f[2*nxe_c] = th_c[nxe_c]
        return th_f

    def _restrict_th(self, th_f: np.ndarray) -> np.ndarray:
        """
        Full-weighting restriction for linear nodal unknowns on a dyadic grid:
          interior: th_c[i] = 0.25*th_f[2i-1] + 0.5*th_f[2i] + 0.25*th_f[2i+1]
          boundaries: injection (or half-weighting) - injection is fine for 1D.
        """
        nth_f = th_f.shape[0]
        nxe_f = nth_f - 1
        assert nxe_f % 2 == 0, "Fine grid must be dyadic refinement of coarse grid."
        nxe_c = nxe_f // 2
        nth_c = nxe_c + 1

        th_c = np.zeros(nth_c)
        th_c[0] = th_f[0]
        for i in range(1, nxe_c):
            th_c[i] = 0.25*th_f[2*i - 1] + 0.5*th_f[2*i] + 0.25*th_f[2*i + 1]
        th_c[nxe_c] = th_f[2*nxe_c]
        return th_c

    # -------------------------------------------------------------------------
    # Public API: prolong/restrict full mixed vector u = [w, th]
    # -------------------------------------------------------------------------

    def prolongate(self, u_c: np.ndarray, nxe_c: int) -> np.ndarray:
        """
        Prolongate with explicit coarse element count.
        """
        u_c = np.asarray(u_c)
        nw_c  = nxe_c + 2
        nth_c = nxe_c + 1
        assert u_c.shape[0] == nw_c + nth_c

        w_c  = u_c[:nw_c]
        th_c = u_c[nw_c:]

        # w prolongation
        Rw = self._build_Rw(nxe_c)
        Pw = self._Pw_from_Rw(Rw)
        w_f = Pw @ w_c

        # th prolongation
        th_f = self._prolong_th(th_c)

        u_f = np.concatenate([w_f, th_f])

        # Re-apply essential BCs
        u_f[0] = 0.0
        u_f[nw_c*0 + (w_f.shape[0]-1)] = 0.0  # last w dof
        if self.clamped:
            u_f[w_f.shape[0] + 0] = 0.0
            u_f[w_f.shape[0] + (th_f.shape[0]-1)] = 0.0

        return u_f

    def restrict_defect(self, r_f: np.ndarray, nxe_c: int) -> np.ndarray:
        """
        Restrict fine defect r_f -> coarse defect r_c with explicit coarse element count.
        Fine is assumed dyadic: nxe_f = 2*nxe_c.
        """
        r_f = np.asarray(r_f)
        nxe_f = 2 * nxe_c
        nw_f  = nxe_f + 2
        nth_f = nxe_f + 1
        assert r_f.shape[0] == nw_f + nth_f

        w_f  = r_f[:nw_f]
        th_f = r_f[nw_f:]

        # w restriction
        Rw = self._build_Rw(nxe_c)
        w_c = Rw @ w_f

        # th restriction
        th_c = self._restrict_th(th_f)

        r_c = np.concatenate([w_c, th_c])

        # Re-apply essential BCs on restricted defect (usually set to 0)
        nw_c = nxe_c + 2
        r_c[0] = 0.0
        r_c[nw_c - 1] = 0.0
        if self.clamped:
            r_c[nw_c + 0] = 0.0
            r_c[nw_c + (nxe_c + 1) - 1] = 0.0

        return r_c