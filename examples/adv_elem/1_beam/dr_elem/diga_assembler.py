import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt


def second_order_quadrature():
    rt35 = np.sqrt(3.0 / 5.0)
    return [-rt35, 0.0, rt35], [5.0/9.0, 8.0/9.0, 5.0/9.0]


def quad_bernstein(xi):
    N = np.array([(1-xi)**2, 2*xi*(1-xi), xi**2])
    dN = np.array([-2*(1-xi), 2*(1-2*xi), 2*xi])
    return N, dN


def get_iga2_basis(xi, left_bndry, right_bndry):
    B, dB = quad_bernstein(xi)

    # bndry adjustment and regular basis
    # on GPU can code it up with ? ternary operators probably
    N = 0.5 * np.array([B[0], np.sum(B) + B[1], B[2]])
    N += 0.5 * left_bndry * np.array([B[0], -B[0], 0.0])
    N += 0.5 * right_bndry * np.array([0.0, -B[2], B[2]])

    # bndry adjustment and regular derivs
    dN = 0.5 * np.array([dB[0], np.sum(dB) + dB[1], dB[2]])
    dN += 0.5 * left_bndry * np.array([dB[0], -dB[0], 0.0])
    dN += 0.5 * right_bndry * np.array([0.0, -dB[2], dB[2]])
    return N, dN

def get_lagrange_basis(xi):
    N = np.array([0.5*(1-xi), 0.5*(1+xi)])
    dN = np.array([-0.5, 0.5])
    return N, dN

def get_lagrange_basis_01(xi):
    N = np.array([1.0 - xi, xi])
    dN = np.array([-1.0, 1.0])
    return N, dN


class DeRhamIGABeamAssembler_V0:
    # uses 0-form and 1-forms in IGA for Timoshenko beam for DeRham diagram
    # like Nedelec elements with vertices for w and edges for theta
    # so fully compatible

    # see refs: # De Rham (cohomology) IGA plate element
    # based on paper: https://www.sciencedirect.com/science/article/abs/pii/S0045782511003215
    # "An isogeometric method for the Reissner-Mindlin plate bending problem" by Veiga

    # see also the thickness-independent Schwarz multigrid smoothers for it in Benzaken et al.,
    # https://grandmaster.colorado.edu/copper/2016/StudentCompetition/Benzaken_Isogeometric_Multigrid.pdf
    # "Multigrid Methods for Isogeometric Thin Plate Discretizations" 

    def __init__(self, ELEMENT, nxe: int, E=70e9, nu=0.3, thick=1e-2, L=1.0,
                 load_fcn=lambda x: 1.0, clamped: bool = False, split_disp_bc:bool=None):

        # split_disp_bc and ELEMENT parameters currently ignored..
        self.element = ELEMENT
        self.nxe = int(nxe)
        self.E = float(E)
        self.nu = float(nu)
        self.thick = float(thick)
        self.L = float(L)
        self.load_fcn = load_fcn

        self.kmat = None
        self.force = None
        self.u = None

        self.nw  = self.nxe + 2   # quadratic IGA field for w (3 per elem, overlapping)
        self.nth = self.nxe + 1   # linear field for theta (2 per elem, overlapping)
        self.N   = self.nw + self.nth

        self.dx_ctrl = self.L / self.nxe

        # Dirichlet BC dofs in the GLOBAL ordering [w(0..nw-1), th(0..nth-1)]
        if clamped:
            self.bcs = [0, self.nw - 1, self.nw, self.nw + self.nth - 1]
        else:
            self.bcs = [0, self.nw - 1]

        # element connectivity:
        # w nodes: [e, e+1, e+2]
        # th edges: [e, e+1]
        self.conn_w  = [[e, e+1, e+2] for e in range(self.nxe)]
        self.conn_th = [[e, e+1]      for e in range(self.nxe)]

        # ---------------------------
        # Build sparsity for 4 blocks
        # ---------------------------
        self.ww_rowp = [0]; self.ww_cols = []; self.ww_nnz = 0
        for i in range(self.nw):
            # 5-band is safe for p=2 across overlapping elements
            cols = [i-2, i-1, i, i+1, i+2]
            cols = [c for c in cols if 0 <= c < self.nw]
            self.ww_cols += cols
            self.ww_nnz  += len(cols)
            self.ww_rowp += [self.ww_nnz]

        self.wth_rowp = [0]; self.wth_cols = []; self.wth_nnz = 0
        for i in range(self.nw):
            # w row couples to nearby theta edges
            cols = [i-2, i-1, i, i+1] # maybe also ..., i+2]
            cols = [c for c in cols if 0 <= c < self.nth]
            self.wth_cols += cols
            self.wth_nnz  += len(cols)
            self.wth_rowp += [self.wth_nnz]

        self.thw_rowp = [0]; self.thw_cols = []; self.thw_nnz = 0
        for i in range(self.nth):
            cols = [i-1, i, i+1, i+2] # maybe also [i-2, ...
            cols = [c for c in cols if 0 <= c < self.nw]
            self.thw_cols += cols
            self.thw_nnz  += len(cols)
            self.thw_rowp += [self.thw_nnz]

        self.thth_rowp = [0]; self.thth_cols = []; self.thth_nnz = 0
        for i in range(self.nth):
            cols = [i-1, i, i+1]
            cols = [c for c in cols if 0 <= c < self.nth]
            self.thth_cols += cols
            self.thth_nnz  += len(cols)
            self.thth_rowp += [self.thth_nnz]

        # Allocate CSR blocks
        self.ww_data   = np.zeros(self.ww_nnz)
        self.wth_data  = np.zeros(self.wth_nnz)
        self.thw_data  = np.zeros(self.thw_nnz)
        self.thth_data = np.zeros(self.thth_nnz)

        self.kmat_ww   = sp.csr_matrix((self.ww_data,   self.ww_cols,   self.ww_rowp),   shape=(self.nw,  self.nw))
        self.kmat_wth  = sp.csr_matrix((self.wth_data,  self.wth_cols,  self.wth_rowp),  shape=(self.nw,  self.nth))
        self.kmat_thw  = sp.csr_matrix((self.thw_data,  self.thw_cols,  self.thw_rowp),  shape=(self.nth, self.nw))
        self.kmat_thth = sp.csr_matrix((self.thth_data, self.thth_cols, self.thth_rowp), shape=(self.nth, self.nth))

        self.kmat = sp.bmat([[self.kmat_ww, self.kmat_wth],
                             [self.kmat_thw, self.kmat_thth]], format="csr")

        # RHS blocks + global RHS
        self.force_w  = np.zeros(self.nw)
        self.force_th = np.zeros(self.nth)
        self.force    = np.zeros(self.N)

    # -----------------------------
    # Helpers for CSR scatter-add
    # -----------------------------
    @staticmethod
    def _row_find_col(col_ind: np.ndarray, start: int, end: int, target_col: int) -> int:
        """Return position p in col_ind[start:end] where col_ind[p]==target_col, else -1."""
        # For tiny bands (<=5), linear search is fastest and simplest.
        for p in range(start, end):
            if col_ind[p] == target_col:
                return p
        return -1

    def _add_to_block(self, A_csr: sp.csr_matrix, rowp, cols, i: int, j: int, val: float):
        """Add 'val' to CSR A(i,j) (assumes (i,j) exists in the pattern)."""
        start, end = rowp[i], rowp[i+1]
        p = self._row_find_col(cols, start, end, j)
        if p < 0:
            # pattern mismatch means your stencil was too small
            raise RuntimeError(f"CSR pattern missing entry ({i},{j}). Increase stencil.")
        A_csr.data[p] += val

    # -----------------------------
    # Element routine (yours)
    # -----------------------------
    def _get_kelem(self, E, nu, thick, elem_length, left_bndry: bool, right_bndry: bool):
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

    def _get_felem(self, elem_length: float, left_bndry: bool, right_bndry: bool, x0: float):
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
            q = float(self.load_fcn(x))

            fw += (q * Nv) * (wt * J)

        return fw, fth

    # -----------------------------
    # Full assembly
    # -----------------------------
    def _assemble_system(self):
        # zero blocks
        self.kmat_ww.data[:]   = 0.0
        self.kmat_wth.data[:]  = 0.0
        self.kmat_thw.data[:]  = 0.0
        self.kmat_thth.data[:] = 0.0

        self.force_w[:]  = 0.0
        self.force_th[:] = 0.0

        for e in range(self.nxe):
            w_dofs  = self.conn_w[e]      # 3
            th_dofs = self.conn_th[e]     # 2

            left_b  = (e == 0)
            right_b = (e == self.nxe - 1)

            Kww, Kwth, Kthw, Kthth = self._get_kelem(
                E=self.E, nu=self.nu, thick=self.thick,
                elem_length=self.dx_ctrl,
                left_bndry=left_b, right_bndry=right_b
            )

            x0 = e * self.dx_ctrl
            fw, fth = self._get_felem(self.dx_ctrl, left_b, right_b, x0)

            # ----- scatter-add stiffness -----
            # ww
            for a, I in enumerate(w_dofs):
                for b, J in enumerate(w_dofs):
                    self._add_to_block(self.kmat_ww, self.ww_rowp, self.kmat_ww.indices, I, J, Kww[a, b])

            # wth
            for a, I in enumerate(w_dofs):
                for b, J in enumerate(th_dofs):
                    self._add_to_block(self.kmat_wth, self.wth_rowp, self.kmat_wth.indices, I, J, Kwth[a, b])

            # thw
            for a, I in enumerate(th_dofs):
                for b, J in enumerate(w_dofs):
                    self._add_to_block(self.kmat_thw, self.thw_rowp, self.kmat_thw.indices, I, J, Kthw[a, b])

            # thth
            for a, I in enumerate(th_dofs):
                for b, J in enumerate(th_dofs):
                    self._add_to_block(self.kmat_thth, self.thth_rowp, self.kmat_thth.indices, I, J, Kthth[a, b])

            # ----- scatter-add load -----
            for a, I in enumerate(w_dofs):
                self.force_w[I] += fw[a]
            # (theta loads typically zero unless you add moments)
            for a, I in enumerate(th_dofs):
                self.force_th[I] += fth[a]

        # form global K and f (global view)
        self.kmat = sp.bmat([[self.kmat_ww, self.kmat_wth],
                             [self.kmat_thw, self.kmat_thth]], format="csr")
        self.force[:self.nw] = self.force_w
        self.force[self.nw:] = self.force_th
        # NOTE : could also use self.force = np.bmat([self.force_w, self.force_th]) if you want
        # and same for bcs..

        # Apply Dirichlet BCs strongly by row/col modification
        self._apply_bcs()

    def _apply_bcs(self):
        """
        Strong Dirichlet BC enforcement on the assembled global matrix.
        For your current bc list, we enforce u[bc]=0.
        """
        if self.kmat is None:
            raise RuntimeError("Assemble first.")

        K = self.kmat.tolil()  # easiest for row/col ops
        f = self.force.copy()

        for dof in self.bcs:
            # zero row, set diagonal
            K.rows[dof] = [dof]
            K.data[dof] = [1.0]
            f[dof] = 0.0

        # also zero the column entries for symmetry if you want:
        # (LIL makes col-zeroing awkward; do it via CSR after.)
        K = K.tocsr()
        for dof in self.bcs:
            # zero column dof, keep diagonal 1
            col = K[:, dof].tocoo()
            if col.nnz:
                K.data[col.row * 0 + 0]  # no-op; just a reminder: col ops on CSR are not direct
        # Simpler symmetric enforcement:
        #   K = K.tolil()
        #   for dof: K[:,dof]=0; K[dof,dof]=1
        # If you need symmetry, do:
        K = K.tolil()
        for dof in self.bcs:
            K[:, dof] = 0.0
            K[dof, dof] = 1.0
        self.kmat = K.tocsr()
        self.force = f

    def direct_solve(self):
        self._assemble_system()
        self.u = sp.linalg.spsolve(self.kmat, self.force)
        # print(f"{self.u=}")
        return self.u

    # def plot_disp(self, idof:int=0):
    #     xvec = self.get_node_pts() # not same as control points
    #     # xvec = self.control_pts
    #     # print(f"{self.u=}")
    #     dpn = self.dof_per_node
    #     w = self.u[idof::dpn]
    #     if self.split_disp_bc:
    #         if dpn == 3: # hhd hermite hierarchic disp
    #             w = self.u[0::3] + self.u[2::3] # wb + ws
    #         elif dpn == 2: # higd iga hierarchic disp
    #             w = self.u[0::2] + self.u[1::2] # wb + ws
    #     plt.figure()
    #     plt.plot(xvec, w)
    #     plt.plot(xvec, np.zeros((self.nnodes,)), "k--")
    #     plt.xlabel("x")
    #     plt.ylabel("w(x)" if idof == 0 else "th(x)")
    #     plt.show()     

    # def prolongate(self, coarse_soln):
    #     return self.element.prolongate(coarse_soln, self.L)
    
    # def restrict_defect(self, fine_defect):
    #     return self.element.restrict_defect(fine_defect, self.L)