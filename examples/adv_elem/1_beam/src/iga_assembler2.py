import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt

# assumes get_iga2_basis is in scope (as in your file)
# from .basis import get_iga2_basis


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


class IGABeamAssemblerV2:
    # for IGA bases or elements

    def __init__(
        self,
        ELEMENT,
        nxe: int,
        E: float = 70e9,
        nu: float = 0.3,
        thick: float = 1.0e-2,
        L: float = 1.0,
        load_fcn=lambda x: 1.0,
        clamped: bool = False,
        split_disp_bc: bool = False,
    ):
        self.element = ELEMENT
        self.nxe = int(nxe)
        self.E = E
        self.nu = nu
        self.thick = thick
        self.L = float(L)
        self.load_fcn = load_fcn
        self.split_disp_bc = split_disp_bc

        self.kmat = None
        self.force = None
        self.u = None

        assert self.element.ORDER == 2  # 2nd order IGA only currently
        self.nnodes = self.nxe + 2      # p=2 => nnodes = nxe + p
        self.dof_per_node = self.element.dof_per_node
        self.N = self.dof_per_node * self.nnodes

        dpn = self.dof_per_node

        # ----------------------------
        # FIX 1: separate control spacing from knot-span length
        # ----------------------------
        # MAIN CHANGE IS HERE.. SPE
        self.dx_ctrl = self.L / (self.nnodes - 1)   # control point spacing
        self.h_span  = self.L / self.nxe            # knot span length (integration interval)

        # BCs (unchanged)
        if clamped:
            self.element.clamped = True
            if dpn == 3:
                self.bcs = [0, dpn * (self.nnodes - 1)]
            else:
                self.bcs = list(range(dpn)) + list(range(dpn * (self.nnodes - 1), dpn * self.nnodes))
        else:
            self.element.clamped = False
            self.bcs = [0, dpn * (self.nnodes - 1)]

        # p=2 span connectivity: each span uses 3 control points [e, e+1, e+2]
        self.conn = [[e, e + 1, e + 2] for e in range(self.nxe)]

        # matrix sparsity (unchanged)
        self.rowp = [0]
        self.cols = []
        self.nnzb = 0
        for inode in range(self.nnodes):
            if inode == 0:
                current_cols = [0, 1, 2]
            elif inode == 1:
                current_cols = [0, 1, 2, 3]
            elif inode == self.nnodes - 2:
                current_cols = [self.nnodes - 4, self.nnodes - 3, self.nnodes - 2, self.nnodes - 1]
            elif inode == self.nnodes - 1:
                current_cols = [self.nnodes - 3, self.nnodes - 2, self.nnodes - 1]
            else:
                current_cols = [inode - 2, inode - 1, inode, inode + 1, inode + 2]
            self.nnzb += len(current_cols)
            self.rowp += [self.nnzb]
            self.cols += current_cols

        self.rowp = np.array(self.rowp, dtype=int)
        self.cols = np.array(self.cols, dtype=int)

    @property
    def dof_conn(self):
        dpn = self.dof_per_node
        return [[dpn * e + j for j in range(3 * dpn)] for e in range(self.nxe)]

    def _assemble_system(self):
        dpn = self.dof_per_node

        # assemble BSR matrix
        self.data = np.zeros((self.nnzb, dpn, dpn), dtype=np.double)
        self.force = np.zeros(self.N)

        # ----------------------------
        # FIX 2: span bounds must be per knot span, not "3 control points"
        # xbnd = [x_e, x_{e+1}] = [e*h_span, (e+1)*h_span]
        # ----------------------------
        xbnd_vals = [[e * self.h_span, (e + 1) * self.h_span] for e in range(self.nxe)]

        # precompute interior stiffness for speed (still depends on span length)
        interior_kelem = self.element.get_kelem(
            self.E, self.nu, self.thick, self.h_span,
            left_bndry=False, right_bndry=False
        )

        for e in range(self.nxe):
            left_bndry  = (e == 0)
            right_bndry = (e == self.nxe - 1)

            if left_bndry:
                kelem = self.element.get_kelem(self.E, self.nu, self.thick, self.h_span, left_bndry=True, right_bndry=False)
            elif right_bndry:
                kelem = self.element.get_kelem(self.E, self.nu, self.thick, self.h_span, left_bndry=False, right_bndry=True)
            else:
                kelem = interior_kelem

            felem = self.element.get_felem(
                mag=self.load_fcn,
                elem_length=self.h_span,
                left_bndry=left_bndry,
                right_bndry=right_bndry,
                xbnd=xbnd_vals[e],
            )

            local_conn = np.array(self.dof_conn[e], dtype=int)

            # add kelem into LHS sparse structure
            for lblock_row, block_row in enumerate([e, e + 1, e + 2]):
                for colp in range(self.rowp[block_row], self.rowp[block_row + 1]):
                    block_col = self.cols[colp]
                    if block_col in [e, e + 1, e + 2]:
                        lblock_col = block_col - e
                        self.data[colp, :, :] += kelem[
                            dpn * lblock_row: dpn * (lblock_row + 1),
                            dpn * lblock_col: dpn * (lblock_col + 1)
                        ]

            # add felem into RHS
            np.add.at(self.force, local_conn, felem)

        # BCs (unchanged)
        if self.split_disp_bc:
            inode = 0
            for colp in range(self.rowp[inode], self.rowp[inode + 1]):
                block_col = self.cols[colp]
                idof = 0
                for jdof in range(dpn):
                    self.data[colp, idof, jdof] = 0.0
                if block_col == inode:
                    self.data[colp, 0, 0] = 1.0
                    self.data[colp, 0, 1] = 1.0

            inode = 0
            for colp in range(self.rowp[inode], self.rowp[inode + 1]):
                block_col = self.cols[colp]
                idof = 1
                for jdof in range(dpn):
                    self.data[colp, idof, jdof] = 0.0
                if block_col == inode:
                    self.data[colp, 1, 1] = 1.0

            inode = self.nnodes - 1
            for colp in range(self.rowp[inode], self.rowp[inode + 1]):
                block_col = self.cols[colp]
                idof = 0
                for jdof in range(dpn):
                    self.data[colp, idof, jdof] = 0.0
                if block_col == inode:
                    self.data[colp, 0, 0] = 1.0
                    self.data[colp, 0, 1] = 1.0

            self.force[dpn * 0 + 0] = 0.0
            self.force[dpn * 0 + 1] = 0.0
            self.force[dpn * (self.nnodes - 1) + 0] = 0.0

        else:
            # node 0
            for colp in range(self.rowp[0], self.rowp[1]):
                block_col = self.cols[colp]
                for idof in range(dpn):
                    row = idof
                    if row not in self.bcs:
                        continue
                    for jdof in range(dpn):
                        col = dpn * block_col + jdof
                        self.data[colp, idof, jdof] = 1.0 if (row == col) else 0.0

            # last node
            for colp in range(self.rowp[self.nnodes - 1], self.rowp[self.nnodes]):
                block_col = self.cols[colp]
                for idof in range(dpn):
                    row = dpn * (self.nnodes - 1) + idof
                    if row not in self.bcs:
                        continue
                    for jdof in range(dpn):
                        col = dpn * block_col + jdof
                        self.data[colp, idof, jdof] = 1.0 if (row == col) else 0.0

            for bc in self.bcs:
                self.force[bc] = 0.0

        self.kmat = sp.bsr_matrix((self.data, self.cols, self.rowp), shape=(self.N, self.N))

    def direct_solve(self):
        self._assemble_system()
        self.u = sp.linalg.spsolve(self.kmat, self.force)
        return self.u

    @property
    def control_pts(self):
        # ----------------------------
        # FIX 3: control points use control spacing (dx_ctrl), not span length
        # ----------------------------
        return [i * self.dx_ctrl for i in range(self.nnodes)]


    def get_node_pts(self) -> list:
        """
        Quadratic (p=2) 1D IGA: return physical 'node points' using Greville abscissae.
        Output length = nxe + 2.

        u_i = (U[i+1] + U[i+2]) / 2   for i = 0..n_ctrl-1, with n_ctrl = nxe + 2
        x(u) = sum_j N_j(u) * x_ctrl[j]
        """
        nxe = int(self.nxe)
        p = 2
        n_ctrl = nxe + p  # = nxe + 2

        # Open-uniform knot vector on [0,1], length = n_ctrl + p + 1 = nxe + 5
        U = np.array([0.0] * (p + 1) +
                    [i / nxe for i in range(1, nxe)] +
                    [1.0] * (p + 1), dtype=float)

        x_ctrl = np.asarray(self.control_pts, dtype=float)
        if x_ctrl.size != n_ctrl:
            raise ValueError(
                f"Expected {n_ctrl} control points for p=2, nxe={nxe}, "
                f"but got {x_ctrl.size}."
            )

        def find_span(n_ctrl, degree, u, U):
            # Cox–de Boor span search; returns span in [degree, n_ctrl-1]
            if u >= U[-1] - 1e-14:
                return n_ctrl - 1
            low = degree
            high = len(U) - degree - 2
            mid = (low + high) // 2
            while True:
                if u < U[mid]:
                    high = mid - 1
                elif u >= U[mid + 1]:
                    low = mid + 1
                else:
                    return mid
                mid = (low + high) // 2

        
        def basis_functions_and_derivatives(span, u, degree, U, n_deriv=1):
            # Compute nonzero basis functions and first derivatives using Cox-de Boor + derivative formula
            # Returns arrays N[0:degree] and dN[0:degree]
            left = np.zeros(degree+1)
            right = np.zeros(degree+1)
            ndu = np.zeros((degree+1, degree+1))
            ndu[0,0] = 1.0
            for j in range(1, degree+1):
                left[j] = u - U[span+1-j]
                right[j] = U[span+j] - u
                saved = 0.0
                for r in range(j):
                    ndu[j,r] = right[r+1] + left[j-r]
                    temp = ndu[r,j-1]/ndu[j,r]
                    ndu[r,j] = saved + right[r+1]*temp
                    saved = left[j-r]*temp
                ndu[j,j] = saved
            N = ndu[:,degree].copy()
            # derivatives
            ders = np.zeros((n_deriv+1, degree+1))
            a = np.zeros((2, degree+1))
            # compute a triangular table of derivatives
            for r in range(degree+1):
                s1 = 0; s2 = 1
                a[0,0] = 1.0
                for k in range(1, n_deriv+1):
                    d = 0.0
                    rk = r - k
                    pk = degree - k
                    if r >= k:
                        a[s2,0] = a[s1,0]/ndu[pk+1,rk]
                        d = a[s2,0]*ndu[rk,pk]
                    j1 = 1 if rk >= -1 else -rk
                    j2 = k-1 if r-1 <= pk else degree - r
                    for j in range(j1, j2+1):
                        a[s2,j] = (a[s1,j] - a[s1,j-1]) / ndu[pk+1, rk+j]
                        d += a[s2,j]*ndu[rk+j, pk]
                    if r <= pk:
                        a[s2,k] = -a[s1,k-1]/ndu[pk+1, r]
                        d += a[s2,k]*ndu[r, pk]
                    ders[k,r] = d
                    s1, s2 = s2, s1
            # Multiply by correct factors
            for k in range(1, n_deriv+1):
                for j in range(degree+1):
                    ders[k,j] *= degree
            return N, ders[1]
        # Greville abscissae (parametric "nodes"), length n_ctrl = nxe+2 for p=2
        u_nodes = 0.5 * (U[1:1 + n_ctrl] + U[2:2 + n_ctrl])

        node_pts = []
        for u in u_nodes:
            span = find_span(n_ctrl, p, float(u), U)
            N, _ = basis_functions_and_derivatives(span, float(u), p, U, n_deriv=1)

            i0 = span - p  # active basis indices: i0..i0+p
            x = 0.0
            for a in range(p + 1):
                x += N[a] * x_ctrl[i0 + a]
            node_pts.append(float(x))
        # print(f"{node_pts=}\n{self.control_pts=}")

        return node_pts

    def get_max_deflection_greville(self):
        if self.u is None:
            raise RuntimeError("Call direct_solve() before evaluating deflection.")

        dpn = self.dof_per_node

        # control DOFs for w (on control points)
        if self.split_disp_bc:
            if dpn == 3:
                w_ctrl = self.u[0::3] + self.u[2::3]
            elif dpn == 2:
                w_ctrl = self.u[0::2] + self.u[1::2]
            else:
                w_ctrl = self.u[0::dpn]
        else:
            w_ctrl = self.u[0::dpn]

        # NOTE: your get_node_pts() routine is fine; it uses self.control_pts
        xg = np.asarray(self.get_node_pts(), dtype=float)
        w_g = np.zeros_like(xg)

        def _clamp(a, lo, hi):
            return lo if a < lo else (hi if a > hi else a)

        h = float(self.h_span)  # ---------------- FIX 4: use span length for element search
        L = float(self.L)

        for ig, x in enumerate(xg):
            if x <= 0.0:
                e = 0
            elif x >= L:
                e = self.nxe - 1
            else:
                e = int(np.floor(x / h))
                e = _clamp(e, 0, self.nxe - 1)

            x0 = e * h
            xi = (x - x0) / h
            xi = _clamp(xi, 0.0, 1.0)

            left_bndry = (e == 0)
            right_bndry = (e == self.nxe - 1)

            N, _ = get_iga2_basis(float(xi), left_bndry, right_bndry)
            w_g[ig] = N[0] * w_ctrl[e] + N[1] * w_ctrl[e + 1] + N[2] * w_ctrl[e + 2]

        return float(np.max(np.abs(w_g)))
    
    def plot_disp(self, idof:int=0):
        xvec = self.get_node_pts() # not same as control points
        # xvec = self.control_pts
        # print(f"{self.u=}")
        dpn = self.dof_per_node
        w = self.u[idof::dpn]
        if self.split_disp_bc:
            if dpn == 3: # hhd hermite hierarchic disp
                w = self.u[0::3] + self.u[2::3] # wb + ws
            elif dpn == 2: # higd iga hierarchic disp
                w = self.u[0::2] + self.u[1::2] # wb + ws
        plt.figure()
        plt.plot(xvec, w)
        plt.plot(xvec, np.zeros((self.nnodes,)), "k--")
        plt.xlabel("x")
        plt.ylabel("w(x)" if idof == 0 else "th(x)")
        plt.show()  

    def prolongate(self, coarse_soln):
        return self.element.prolongate(coarse_soln, self.L)
    
    def restrict_defect(self, fine_defect):
        return self.element.restrict_defect(fine_defect, self.L)