import numpy as np
from .basis import second_order_quadrature
from .basis import get_iga2_basis_2d_all

# based on Oesterle paper, "A shear deformable, rotation-free isogeometric shell formulation"
# https://www.sciencedirect.com/science/article/pii/S004578251630202X

class HierarchicIsogeometricDispElement9:
    """hierarchic displacement element HIGD with 2nd order IGA basis to allow 2nd derivatives in weak form"""
    def __init__(self, reduced_integrated:bool=False):              
        self.dof_per_node = 3
        self.nodes_per_elem = 9
        self.ndof = 27
        self.reduced_integrated = reduced_integrated
        self.clamped = True
        self.ORDER = 2 # 2nd order IGA

    def get_kelem(
        self,
        E:float, nu:float, thick:float, elem_xpts:np.ndarray,
        left_bndry:bool,
        right_bndry:bool,
        bot_bndry:bool,
        top_bndry:bool,
    ):
        # not reduced integrated
        pts, wts = second_order_quadrature()

        kelem = np.zeros((self.ndof, self.ndof))
        EI = E * thick**3 / 12.0 / (1.0 - nu**2)
        ks = 5.0 / 6.0
        G = E / 2.0 / (1 + nu)
        ksGA = ks * G * thick
        x = elem_xpts[0::3]
        y = elem_xpts[1::3]

        Db = EI * np.array([
            [1.0,  nu,          0.0],
            [nu,   1.0,         0.0],
            [0.0,  0.0, (1.0-nu)/2.0],
        ])

        for ipt in range(9):
            ii, jj = ipt % 3, ipt // 3
            _xi = pts[ii]; _eta = pts[jj]
            wt = wts[ii] * wts[jj]
            wt *= 0.25 # since xi in [0,1]
            xi = 0.5 * (_xi + 1)
            eta = 0.5 * (_eta + 1)

            # basis + 2nd parametric derivs
            N, Nxi, Neta, Nxixi, Netaeta, Nxieta = \
                get_iga2_basis_2d_all(xi, eta, left_bndry, right_bndry, bot_bndry, top_bndry)

            # geometry first derivatives (still computed the same)
            x_xi  = np.dot(Nxi,  x);  x_eta = np.dot(Neta, x)
            y_xi  = np.dot(Nxi,  y);  y_eta = np.dot(Neta, y)

            # ---- DIAGONAL MAP ASSUMPTION ----
            # enforce/assume: x_eta = 0, y_xi = 0
            # (you can leave the dot-products above for debugging; but don't use them)
            # x_eta = 0.0
            # y_xi  = 0.0

            # Jacobian determinant for diagonal map
            J = x_xi * y_eta

            # inverse map first derivatives:
            xi_x  = 1.0 / x_xi
            eta_y = 1.0 / y_eta
            xi_y  = 0.0
            eta_x = 0.0

            # geometry second derivatives needed for 2nd-order chain rule
            x_xixi   = np.dot(Nxixi, x)      # ∂²x/∂ξ²
            y_etaeta = np.dot(Netaeta, y)    # ∂²y/∂η²

            # inverse map second derivatives (diagonal case)
            # xi_xx  = d/dx (xi_x) = -x_xixi / x_xi^3
            # eta_yy = d/dy (eta_y)= -y_etaeta / y_eta^3
            xi_xx  = -x_xixi   / (x_xi**3)
            eta_yy = -y_etaeta / (y_eta**3)

            # cross second derivatives are zero for diagonal map
            # xi_xy  = 0.0
            # eta_xy = 0.0

            # ---- physical 2nd derivatives of basis (FULL chain rule, diagonal map) ----
            # Nxx = N_xixi*(xi_x)^2 + N_xi*xi_xx
            # Nyy = N_etaeta*(eta_y)^2 + N_eta*eta_yy
            # Nxy = N_xieta*xi_x*eta_y
            Nxx = Nxixi * (xi_x**2) + Nxi * xi_xx
            Nyy = Netaeta * (eta_y**2) + Neta * eta_yy
            Nxy = Nxieta * (xi_x * eta_y)

            # ---- bending B-matrix: [kxx, kyy, kxy] with kxy = 2*k12 ----
            # DOF layout per node: [wb, ws1, ws2] repeated (so 27 dof for 9 nodes)
            Bb = np.zeros((3, self.ndof))

            # k11 = -(wb + ws2)_{,xx}
            Bb[0, 0::3] -= Nxx
            Bb[0, 2::3] -= Nxx

            # k22 = -(wb + ws1)_{,yy}
            Bb[1, 0::3] -= Nyy
            Bb[1, 1::3] -= Nyy

            # kxy = 2*k12 = -(2*wb + ws1 + ws2)_{,xy}
            Bb[2, 0::3] -= 2.0 * Nxy
            Bb[2, 1::3] -= 1.0 * Nxy
            Bb[2, 2::3] -= 1.0 * Nxy

            # add bending energy
            kelem += (Bb.T @ Db @ Bb) * (wt * J)

            # ---- transverse shear (simple): 2*e13 = ws1,x ; 2*e23 = ws2,y ----
            # first physical derivatives:
            Nx = Nxi * xi_x + Neta * eta_x
            Ny = Nxi * xi_y + Neta * eta_y

            Bs = np.zeros((2, self.ndof))
            # gamma_xz (or 2e13) = ws1,x
            Bs[0, 1::3] = Nx
            # gamma_yz (or 2e23) = ws2,y
            Bs[1, 2::3] = Ny

            Ds = ksGA * np.eye(2)  # isotropic shear in xz,yz with same ksGA
            kelem += (Bs.T @ Ds @ Bs) * (wt * J)

        return kelem

    def get_felem(
        self,
        mag,                      # callable: mag(x,y)
        elem_xpts: np.ndarray,    # length 27: [x0,y0,?, x1,y1,?, ...] (same as kelem)
        left_bndry: bool,
        right_bndry: bool,
        bot_bndry: bool,
        top_bndry: bool,
    ):
        """
        Consistent load vector for transverse pressure q(x,y) applied to the TOTAL transverse deflection:
            w = wb + ws1 + ws2   (matches your bending kinematics that use wb, ws1, ws2 in curvatures)

        If instead your theory wants load only on wb, just change the last three lines where fe is assembled.
        """
        pts, wts = second_order_quadrature()

        felem = np.zeros(self.ndof)

        x = elem_xpts[0::3]
        y = elem_xpts[1::3]

        for ipt in range(9):
            ii, jj = ipt % 3, ipt // 3
            _xi = pts[ii]; _eta = pts[jj]
            wt = wts[ii] * wts[jj]
            wt *= 0.25 # since xi in [0,1]
            xi = 0.5 * (_xi + 1)
            eta = 0.5 * (_eta + 1)
            # basis (need N to map load; Nxi/Neta to get geometry jacobian)
            # If you don't have a lighter function, you can call get_iga2_basis_2d_all and ignore extras.
            N, Nxi, Neta, *_ = get_iga2_basis_2d_all(
                xi, eta, left_bndry, right_bndry, bot_bndry, top_bndry
            )

            # geometry jacobian determinant
            x_xi  = np.dot(Nxi,  x);  x_eta = np.dot(Neta, x)
            y_xi  = np.dot(Nxi,  y);  y_eta = np.dot(Neta, y)
            J = x_xi * y_eta - x_eta * y_xi

            # physical point (x,y) at this quadrature point
            xq = float(np.dot(N, x))
            yq = float(np.dot(N, y))

            q = float(mag(xq, yq))   # distributed transverse load

            # consistent nodal load contribution: ∫ N^T q dA = Σ N_i q * wt * J
            fN = q * wt * J * N  # length 9

            # Apply load to the DOFs that contribute to transverse displacement.
            # With your kinematics you used: (wb + ws2), (wb + ws1), etc.
            # A consistent choice is to load w_total = wb + ws1 + ws2.
            felem[0::3] += fN  # wb
            felem[1::3] += fN  # ws1
            felem[2::3] += fN  # ws2

        return felem

    def _build_1d_restriction_matrix(self, nxe_c):
        # restriction operator or P^T global
        n_coarse = nxe_c + 2
        nxe_f = 2 * nxe_c
        n_fine = nxe_f + 2
        R = np.zeros((n_coarse, n_fine))
        counts = 1e-20 * np.ones((n_coarse, n_fine))

        # see asym_iga/_python_demo/2_iga/4_multigrid_oned.ipynb
        for ielem_c in range(nxe_c):
            # left half elem
            left_felem = 2 * ielem_c
            l_mat = np.array([
                [0.75, 0.25, 0.0],
                [0.25, 0.75, 0.75],
                [0.0, 0.0, 0.25],
            ])
            if ielem_c == 0:
                l_mat[0,:2] = np.array([1.0, 0.5])
                l_mat[1,:2] = np.array([0.0, 0.5])
            if ielem_c == nxe_c - 1:
                l_mat[1, 2] = 0.5
                l_mat[2, 2] = 0.5

            # print(F"{l_mat=} {l_mat.shape=}")

            l_nz_mat = l_mat / (l_mat + 1e-20)
            R[ielem_c:(ielem_c+3), left_felem:(left_felem+3)] += l_mat
            counts[ielem_c:(ielem_c+3), left_felem:(left_felem+3)] += l_nz_mat

            # right half elem
            right_felem = 2 * ielem_c + 1
            r_mat = np.array([
                [0.25, 0.0, 0.0],
                [0.75, 0.75, 0.25],
                [0.0, 0.25, 0.75],
            ])
            if ielem_c == 0:
                r_mat[0,0] = 0.5
                r_mat[1,0] = 0.5
            if ielem_c == nxe_c - 1:
                r_mat[1, 1:] = np.array([0.5, 0.0])
                r_mat[2, 1:] = np.array([0.5, 1.0])
            r_nz_mat = r_mat / (r_mat + 1e-20)
            R[ielem_c:(ielem_c+3), right_felem:(right_felem+3)] += r_mat
            counts[ielem_c:(ielem_c+3), right_felem:(right_felem+3)] += r_nz_mat

        # normalize it by weights added into each spot?
        R /= counts
        # print(f"{R=}")
        return R

    def _build_2d_restr_matrix(self, nxe_c:int, _nye_c):
        # see examples/iga/_python_demo/2_iga
        n_coarse = nxe_c + 2
        nxe_f = 2 * nxe_c
        n_fine = nxe_f + 2

        R = self._build_1d_restriction_matrix(nxe_c)        
        R_twod = np.zeros((n_coarse**2, n_fine**2))

        for inode_c in range(n_coarse**2):
            ix_c, iy_c = inode_c % n_coarse, inode_c // n_coarse

            for inode_f in range(n_fine**2):
                ix_f, iy_f = inode_f % n_fine, inode_f // n_fine

                R_twod[inode_c, inode_f] = R[ix_c, ix_f] * R[iy_c, iy_f]

        return R_twod

    def apply_bcs_2d(self, u: np.ndarray, nx: int, ny: int, mode: str = "prolong"):
        dpn = 3
        assert self.dof_per_node == dpn
        U = u.reshape((nx * ny, dpn))

        def node(i, j): return i + nx * j

        for j in range(ny):
            for i in range(nx):
                on_left   = (i == 0)
                on_right  = (i == nx - 1)
                on_bottom = (j == 0)
                on_top    = (j == ny - 1)
                on_edge = on_left or on_right or on_bottom or on_top
                if not on_edge:
                    continue

                k = node(i, j)

                # bottom-left corner special
                if on_left and on_bottom:
                    U[k, :] = 0.0
                    continue

                if mode == "restrict":
                    # constraint eqn lives on w_b row on ALL edges
                    U[k, 0] = 0.0                 # zero w_b row (w_b + w_s1 + w_s2 = 0)

                    if on_left:
                        U[k, 1] = 0.0             # w_s1 = 0
                    if on_bottom:
                        U[k, 2] = 0.0             # w_s2 = 0

                else:  # "prolong": projector/substitution
                    if on_left:
                        U[k, 1] = 0.0             # w_s1 = 0
                    if on_bottom:
                        U[k, 2] = 0.0             # w_s2 = 0

                    wb, ws1, ws2 = U[k, 0], U[k, 1], U[k, 2]
                    if on_bottom:                 # ws2=0 -> wb+ws1=0
                        U[k, 1] = -wb
                    elif on_left:                 # ws1=0 -> wb+ws2=0
                        U[k, 2] = -wb
                    else:                         # general edge -> wb = -(ws1+ws2)
                        U[k,1] = -(wb + ws2)

        return u

    def check_bcs_2d(self, u: np.ndarray, nx: int, ny: int, tol: float = 1e-10, verbose: bool = True):
        """
        Debug check for *prolongation* BCs:
        left:   w_s1 = 0
        bottom: w_s2 = 0
        edges:  w_b + w_s1 + w_s2 = 0
        bottom-left: w_b = w_s1 = w_s2 = 0
        Returns dict of max violations.
        """
        dpn = 3
        assert self.dof_per_node == dpn
        U = u.reshape((nx * ny, dpn))

        def node(i, j): return i + nx * j

        max_left_ws1 = 0.0
        max_bottom_ws2 = 0.0
        max_edge_sum = 0.0
        max_corner = 0.0

        for j in range(ny):
            for i in range(nx):
                on_left   = (i == 0)
                on_right  = (i == nx - 1)
                on_bottom = (j == 0)
                on_top    = (j == ny - 1)
                on_edge = on_left or on_right or on_bottom or on_top
                if not on_edge:
                    continue

                k = node(i, j)
                wb, ws1, ws2 = U[k, 0], U[k, 1], U[k, 2]

                if on_left and on_bottom:
                    max_corner = max(max_corner, abs(wb), abs(ws1), abs(ws2))
                    continue

                if on_left:
                    max_left_ws1 = max(max_left_ws1, abs(ws1))
                if on_bottom:
                    max_bottom_ws2 = max(max_bottom_ws2, abs(ws2))

                max_edge_sum = max(max_edge_sum, abs(wb + ws1 + ws2))

        out = {
            "max_left_ws1": max_left_ws1,
            "max_bottom_ws2": max_bottom_ws2,
            "max_edge_sum": max_edge_sum,
            "max_bottom_left_corner": max_corner,
        }

        if verbose:
            print("BC check (prolong):")
            for k, v in out.items():
                flag = " OK" if v <= tol else " FAIL"
                print(f"  {k}: {v:.3e}{flag}")

        return out



    def prolongate(self, coarse_soln: np.ndarray):
        dpn = self.dof_per_node
        nnode_c = coarse_soln.size // dpn
        nx_c = int(round(np.sqrt(nnode_c))); ny_c = nnode_c // nx_c
        nxe_c, nye_c = nx_c - 2, ny_c - 2

        nx_f, ny_f = 2*nxe_c + 2, 2*nye_c + 2 # shortcut cause nxe_f = 2 * nxe_c
        nnode_f = nx_f * ny_f
        ndof_f = dpn * nnode_f

        R = self._build_2d_restr_matrix(nxe_c, nye_c)   # (nnode_c, nnode_f)
        P = R.T                                         # (nnode_f, nnode_c)

        fine = np.zeros(ndof_f)
        wts  = np.zeros(ndof_f)
        ones_c = np.ones(nnode_c)

        for d in range(dpn):
            fine[d::dpn] = P @ coarse_soln[d::dpn]
            wts[d::dpn]  = P @ ones_c

        fine /= wts
        # if hasattr(self, "apply_bcs_2d"):  # optional
        fine = self.apply_bcs_2d(fine, nx_f, ny_f, mode="prolong")
        # self.check_bcs_2d(fine, nx_f, ny_f) # DEBUGGINg
        return fine


    def restrict_defect(self, fine_defect: np.ndarray):
        dpn = self.dof_per_node
        nnode_f = fine_defect.size // dpn
        nx_f = int(round(np.sqrt(nnode_f))); ny_f = nnode_f // nx_f
        nxe_f, nye_f = nx_f - 2, ny_f - 2

        nxe_c, nye_c = nxe_f // 2, nye_f // 2
        nx_c, ny_c = nxe_c + 2, nye_c + 2
        nnode_c = nx_c * ny_c
        ndof_c = dpn * nnode_c

        R = self._build_2d_restr_matrix(nxe_c, nye_c)   # (nnode_c, nnode_f)

        coarse = np.zeros(ndof_c)
        for d in range(dpn):
            coarse[d::dpn] = R @ fine_defect[d::dpn]

        # if hasattr(self, "apply_bcs_2d"):  # optional
        coarse = self.apply_bcs_2d(coarse, nx_c, ny_c, mode="restrict")
        return coarse