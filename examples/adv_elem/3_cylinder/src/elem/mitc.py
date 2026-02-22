import numpy as np
import scipy.sparse as sp

from .basis import second_order_quadrature, first_order_quadrature
from .basis import get_lagrange_basis_2d_all_standard

import numpy as np
import scipy.sparse as sp

class MITCShellElement:
    """
    Q4 Reissner–Mindlin shell element for linear structure.
    Uses standard prolongator.

    Adapted from / extension of adv_elems/2_plate/src/elem/mitc_elem.py
    """

    def __init__(
        self,
        prolong_mode: str = "locking-global",  # 'standard'
        lam: float = 1e-2,
        n_lock_sweeps:int=10,
        omega:float=0.5,
        debug:bool=False,
    ):
        assert prolong_mode in ["locking-global", "locking-local", "standard", "energy-jacobi"]

        self.dof_per_node = 6
        self.nodes_per_elem = 4
        self.ndof = self.dof_per_node * self.nodes_per_elem
        self.debug = debug

        self.prolong_mode = prolong_mode
        self.clamped = False
        self.lam = float(lam)
        self.n_lock_sweeps = n_lock_sweeps
        self.omega = float(omega)

        # cache for prolong/restrict operators
        self._P1_cache = {}   # key: nxe_coarse -> P1 (csr)
        self._P2_cache = {}   # key: nxe_coarse -> P2 (csr)
        self._P2_u3_cache = {}
        self._lock_P_cache = {}
        self._kmat_cache = {}

    # ----------------------------
    # Element stiffness (MITC4)
    # ----------------------------
    def get_kelem(self, E: float, nu: float, thick: float, elem_xpts: np.ndarray):
        """
        Method based on k_add_jacobian_fast assembly from MITC shell GPU code
        """
        pts, wts = first_order_quadrature()

        kelem = np.zeros((self.ndof, self.ndof))

        comp_data = (E, nu, thick)
        fn = self._get_shell_normals(elem_xpts)

        for iquad in range(4):
            
            # get quad pt
            ixi, ieta = iquad % 2, iquad // 2
            xi = pts[ixi]; eta = pts[ieta]
            wt = wts[ixi] * wts[ieta]
            pt = [xi, eta]

            # print(f"{pt=}")
            # get xpts related data
            detXd = self._get_detXd(pt, elem_xpts, fn)
            scale = detXd * wt
            XdinvT, Tmat, XdinvzT = self._get_shell_rotations(
                pt, elem_xpts, fn
            )

            # now loop over each of 24 DOF elem col
            for ideriv in range(self.ndof):
                p_vars = np.zeros(self.ndof)
                p_vars[ideriv] = 1.0
                local_mat_col = np.zeros(self.ndof)

                local_mat_col += self._add_jac_col_drill(
                    pt, scale, elem_xpts, fn,
                    XdinvT, Tmat, XdinvzT, comp_data,
                    p_vars
                )

                local_mat_col += self._add_jac_col_bending(
                    pt, scale, elem_xpts, fn,
                    XdinvT, Tmat, XdinvzT, comp_data,
                    p_vars
                )

                local_mat_col += self._add_jac_col_tying(
                    pt, scale, elem_xpts, fn,
                    XdinvT, Tmat, XdinvzT, comp_data,
                    p_vars
                )

                # add mat col into the kelem
                kelem[:, ideriv] += local_mat_col

        # import matplotlib.pyplot as plt
        # plt.imshow(kelem)
        # plt.show()

        return kelem
    
    def _interp_fields(self, pt:list, vpn:int, num_fields:int, vec:np.ndarray):
        out = np.zeros(num_fields)
        N, _, _ = get_lagrange_basis_2d_all_standard(pt[0], pt[1])
        # print(f"{pt=} {N=}")
        for ifield in range(num_fields):
            for inode in range(4):
                out[ifield] += N[inode] * vec[vpn * inode + ifield]
        return out
    
    def _interp_fields_transpose(self, pt:list, vpn:int, num_fields:int, vec:np.ndarray):
        out = np.zeros(vpn * 4)
        N, _, _ = get_lagrange_basis_2d_all_standard(pt[0], pt[1])
        for ifield in range(num_fields):
            for inode in range(4):
                out[vpn * inode + ifield] += N[inode] * vec[ifield]
        return out

    def _interp_fields_grad(self, pt:list, vpn:int, num_fields:int, vec:np.ndarray):
        dxi = np.zeros(num_fields); deta = np.zeros(num_fields)
        
        _, dNdxi, dNdeta = get_lagrange_basis_2d_all_standard(pt[0], pt[1])
        for ifield in range(num_fields):
            for inode in range(4):
                dxi[ifield] += dNdxi[inode] * vec[vpn * inode + ifield]
                deta[ifield] += dNdeta[inode] * vec[vpn * inode + ifield]
        return dxi, deta
    
    def _interp_fields_grad_transpose(self, pt:list, vpn:int, num_fields:int, dxi:np.ndarray, deta:np.ndarray):
        dout = np.zeros(vpn * 4)

        _, dNdxi, dNdeta = get_lagrange_basis_2d_all_standard(pt[0], pt[1])
        for ifield in range(num_fields):
            for inode in range(4):
                dout[vpn * inode + ifield] += dxi[ifield] * dNdxi[inode] + \
                    deta[ifield] * dNdeta[inode]
        return dout

    def _get_shell_normals(self, elem_xpts:np.ndarray):
        fn = np.zeros(12)

        for i in range(4):
            # node pt
            ixi, ieta = i % 2, i // 2
            xi = -1 + 2.0 * ixi
            eta = -1 + 2.0 * ieta
            node_pt = [xi, eta]
            # print(f"{elem_xpts=}")

            dX_dxi, dX_deta = self._interp_fields_grad(node_pt, vpn=3, num_fields=3, vec=elem_xpts)
            normal = np.cross(dX_dxi, dX_deta)
            normal /= np.linalg.norm(normal)
            fn[3*i:(3*i+3)] += normal

            # xpt = self._interp_fields(node_pt, vpn=3, num_fields=3, vec=elem_xpts)
            # print(f"{dX_dxi=} {dX_deta=} {normal=}")
        return fn
    
    def _get_detXd(self, pt:list, elem_xpts:np.ndarray, fn:np.ndarray):
        n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
        dX_dxi, dX_deta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)        
        Xd = np.column_stack((dX_dxi, dX_deta, n0))
        _detXd = np.linalg.det(Xd)
        return np.abs(_detXd)
    
    def _shell_compute_transform(
        self, 
        n0:np.ndarray, 
        ref_axis:np.ndarray=np.array([1, 0, 0]),
    ):
        
        nhat = n0 / np.linalg.norm(n0)
        t1 = ref_axis
        t1hat = t1 - np.dot(t1, nhat) * nhat
        t1hat /= np.linalg.norm(t1hat)
        t2hat = np.cross(nhat, t1hat)
        Tmat = np.column_stack((t1hat, t2hat, nhat))
        # should just be eye(3) or rotation mat version of that often times
        # but not on cylindrical shell, then changes some depending on curvilinear spot
        # print(f"{Tmat=}")
        return Tmat
    
    def _get_shell_rotations(self, pt, elem_xpts, fn):
        # compute Tmat, XdinvT, XdinvzT transformation matrices for shell rotation
        n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
        dX_dxi, dX_deta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)        
        nxi, neta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=fn)
        Xd = np.column_stack((dX_dxi, dX_deta, n0))
        Xdz = np.column_stack((nxi, neta, np.zeros(3)))
        Tmat = self._shell_compute_transform(n0, ref_axis=np.array([1, 0, 0]))

        Xdinv = np.linalg.inv(Xd)
        XdinvT = Xdinv @ Tmat
        XdinvzT = -Xdinv @ Xdz @ XdinvT

        # print(f"{XdinvT=} {Tmat=} {XdinvzT=}")
        return XdinvT, Tmat, XdinvzT
        
    def _add_jac_col_drill(
        self, pt:list, scale:float, 
        elem_xpts:np.ndarray, fn:np.ndarray,
        XdinvT:np.ndarray, Tmat:np.ndarray, 
        XdinvzT:np.ndarray, comp_data:list,
        p_vars:np.ndarray
    ):

        # just the linear drill strains here (so no nonlinearity)
        # et = self._compute_drill_strain(pt, Tmat, XdinvT, p_vars)
        # done explicitly here
        u0xi, u0eta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
        u0xn = np.column_stack((u0xi, u0eta, np.zeros(3)))
        u0 = self._interp_fields(pt, vpn=6, num_fields=6, vec=p_vars)
        thx, thy, thz = u0[3], u0[4], u0[5]
        C0 = np.array([ # rotation matrix for drill strain
            [1.0, thz, -thy],
            [-thz, 1.0, thx],
            [thy, -thx, 1.0]
        ])
        C = Tmat.T @ C0 @ Tmat
        u0x = Tmat.T @ u0xn @ XdinvT
        # eval drill strain
        et = 0.5 * (C[1,0] + u0x[1,0] - C[0,1] - u0x[0,1])

        # now compute drill stress
        # st = self._compute_drill_stress(scale, comp_data, et)
        E, nu, thick = comp_data
        G = E / 2.0 / (1.0 + nu)
        ks = 5.0 / 6.0
        DRILL_REG = 10.0
        As = ks * G * thick
        A_drill = As * DRILL_REG
        st = scale * A_drill * et # equiv to et_bar

        # now do the transpose version of the drill strains
        # mat_col = self._compute_drill_strain_transpose(pt, Tmat, XdinvT, st)
        u0x_hat = np.zeros_like(u0x); C_hat = np.zeros_like(C)
        u0x_hat[1,0] += 0.5 * st
        C_hat[1,0] += 0.5 * st
        u0x_hat[0,1] -= 0.5 * st
        C_hat[0,1] -= 0.5 * st

        C0_hat = Tmat @ C_hat @ Tmat.T
        u0xn_hat = Tmat @ u0x_hat @ XdinvT.T
        u0xi_hat = u0xn_hat[:,0]
        u0eta_hat = u0xn_hat[:,1]

        thx_hat = C0_hat[1,2] - C0_hat[2,1]
        thy_hat = C0_hat[2,0] - C0_hat[0,2]
        thz_hat = C0_hat[0,1] - C0_hat[1,0]

        u0_sens = np.zeros(6)
        u0_sens[3] += thx_hat
        u0_sens[4] += thy_hat
        u0_sens[5] += thz_hat

        mat_col = self._interp_fields_transpose(
            pt, vpn=6, num_fields=6, vec=u0_sens
        )
        mat_col += self._interp_fields_grad_transpose(
            pt, vpn=6, num_fields=3, dxi=u0xi_hat, deta=u0eta_hat
        )
        
        return mat_col
    
    def _compute_director(self, p_vars, fn):
        d = np.zeros(12)
        for i in range(4):
            d[3*i:(3*i+3)] += np.cross(
                p_vars[(6*i+3):(6*i+6)],
                fn[3*i:(3*i+3)],
            )
        return d
    
    def _compute_director_sens(self, fn, d_bar):
        mat_col = np.zeros(24)
        for i in range(4):
            mat_col[(6*i+3):(6*i+6)] += np.cross(
                fn[3*i:(3*i+3)],
                d_bar[3*i:(3*i+3)],
            )
        return mat_col
    
    def _compute_mitc_tying_strains(self, elem_xpts, fn, p_vars, p_d):
        # compute each of the tying strains (linear form only here)

        # 9 total tying strains in combinations of 2 and 1 per each of 5 strains
        # order is [2 x g11, 2 x g22, 1 x g12, 2 x g23, 2 x g13]
        ety = np.zeros(9) 

        # g11 tying strain
        for j in range(2):
            # tying pt
            pt = [0.0, -1 + 2 * j]
            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            ety[j] = np.dot(Uxi, Xxi)

        # g22 tying strain
        for j in range(2):
            # tying pt
            pt = [-1 + 2 * j, 0.0]
            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            ety[2 + j] = np.dot(Ueta, Xeta)

        # g12 tying strain
        pt = [0.0, 0.0]
        Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
        Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
        ety[4] = 0.5 * (np.dot(Uxi, Xeta) + np.dot(Ueta, Xxi))

        # g23 tying strain
        for j in range(2):
            # tying pt
            pt = [-1 + 2 * j, 0.0]
            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
            d0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=p_d)
            ety[5 + j] = 0.5 * (np.dot(Xeta, d0) + np.dot(n0, Ueta))

        # g13 tying strain
        for j in range(2):
            # tying pt
            pt = [0.0, -1 + 2 * j]
            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
            d0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=p_d)
            ety[7 + j] = 0.5 * (np.dot(Xxi, d0) + np.dot(n0, Uxi))

        return ety
    
    def _compute_mitc_tying_strains_transpose(self, elem_xpts, fn, p_vars, p_d, ety_bar):
        """
        Transpose/adjoint of _compute_mitc_tying_strains.

        Inputs
        ------
        elem_xpts : geometry nodal vector (used in forward for Xxi/Xeta)
        fn        : nodal director/normal vector (used in forward for n0)
        p_vars    : nodal dofs (num_fields=6)
        p_d       : nodal director-like dofs (num_fields=3)
        ety_bar   : adjoint of ety, shape (9,)

        Returns
        -------
        mat_col : sensitivity w.r.t. p_vars (same shape as p_vars)
        d_bar   : sensitivity w.r.t. p_d   (same shape as p_d)
        """
        # ety_bar = np.asarray(ety_bar, dtype=float).reshape(9,)

        mat_col = np.zeros_like(p_vars, dtype=float)
        d_bar   = np.zeros_like(p_d, dtype=float)

        # -------------------------
        # g11 tying strain (2 pts): ety[j] = dot(Uxi, Xxi)
        # -------------------------
        for j in range(2):
            pt = [0.0, -1.0 + 2.0 * j]

            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)

            s = float(ety_bar[j])

            # ety = dot(Uxi, Xxi)
            Uxi_bar  = s * Xxi
            Ueta_bar = np.zeros_like(Ueta)

            # push back to p_vars via grad-transpose
            mat_col += self._interp_fields_grad_transpose(
                pt, vpn=6, num_fields=3, dxi=Uxi_bar, deta=Ueta_bar
            )

        # -------------------------
        # g22 tying strain (2 pts): ety[2+j] = dot(Ueta, Xeta)
        # -------------------------
        for j in range(2):
            pt = [-1.0 + 2.0 * j, 0.0]

            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)

            s = float(ety_bar[2 + j])

            Uxi_bar  = np.zeros_like(Uxi)
            Ueta_bar = s * Xeta

            mat_col += self._interp_fields_grad_transpose(
                pt, vpn=6, num_fields=3, dxi=Uxi_bar, deta=Ueta_bar
            )

        # -------------------------
        # g12 tying strain (center):
        # ety[4] = 0.5*(dot(Uxi,Xeta) + dot(Ueta,Xxi))
        # -------------------------
        pt = [0.0, 0.0]
        Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
        Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)

        s = float(ety_bar[4])

        Uxi_bar  = 0.5 * s * Xeta
        Ueta_bar = 0.5 * s * Xxi

        mat_col += self._interp_fields_grad_transpose(
            pt, vpn=6, num_fields=3, dxi=Uxi_bar, deta=Ueta_bar
        )

        # -------------------------
        # g23 tying strain (2 pts):
        # ety[5+j] = 0.5*(dot(Xeta,d0) + dot(n0,Ueta))
        # -------------------------
        for j in range(2):
            pt = [-1.0 + 2.0 * j, 0.0]

            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
            d0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=p_d)

            s = float(ety_bar[5 + j])

            # ety = 0.5*(dot(Xeta,d0) + dot(n0,Ueta))
            # d0_bar from dot(Xeta,d0)
            d0_bar = 0.5 * s * Xeta

            # Ueta_bar from dot(n0,Ueta)
            Uxi_bar  = np.zeros_like(Uxi)
            Ueta_bar = 0.5 * s * n0

            # push back to p_vars and p_d
            mat_col += self._interp_fields_grad_transpose(
                pt, vpn=6, num_fields=3, dxi=Uxi_bar, deta=Ueta_bar
            )
            d_bar += self._interp_fields_transpose(
                pt, vpn=3, num_fields=3, vec=d0_bar
            )

        # -------------------------
        # g13 tying strain (2 pts):
        # ety[7+j] = 0.5*(dot(Xxi,d0) + dot(n0,Uxi))
        # -------------------------
        for j in range(2):
            pt = [0.0, -1.0 + 2.0 * j]

            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
            d0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=p_d)

            s = float(ety_bar[7 + j])

            # ety = 0.5*(dot(Xxi,d0) + dot(n0,Uxi))
            d0_bar = 0.5 * s * Xxi

            Uxi_bar  = 0.5 * s * n0
            Ueta_bar = np.zeros_like(Ueta)

            mat_col += self._interp_fields_grad_transpose(
                pt, vpn=6, num_fields=3, dxi=Uxi_bar, deta=Ueta_bar
            )
            d_bar += self._interp_fields_transpose(
                pt, vpn=3, num_fields=3, vec=d0_bar
            )

        return mat_col, d_bar
    
    def _1d_tying_interp(self, u, vec: np.ndarray):
        N0 = 0.5 * (1.0 - u)
        N1 = 0.5 * (1.0 + u)
        return N0 * vec[0] + N1 * vec[1]

    def _1d_tying_interp_transpose(self, u, out_bar: float):
        """
        Transpose of: out = N0*v0 + N1*v1
        Returns contributions to [v0_bar, v1_bar].
        """
        N0 = 0.5 * (1.0 - u)
        N1 = 0.5 * (1.0 + u)
        return np.array([N0 * out_bar, N1 * out_bar], dtype=float)

    def _interp_tying_strains(self, quad_pt, p_ety):
        xi = quad_pt[0]; eta = quad_pt[1]
        gty = np.zeros((3, 3))

        # membrane normals
        gty[0, 0] = self._1d_tying_interp(u=eta, vec=[p_ety[0], p_ety[1]]) # g11
        gty[1, 1] = self._1d_tying_interp(u=xi,  vec=[p_ety[2], p_ety[3]]) # g22

        # in-plane shear (symmetric)
        gty[0, 1] = gty[1, 0] = p_ety[4] # g12

        # transverse shear terms (symmetric)
        val12 = self._1d_tying_interp(u=xi,  vec=[p_ety[5], p_ety[6]])
        gty[1, 2] = gty[2, 1] = val12 # g23

        val02 = self._1d_tying_interp(u=eta, vec=[p_ety[7], p_ety[8]])
        gty[0, 2] = gty[2, 0] = val02 # g12

        return gty
    
    def _interp_tying_strains_transpose(self, quad_pt, p_gty):
        xi = quad_pt[0]; eta = quad_pt[1]
        p_ety = np.zeros(9, dtype=float)

        # gty[0,0]
        p_ety[0:2] += self._1d_tying_interp_transpose(
            u=eta, out_bar=float(p_gty[0, 0])
        )

        # gty[1,1]
        p_ety[2:4] += self._1d_tying_interp_transpose(
            u=xi, out_bar=float(p_gty[1, 1])
        )

        # gty[0,1] = gty[1,0] = p4
        p_ety[4] += float(p_gty[0, 1]) + float(p_gty[1, 0])

        # gty[1,2] = gty[2,1]
        shear12_bar = float(p_gty[1, 2]) + float(p_gty[2, 1])
        p_ety[5:7] += self._1d_tying_interp_transpose(
            u=xi, out_bar=shear12_bar
        )

        # gty[0,2] = gty[2,0]
        shear02_bar = float(p_gty[0, 2]) + float(p_gty[2, 0])
        p_ety[7:9] += self._1d_tying_interp_transpose(
            u=eta, out_bar=shear02_bar
        )

        return p_ety

    def _add_jac_col_tying(
        self, pt:list, scale:float, 
        elem_xpts:np.ndarray, fn:np.ndarray,
        XdinvT:np.ndarray, Tmat:np.ndarray, 
        XdinvzT:np.ndarray, comp_data:list,
        p_vars:np.ndarray
    ):

        # just do pforward and hrev part cause linear
        p_d = self._compute_director(p_vars, fn)
        p_ety = self._compute_mitc_tying_strains(elem_xpts, fn, p_vars, p_d)
        p_gty = self._interp_tying_strains(pt, p_ety)
        # sym mat rotate frame
        p_e0ty = XdinvT.T @ p_gty @ XdinvT
        # membrane strains
        p_em = np.array([p_e0ty[0,0], p_e0ty[1,1], 2.0 * p_e0ty[0,1]]) 
        # trv shear strains 
        p_es = 2.0 * np.array([p_e0ty[0,2], p_e0ty[1,2]]) # or just sym sums here which means reverse is copy to each component

        # tangent stiffness 2D matrix
        E, nu, thick = comp_data
        C = E / (1.0 - nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1.0 - nu) / 2.0]
        ])
        # print(f"{E=} {nu=} {thick=}")
        A = C * thick * scale # A matrix from shell theory
        # B matrix is zero
        # trv shear A matrix
        ks = 5.0 / 6.0
        As = np.eye(2) * A[-1,-1] * ks

        # membrane and trv shear stresses
        h_sm = np.dot(A, p_em)
        h_ss = np.dot(As, p_es)

        # forward is double that or you can just add both off-diag entries
        # so reverse / transpose is still copy to each off-diag component not double
        h_e0ty = np.array([
            [h_sm[0], h_sm[2], h_ss[0]],
            [h_sm[2], h_sm[1], h_ss[1]],
            [h_ss[0], h_ss[1], 0.0],
        ])

        h_gty = XdinvT @ h_e0ty @ XdinvT.T
        h_ety = self._interp_tying_strains_transpose(pt, h_gty)
        add_mat_col, d_hat = self._compute_mitc_tying_strains_transpose(elem_xpts, fn, p_vars, p_d, h_ety)
        mat_col = add_mat_col
        mat_col += self._compute_director_sens(fn, d_hat)
        return mat_col
    
    def _compute_bending_disp_grad(self, pt, p_vars, p_d, Tmat, XdinvT, XdinvzT):    
        d0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=p_d)
        d0xi, d0eta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=p_d)
        u0xi, u0eta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
        u0x_init = np.column_stack((u0xi, u0eta, d0))
        u1x_init = np.column_stack((d0xi, d0eta, np.zeros(3)))

        p_u1x = Tmat.T @ (u1x_init @ XdinvT + u0x_init @ XdinvzT)
        p_u0x = Tmat.T @ u0x_init @ XdinvT
        return p_u0x, p_u1x

    def _compute_bending_strain(self, p_u0x, p_u1x):
        # linear strains only here
        p_ek = np.array([
            p_u1x[0,0],
            p_u1x[1,1],
            p_u1x[0,1] + p_u1x[1,0],
        ])
        return p_ek

    def _compute_bending_strain_sens(self, h_ek):
        h_u0x = np.zeros((3,3))
        h_u1x = np.zeros((3,3))
        h_u1x[0,0] += h_ek[0]
        h_u1x[1,1] += h_ek[1]
        h_u1x[0,1] += h_ek[2]
        h_u1x[1,0] += h_ek[2]
        return h_u0x, h_u1x

    def _compute_bending_disp_grad_sens(self, pt, Tmat, XdinvT, XdinvzT, h_u0x, h_u1x):
        
        u0d_barT = XdinvT @ h_u0x.T @ Tmat.T
        u0d_barT += XdinvzT @ h_u1x.T @ Tmat.T
        h_d = self._interp_fields_transpose(pt, vpn=3, num_fields=3, vec=u0d_barT[2,:])
        mat_col = self._interp_fields_grad_transpose(pt, vpn=6, num_fields=3, dxi=u0d_barT[0,:], deta=u0d_barT[1,:])

        u1d_barT = XdinvT @ h_u1x.T @ Tmat.T
        h_d += self._interp_fields_grad_transpose(pt, vpn=3, num_fields=3, dxi=u1d_barT[0,:], deta=u1d_barT[1,:])
        return mat_col, h_d

    def _add_jac_col_bending(
        self, pt:list, scale:float, 
        elem_xpts:np.ndarray, fn:np.ndarray,
        XdinvT:np.ndarray, Tmat:np.ndarray, 
        XdinvzT:np.ndarray, comp_data:list,
        p_vars:np.ndarray
    ):

        # p-forward section
        p_d = self._compute_director(p_vars, fn)
        p_u0x, p_u1x = self._compute_bending_disp_grad(pt, p_vars, p_d, Tmat, XdinvT, XdinvzT)
        p_ek = self._compute_bending_strain(p_u0x, p_u1x)

        # compute bending stress (constitutive)
        E, nu, thick = comp_data
        C = E / (1.0 - nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1.0 - nu) / 2.0]
        ])
        D = C * thick**3 / 12.0 * scale # A matrix from shell theory
        h_ek = np.dot(D, p_ek)

        # h-reverse section
        h_u0x, h_u1x = self._compute_bending_strain_sens(h_ek)
        add_mat_col, h_d = self._compute_bending_disp_grad_sens(pt, Tmat, XdinvT, XdinvzT, h_u0x, h_u1x)
        mat_col = add_mat_col
        mat_col += self._compute_director_sens(fn, h_d)
        return mat_col

    def get_felem(self, mag, elem_xpts: np.ndarray):
        pts, wts = first_order_quadrature()
        felem = np.zeros(self.ndof)

        fn = self._get_shell_normals(elem_xpts)  # once

        for iquad in range(4):
            ixi, ieta = iquad % 2, iquad // 2
            xi = pts[ixi]; eta = pts[ieta]
            wt = wts[ixi] * wts[ieta]
            pt = [xi, eta]

            detXd = self._get_detXd(pt, elem_xpts, fn)
            scale = abs(detXd) * wt   # consider abs

            N, _, _ = get_lagrange_basis_2d_all_standard(xi, eta)

            xyzq = self._interp_fields(pt, vpn=3, num_fields=3, vec=elem_xpts)
            q = float(mag(xyzq[0], xyzq[1], xyzq[2]))

            n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)

            # add q*n0 to translational DOFs (u,v,w) at each node
            felem[0::6] += n0[0] * (scale * q) * N
            felem[1::6] += n0[1] * (scale * q) * N
            felem[2::6] += n0[2] * (scale * q) * N

        return felem

    # # ----------------------------
    # # Prolongation / Restriction (same as your class, but locking constraints changed)
    # # ----------------------------
    # def _build_P1_scalar(self, nxe_coarse: int) -> sp.csr_matrix:
    #     if nxe_coarse in self._P1_cache:
    #         return self._P1_cache[nxe_coarse]

    #     nc = nxe_coarse + 1
    #     nf = 2 * nxe_coarse + 1  # = 2*nc - 1

    #     rows, cols, vals = [], [], []

    #     for i in range(nc):
    #         rows.append(2 * i)
    #         cols.append(i)
    #         vals.append(1.0)

    #     for i in range(nc - 1):
    #         r = 2 * i + 1
    #         rows += [r, r]
    #         cols += [i, i + 1]
    #         vals += [0.5, 0.5]

    #     P1 = sp.coo_matrix((vals, (rows, cols)), shape=(nf, nc)).tocsr()
    #     self._P1_cache[nxe_coarse] = P1
    #     return P1

    # def _build_P2_scalar(self, nxe_coarse: int) -> sp.csr_matrix:
    #     if nxe_coarse in self._P2_cache:
    #         return self._P2_cache[nxe_coarse]
    #     P1 = self._build_P1_scalar(nxe_coarse)
    #     P2 = sp.kron(P1, P1, format="csr")
    #     self._P2_cache[nxe_coarse] = P2
    #     return P2

    # def _build_P2_uncoupled3(self, nxe_coarse: int) -> sp.csr_matrix:
    #     if nxe_coarse in self._P2_u3_cache:
    #         return self._P2_u3_cache[nxe_coarse]
    #     P2s = self._build_P2_scalar(nxe_coarse)
    #     P = sp.kron(P2s, sp.eye(3, format="csr"), format="csr")
    #     self._P2_u3_cache[nxe_coarse] = P
    #     return P
    
    # def _apply_bcs_to_P(self, P: sp.csr_matrix, nxe_c: int) -> sp.csr_matrix:
    #     """
    #     Enforce Dirichlet BC structure directly on P (fine rows, coarse cols).
    #     For simply-supported: constrain w on boundary nodes.
    #     For clamped: constrain w, thx, thy on boundary nodes.
    #     """
    #     nxe_f = 2 * nxe_c
    #     nx_f = nxe_f + 1
    #     nx_c = nxe_c + 1

    #     nnodes_f = nx_f * nx_f
    #     nnodes_c = nx_c * nx_c

    #     # which dofs are constrained at a boundary node
    #     if self.clamped:
    #         dofs = (0, 1, 2)   # w, thx, thy
    #     else:
    #         dofs = (0,)        # w only

    #     fixed_rows_f = []
    #     for inode in range(nnodes_f):
    #         i = inode % nx_f
    #         j = inode // nx_f
    #         on_edge = (i == 0) or (i == nx_f - 1) or (j == 0) or (j == nx_f - 1)
    #         if on_edge:
    #             base = 3 * inode
    #             for a in dofs:
    #                 fixed_rows_f.append(base + a)

    #     fixed_cols_c = []
    #     for inode in range(nnodes_c):
    #         i = inode % nx_c
    #         j = inode // nx_c
    #         on_edge = (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1)
    #         if on_edge:
    #             base = 3 * inode
    #             for a in dofs:
    #                 fixed_cols_c.append(base + a)

    #     fixed_rows_f = np.array(sorted(set(fixed_rows_f)), dtype=int)
    #     fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=int)

    #     # IMPORTANT: for Dirichlet dofs, we want the prolongation to output exactly 0,
    #     # independent of coarse values. So:
    #     #  - zero those fine rows
    #     #  - zero those coarse columns (optional but recommended for consistency)
    #     P = P.tolil()
    #     P[fixed_rows_f, :] = 0.0
    #     P[:, fixed_cols_c] = 0.0
    #     P = P.tocsr()
    #     P.eliminate_zeros()
    #     return P

    # def apply_bcs_2d(self, u: np.ndarray, nxe: int):
    #     nx = nxe + 1
    #     U = u.reshape((nx * nx, 3))
    #     for j in range(nx):
    #         for i in range(nx):
    #             on_edge = (i == 0) or (i == nx - 1) or (j == 0) or (j == nx - 1)
    #             if not on_edge:
    #                 continue
    #             k = i + nx * j
    #             U[k, 0] = 0.0
    #             if self.clamped:
    #                 U[k, 1] = 0.0
    #                 U[k, 2] = 0.0

    # def _locking_aware_prolong_global_mitc_v1(self, nxe_c: int, length: float = 1.0):
    #     """
    #     Locking-aware prolongation where constraints are on MITC tying strains:
    #       [ gx(0,-b), gx(0,+b), gy(-a,0), gy(+a,0) ] = 0   per element
    #     => 4 constraints per element.
    #     """
    #     if nxe_c in self._lock_P_cache:
    #         return self._lock_P_cache[nxe_c]

    #     # sizes
    #     nxe_f = 2 * nxe_c

    #     nx_f = nxe_f + 1
    #     nnodes_f = nx_f**2
    #     nelems_f = nxe_f**2
    #     N_f = 3 * nnodes_f

    #     nx_c = nxe_c + 1
    #     nnodes_c = nx_c**2
    #     nelems_c = nxe_c**2
    #     N_c = 3 * nnodes_c

    #     # element reference coords (axis-aligned mapping as in your current code)
    #     dx_f = length / nxe_f
    #     x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
    #     y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

    #     dx_c = length / nxe_c
    #     x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
    #     y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

    #     # Build telling-strain operator G_f, G_c (dense):
    #     # rows per element: [gx(0,-b), gx(0,+b), gy(-a,0), gy(+a,0)]
    #     G_f = np.zeros((4 * nelems_f, N_f), dtype=float)
    #     for ielem_f in range(nelems_f):
    #         ex = ielem_f % nxe_f
    #         ey = ielem_f // nxe_f
    #         loc_nodes = np.array([
    #             ex + nx_f * ey,
    #             (ex + 1) + nx_f * ey,
    #             (ex + 1) + nx_f * (ey + 1),
    #             ex + nx_f * (ey + 1),
    #         ], dtype=int)
    #         loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)], dtype=int)

    #         # gx at (0, -b) and (0, +b)
    #         _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_f, y_f)
    #         _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_f, y_f)
    #         # gy at (-a, 0) and (+a, 0)
    #         _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_f, y_f)
    #         _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_f, y_f)

    #         r0 = 4 * ielem_f
    #         G_f[r0 + 0, loc_dof] += bx_m
    #         G_f[r0 + 1, loc_dof] += bx_p
    #         G_f[r0 + 2, loc_dof] += by_m
    #         G_f[r0 + 3, loc_dof] += by_p

    #     G_c = np.zeros((4 * nelems_c, N_c), dtype=float)
    #     for ielem_c in range(nelems_c):
    #         ex = ielem_c % nxe_c
    #         ey = ielem_c // nxe_c
    #         loc_nodes = np.array([
    #             ex + nx_c * ey,
    #             (ex + 1) + nx_c * ey,
    #             (ex + 1) + nx_c * (ey + 1),
    #             ex + nx_c * (ey + 1),
    #         ], dtype=int)
    #         loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)], dtype=int)

    #         _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_c, y_c)
    #         _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_c, y_c)
    #         _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_c, y_c)
    #         _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_c, y_c)

    #         r0 = 4 * ielem_c
    #         G_c[r0 + 0, loc_dof] += bx_m
    #         G_c[r0 + 1, loc_dof] += bx_p
    #         G_c[r0 + 2, loc_dof] += by_m
    #         G_c[r0 + 3, loc_dof] += by_p

    #     # # Elementwise injection for tying strains (4-per-elem)
    #     # P_gam = np.zeros((4 * nelems_f, 4 * nelems_c), dtype=float)
    #     # for ielem_f in range(nelems_f):
    #     #     ex = ielem_f % nxe_f
    #     #     ey = ielem_f // nxe_f
    #     #     ielem_c = (ex // 2) + (ey // 2) * nxe_c

    #     #     rf = 4 * ielem_f
    #     #     rc = 4 * ielem_c
    #     #     P_gam[rf + 0, rc + 0] = 1.0
    #     #     P_gam[rf + 1, rc + 1] = 1.0
    #     #     P_gam[rf + 2, rc + 2] = 1.0
    #     #     P_gam[rf + 3, rc + 3] = 1.0

    #     # ---------------------------------------------------------
    #     # P_gam: bilinear averaging (Q1) from coarse elem-grid to fine elem-grid
    #     # Each strain component is interpolated independently.
    #     #
    #     # coarse elements live on a (nxe_c x nxe_c) grid with indices (Ex_c, Ey_c)
    #     # fine elements live on a (nxe_f x nxe_f) grid with indices (ex, ey)
    #     #
    #     # Map fine element center to coarse-index space:
    #     #   x_c = (ex + 0.5)/2 - 0.5   in [ -0.25, nxe_c - 0.75 ]
    #     # so that fine elements in a 2x2 block around a coarse element "see" neighbors.
    #     #
    #     # You can tweak the "-0.5" shift if you want less cross-element blending.
    #     # ---------------------------------------------------------
    #     P_gam = np.zeros((4 * nelems_f, 4 * nelems_c), dtype=float)

    #     # NOTE : elemwise injection gives very similar thin shell perf

    #     def clamp(v, lo, hi):
    #         return max(lo, min(hi, v))

    #     for ielem_f in range(nelems_f):
    #         ex = ielem_f % nxe_f
    #         ey = ielem_f // nxe_f

    #         # fine element "center" mapped into coarse-element index space
    #         x = 0.5 * (ex + 0.5) - 0.5
    #         y = 0.5 * (ey + 0.5) - 0.5

    #         i0 = int(np.floor(x))
    #         j0 = int(np.floor(y))
    #         tx = x - i0
    #         ty = y - j0

    #         # clamp base so neighbors exist; edge blending degenerates gracefully
    #         i0 = clamp(i0, 0, nxe_c - 1)
    #         j0 = clamp(j0, 0, nxe_c - 1)

    #         i1 = clamp(i0 + 1, 0, nxe_c - 1)
    #         j1 = clamp(j0 + 1, 0, nxe_c - 1)

    #         # if clamped to boundary, kill the corresponding fraction
    #         if i1 == i0:
    #             tx = 0.0
    #         if j1 == j0:
    #             ty = 0.0

    #         w00 = (1.0 - tx) * (1.0 - ty)
    #         w10 = (tx)       * (1.0 - ty)
    #         w01 = (1.0 - tx) * (ty)
    #         w11 = (tx)       * (ty)

    #         # coarse element ids
    #         e00 = i0 + j0 * nxe_c
    #         e10 = i1 + j0 * nxe_c
    #         e01 = i0 + j1 * nxe_c
    #         e11 = i1 + j1 * nxe_c

    #         rf = 4 * ielem_f

    #         # for each tying-strain component, interpolate from coarse neighbors
    #         for comp in range(4):
    #             P_gam[rf + comp, 4 * e00 + comp] += w00
    #             P_gam[rf + comp, 4 * e10 + comp] += w10
    #             P_gam[rf + comp, 4 * e01 + comp] += w01
    #             P_gam[rf + comp, 4 * e11 + comp] += w11

    #     RHS = P_gam @ G_c  # (4*nelems_f, 3*nnodes_c)

    #     # Baseline nodal prolong
    #     P_0 = self._build_P2_uncoupled3(nxe_c) # csr
    #     P_0 = self._apply_bcs_to_P(P_0, nxe_c)
    #     lam = float(self.lam)

    #     # Coarse BC columns (same logic as your v2)
    #     constrained_dofs = (0, 1, 2) if self.clamped else (0,)

    #     fixed_cols_c = []
    #     for inode in range(nnodes_c):
    #         i = inode % nx_c
    #         j = inode // nx_c
    #         if (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1):
    #             base = 3 * inode
    #             for a in constrained_dofs:
    #                 fixed_cols_c.append(base + a)
    #     fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=int)
    #     all_cols_c = np.arange(3 * nnodes_c, dtype=int)
    #     free_cols_c = np.setdiff1d(all_cols_c, fixed_cols_c, assume_unique=False)

    #     # Fine BC rows with beam-style E-constraint
    #     fixed_rows_f = []
    #     for inode in range(nnodes_f):
    #         i = inode % nx_f
    #         j = inode // nx_f
    #         if (i == 0) or (i == nx_f - 1) or (j == 0) or (j == nx_f - 1):
    #             base = 3 * inode
    #             for a in constrained_dofs:
    #                 fixed_rows_f.append(base + a)
    #     fixed_rows_f = np.array(sorted(set(fixed_rows_f)), dtype=int)

    #     solve_rows_f = np.arange(3 * nnodes_f, dtype=int)

    #     nE = fixed_rows_f.size
    #     Esel = np.zeros((nE, solve_rows_f.size), dtype=float)
    #     # solve_rows_f is identity, so:
    #     Esel[np.arange(nE), fixed_rows_f] = 1.0

    #     # Least squares solve:
    #     #   minimize ||G_f P - RHS||^2 + ||E P||^2 + lam ||P - P0||^2
    #     A = G_f[:, solve_rows_f]                  # (4*nelems_f, nsolve)
    #     B = RHS[:, free_cols_c]                   # (4*nelems_f, nfreecols)

    #     A_aug = np.vstack([A, Esel])
    #     B_aug = np.vstack([B, np.zeros((nE, B.shape[1]))])
    #     # A_aug = A
    #     # B_aug = B

    #     idx0 = np.ix_(solve_rows_f, free_cols_c)
    #     P0_free = P_0[idx0].toarray()

    #     M = A_aug.T @ A_aug + lam * np.eye(solve_rows_f.size)
    #     rhs = A_aug.T @ B_aug + lam * P0_free
        
    #     # direct solve
    #     P_free = np.linalg.solve(M, rhs)

    #     # same some states to element class for DEBUGGING in locking sandbox 
    #     # need to get it out of this class
    #     self.G_f = G_f
    #     self.G_c = G_c
    #     self.P_gam = P_gam
    #     self.P_0 = P_0
    #     self.M = M
    #     self.RHS = rhs
    #     self.free_cols_c = free_cols_c
    #     self.fixed_cols_c = fixed_cols_c
    #     self.solve_rows_f = solve_rows_f
    #     self.P_0_free = P0_free
    #     self.Mb = None

    #     # # block-Jacobi smoothing (3x3 nodal) instead of direct solve
    #     # P_free = P0_free.copy()
    #     # omega = 0.8
    #     # # n_smooth = 5 (not enough)
    #     # # n_smooth = 15
    #     # # n_smooth = 30 # still takes 80 krylov iterations
    #     # n_smooth = 60

    #     # n = M.shape[0]
    #     # assert n % 3 == 0

    #     # # precompute inv(diag 3x3 blocks)
    #     # Dinv = np.empty((n//3, 3, 3), dtype=M.dtype)
    #     # for b in range(n // 3):
    #     #     i = 3 * b
    #     #     Dinv[b] = np.linalg.inv(M[i:i+3, i:i+3])

    #     # for _ in range(n_smooth):
    #     #     R = rhs - M @ P_free
    #     #     for b in range(n // 3):
    #     #         i = 3 * b
    #     #         P_free[i:i+3, :] += omega * (Dinv[b] @ R[i:i+3, :])


    #     # Assemble full P
    #     P = P_0.toarray()
    #     P[:, fixed_cols_c] = 0.0
    #     P[np.ix_(solve_rows_f, free_cols_c)] = P_free
    #     # P[fixed_rows_f, :] = 0.0

    #     self._lock_P_cache[nxe_c] = P.copy()
    #     return P
    
    # def _locking_aware_prolong_local_mitc_v5_jacobi(
    #         self,
    #         nxe_c: int,
    #         length: float = 1.0,
    #         n_sweeps: int = 10,
    #         omega: float = 0.5,
    #         with_fillin: bool = False,
    #         use_mask: bool = True,
    #     ):
    #     """
    #     v5 change (your requirement):
    #     - DO NOT eliminate coarse BC DOFs from the solve (no free_cols_c).
    #     - Keep RHS_csr and X with ncols = N_c = 3*nnodes_c (multiple of 3).
    #     - Impose coarse BCs by zeroing constrained coarse columns in:
    #             (P_gam @ G_c) term  AND  P0 rows (i.e., P0[:, fixed_cols_c]=0)
    #         so the solve is consistent while keeping full 3-dof node blocks.
    #     - Impose fine BCs via row/col elimination on M and rhs row reset to P0, as before.

    #     Solve (G_f^T G_f + lam I) P = G_f^T (P_gam G_c) + lam P0
    #     using SpMM Jacobi (3x3 block diagonal) with optional fixed sparsity (mask).
    #     """

    #     import numpy as np
    #     import scipy.sparse as sp

    #     cache_key = ("local_mitc_v5_jacobi_no_coarse_elim",
    #                 int(nxe_c), float(length), int(n_sweeps),
    #                 float(omega), bool(with_fillin), bool(use_mask))
    #     if cache_key in self._lock_P_cache:
    #         return self._lock_P_cache[cache_key]

    #     # -----------------------------
    #     # same as v1-v4 up to forming G_f, G_c, P_gam, RHS_full, P0
    #     # -----------------------------
    #     nxe_f = 2 * nxe_c

    #     nx_f = nxe_f + 1
    #     nnodes_f = nx_f**2
    #     nelems_f = nxe_f**2
    #     N_f = 3 * nnodes_f

    #     nx_c = nxe_c + 1
    #     nnodes_c = nx_c**2
    #     nelems_c = nxe_c**2
    #     N_c = 3 * nnodes_c

    #     dx_f = length / nxe_f
    #     x_f = dx_f * np.array([0.0, 1.0, 1.0, 0.0])
    #     y_f = dx_f * np.array([0.0, 0.0, 1.0, 1.0])

    #     dx_c = length / nxe_c
    #     x_c = dx_c * np.array([0.0, 1.0, 1.0, 0.0])
    #     y_c = dx_c * np.array([0.0, 0.0, 1.0, 1.0])

    #     G_f = np.zeros((4 * nelems_f, N_f), dtype=float)
    #     for ielem_f in range(nelems_f):
    #         ex = ielem_f % nxe_f
    #         ey = ielem_f // nxe_f
    #         loc_nodes = np.array([
    #             ex + nx_f * ey,
    #             (ex + 1) + nx_f * ey,
    #             (ex + 1) + nx_f * (ey + 1),
    #             ex + nx_f * (ey + 1),
    #         ], dtype=int)
    #         loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)], dtype=int)

    #         _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_f, y_f)
    #         _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_f, y_f)
    #         _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_f, y_f)
    #         _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_f, y_f)

    #         r0 = 4 * ielem_f
    #         G_f[r0 + 0, loc_dof] += bx_m
    #         G_f[r0 + 1, loc_dof] += bx_p
    #         G_f[r0 + 2, loc_dof] += by_m
    #         G_f[r0 + 3, loc_dof] += by_p

    #     G_c = np.zeros((4 * nelems_c, N_c), dtype=float)
    #     for ielem_c in range(nelems_c):
    #         ex = ielem_c % nxe_c
    #         ey = ielem_c // nxe_c
    #         loc_nodes = np.array([
    #             ex + nx_c * ey,
    #             (ex + 1) + nx_c * ey,
    #             (ex + 1) + nx_c * (ey + 1),
    #             ex + nx_c * (ey + 1),
    #         ], dtype=int)
    #         loc_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)], dtype=int)

    #         _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x_c, y_c)
    #         _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x_c, y_c)
    #         _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x_c, y_c)
    #         _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x_c, y_c)

    #         r0 = 4 * ielem_c
    #         G_c[r0 + 0, loc_dof] += bx_m
    #         G_c[r0 + 1, loc_dof] += bx_p
    #         G_c[r0 + 2, loc_dof] += by_m
    #         G_c[r0 + 3, loc_dof] += by_p

    #     # # P_gam (dense) as in v1
    #     # P_gam = np.zeros((4 * nelems_f, 4 * nelems_c), dtype=float)

    #     # def clamp(v, lo, hi):
    #     #     return max(lo, min(hi, v))

    #     # for ielem_f in range(nelems_f):
    #     #     ex = ielem_f % nxe_f
    #     #     ey = ielem_f // nxe_f

    #     #     x = 0.5 * (ex + 0.5) - 0.5
    #     #     y = 0.5 * (ey + 0.5) - 0.5

    #     #     i0 = int(np.floor(x))
    #     #     j0 = int(np.floor(y))
    #     #     tx = x - i0
    #     #     ty = y - j0

    #     #     i0 = clamp(i0, 0, nxe_c - 1)
    #     #     j0 = clamp(j0, 0, nxe_c - 1)
    #     #     i1 = clamp(i0 + 1, 0, nxe_c - 1)
    #     #     j1 = clamp(j0 + 1, 0, nxe_c - 1)

    #     #     if i1 == i0: tx = 0.0
    #     #     if j1 == j0: ty = 0.0

    #     #     w00 = (1.0 - tx) * (1.0 - ty)
    #     #     w10 = (tx)       * (1.0 - ty)
    #     #     w01 = (1.0 - tx) * (ty)
    #     #     w11 = (tx)       * (ty)

    #     #     e00 = i0 + j0 * nxe_c
    #     #     e10 = i1 + j0 * nxe_c
    #     #     e01 = i0 + j1 * nxe_c
    #     #     e11 = i1 + j1 * nxe_c

    #     #     rf = 4 * ielem_f
    #     #     for comp in range(4):
    #     #         P_gam[rf + comp, 4 * e00 + comp] += w00
    #     #         P_gam[rf + comp, 4 * e10 + comp] += w10
    #     #         P_gam[rf + comp, 4 * e01 + comp] += w01
    #     #         P_gam[rf + comp, 4 * e11 + comp] += w11

    #     # Elementwise injection for tying strains (4-per-elem)
    #     P_gam = np.zeros((4 * nelems_f, 4 * nelems_c), dtype=float)
    #     for ielem_f in range(nelems_f):
    #         ex = ielem_f % nxe_f
    #         ey = ielem_f // nxe_f
    #         ielem_c = (ex // 2) + (ey // 2) * nxe_c

    #         rf = 4 * ielem_f
    #         rc = 4 * ielem_c
    #         P_gam[rf + 0, rc + 0] = 1.0
    #         P_gam[rf + 1, rc + 1] = 1.0
    #         P_gam[rf + 2, rc + 2] = 1.0
    #         P_gam[rf + 3, rc + 3] = 1.0

    #     RHS_full = P_gam @ G_c  # dense (4*nelems_f, 3*nnodes_c)

    #     # baseline P0 (csr) already has BCs applied correctly
    #     P_0 = self._build_P2_uncoupled3(nxe_c)
    #     P_0 = self._apply_bcs_to_P(P_0, nxe_c)

    #     lam = float(self.lam)
    #     constrained_dofs = (0, 1, 2) if self.clamped else (0,)

    #     # coarse constrained cols (still computed, but NOT eliminated from solve)
    #     fixed_cols_c = []
    #     for inode in range(nnodes_c):
    #         i = inode % nx_c
    #         j = inode // nx_c
    #         if (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1):
    #             base = 3 * inode
    #             for a in constrained_dofs:
    #                 fixed_cols_c.append(base + a)
    #     fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=int)

    #     # fine constrained rows
    #     fixed_rows_f = []
    #     for inode in range(nnodes_f):
    #         i = inode % nx_f
    #         j = inode // nx_f
    #         if (i == 0) or (i == nx_f - 1) or (j == 0) or (j == nx_f - 1):
    #             base = 3 * inode
    #             for a in constrained_dofs:
    #                 fixed_rows_f.append(base + a)
    #     fixed_rows_f = np.array(sorted(set(fixed_rows_f)), dtype=int)

    #     solve_rows_f = np.arange(N_f, dtype=int)  # keep all rows

    #     # -----------------------------
    #     # v5: impose coarse BCs by zeroing constrained coarse columns
    #     # in BOTH RHS term and P0 (so system has N_c cols but BC cols are forced to 0)
    #     # -----------------------------
    #     if fixed_cols_c.size > 0:
    #         RHS_full[:, fixed_cols_c] = 0.0
    #         # also guarantee P0 has those cols zero (in case _apply_bcs_to_P doesn't)
    #         P_0 = P_0.tolil(copy=True)
    #         P_0[:, fixed_cols_c] = 0.0
    #         P_0 = P_0.tocsr()

    #     # Build normal equations without augmentation (FULL columns)
    #     A = G_f[:, solve_rows_f]           # (4*nelems_f, N_f)
    #     B = RHS_full                       # (4*nelems_f, N_c)
    #     P0_all = P_0[solve_rows_f, :].toarray()  # (N_f, N_c), cols multiple of 3

    #     M   = A.T @ A + lam * np.eye(solve_rows_f.size)
    #     rhs = A.T @ B + lam * P0_all

    #     # -----------------------------
    #     # enforce fine BCs by row/col elimination on M, set rhs row to P0 row (ALL cols)
    #     # -----------------------------
    #     if fixed_rows_f.size > 0:
    #         M[fixed_rows_f, :] = 0.0
    #         M[:, fixed_rows_f] = 0.0
    #         for d in fixed_rows_f:
    #             M[d, d] = 1.0
    #         rhs[fixed_rows_f, :] = P0_all[fixed_rows_f, :]

    #     # -----------------------------
    #     # SpMM Jacobi solve (same structure, but X/RHS have N_c cols)
    #     # -----------------------------
    #     LHS = sp.csr_matrix(M)
    #     RHS_csr = sp.csr_matrix(rhs)
    #     X = sp.csr_matrix(P0_all)  # initial guess

    #     # mask_mode = 1
    #     mask_mode = 2

    #     if mask_mode == 1:
    #         mask = None
    #         if (not with_fillin) and use_mask:
    #             S_lock = (LHS @ RHS_csr)
    #             tol = 1e-24
    #             mask = (abs(S_lock) > tol).astype(np.int8)

    #     elif mask_mode == 2:
    #         def _structural_mask(A: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
    #             """Structural pattern of A@B (no numeric cancellation)."""
    #             A1 = A.copy()
    #             if A1.nnz:
    #                 A1.data[:] = 1.0
    #             B1 = B.copy()
    #             if B1.nnz:
    #                 B1.data[:] = 1.0
    #             S = A1 @ B1
    #             S.eliminate_zeros()
    #             return (S != 0).astype(np.int8)

    #         mask = None
    #         if (not with_fillin) and use_mask:
    #             P0_pat = (sp.csr_matrix(P0_all) != 0).astype(np.int8)
    #             MR_pat = _structural_mask(LHS, RHS_csr)
    #             mask = (P0_pat + MR_pat)
    #             mask.data[:] = 1
    #             mask.eliminate_zeros()

    #     def sparse_control(Z):
    #         if mask is None:
    #             return Z
    #         if not sp.issparse(Z):
    #             Z = sp.csr_matrix(Z)
    #         return Z.multiply(mask)

    #     block_size = 3
    #     n_unknown = X.shape[0]
    #     assert n_unknown % block_size == 0
    #     n_blk = n_unknown // block_size

    #     Mb = LHS.tobsr(blocksize=(block_size, block_size))

    #     diag_blocks = np.zeros((n_blk, block_size, block_size), dtype=float)
    #     for i in range(n_blk):
    #         start, end = Mb.indptr[i], Mb.indptr[i + 1]
    #         cols = Mb.indices[start:end]
    #         data = Mb.data[start:end]
    #         matches = np.where(cols == i)[0]
    #         if matches.size == 0:
    #             raise RuntimeError("Missing diagonal block in M (unexpected).")
    #         Db = data[matches[0]]
    #         Db = 0.5 * (Db + Db.T)
    #         diag_blocks[i, :, :] = np.linalg.inv(Db)

    #     Dinv_indptr  = np.arange(n_blk + 1, dtype=np.int32)
    #     Dinv_indices = np.arange(n_blk,     dtype=np.int32)
    #     Dinv_data    = diag_blocks

    #     Dinv_op = sp.bsr_matrix(
    #         (Dinv_data, Dinv_indices, Dinv_indptr),
    #         shape=(n_unknown, n_unknown),
    #         blocksize=(block_size, block_size),
    #     ).tocsr()

    #     X = sparse_control(X)

    #     if with_fillin or (mask is None):
    #         for _ in range(int(n_sweeps)):
    #             MX  = LHS @ X
    #             RES = RHS_csr - MX
    #             X   = X + float(omega) * (Dinv_op @ RES)
    #     else:
    #         for _ in range(int(n_sweeps)):
    #             # LHS /= 64.0; RHS_csr /= 32.0; Dinv_op *= 64.0 # DEBUG
    #             MX  = sparse_control(LHS @ X) 
    #             RES = sparse_control(RHS_csr - MX)
    #             X   = sparse_control(X + float(omega) * (Dinv_op @ RES))

    #     P_sol = X.toarray()  # (N_f, N_c)

    #     # -----------------------------
    #     # Assemble full P (dense)
    #     # -----------------------------
    #     P = P_0.toarray()                 # already has coarse BC columns zeroed (we enforced above)
    #     if fixed_cols_c.size > 0:
    #         P[:, fixed_cols_c] = 0.0      # safety
    #     P[solve_rows_f, :] = P_sol        # overwrite all rows/cols (including coarse BC cols which remain 0)

    #     # stash debug
    #     self.G_f = G_f
    #     self.G_c = G_c
    #     self.P_gam = P_gam
    #     self.P_0 = P_0
    #     self.M = M
    #     self.RHS = rhs
    #     self.fixed_cols_c = fixed_cols_c
    #     self.solve_rows_f = solve_rows_f
    #     self.fixed_rows_f = fixed_rows_f
    #     self.P = P

    #     self._lock_P_cache[cache_key] = P.copy()
    #     return P


    # def _energy_smooth_jacobi_v1(
    #     self,
    #     nxe_c: int,
    #     length: float = 1.0,
    #     n_sweeps: int = 10,
    #     omega: float = 0.7,
    #     with_fillin: bool = False,
    #     use_mask: bool = True,
    # ):
    #     """
    #     Standard K-matrix energy smoothing in Jacobi-preconditioned space:
    #         P <- P - omega * D^{-1} (K P)

    #     - K is taken from: self._kmat_cache[nxe_c]   (you provide it)
    #     - Optional fixed sparsity: mask = pattern(K@P) computed once initially.
    #     - Cache key is ONLY nxe_c (simple).
    #     """

    #     import numpy as np
    #     import scipy.sparse as sp

    #     # simple cache: ONLY keyed by nxe_c
    #     cache_key = ("energy_smooth_jacobi_v1", int(nxe_c))
    #     if cache_key in self._lock_P_cache:
    #         return self._lock_P_cache[cache_key]

    #     # baseline P0
    #     P0 = self._build_P2_uncoupled3(nxe_c)
    #     P0 = self._apply_bcs_to_P(P0, nxe_c)
    #     P = P0.tocsr()

    #     # kmat from cache
    #     if not hasattr(self, "_kmat_cache") or (int(nxe_c) not in self._kmat_cache):
    #         raise RuntimeError("Expected self._kmat_cache[nxe_c] to exist for energy smoothing.")
    #     K = self._kmat_cache[int(nxe_c)]
    #     K = K.tocsr() if sp.isspmatrix(K) else sp.csr_matrix(K)

    #     # Jacobi block inverse (3x3 nodal blocks)
    #     bs = 3
    #     N = K.shape[0]
    #     if (N % bs) != 0 or K.shape[1] != N:
    #         raise ValueError(f"kmat must be square with size multiple of 3, got {K.shape}")
    #     nblk = N // bs

    #     Kb = K.tobsr(blocksize=(bs, bs))
    #     diag_inv = np.zeros((nblk, bs, bs), dtype=float)
    #     for i in range(nblk):
    #         s, e = Kb.indptr[i], Kb.indptr[i + 1]
    #         cols = Kb.indices[s:e]
    #         data = Kb.data[s:e]  # (nblocks_in_row, bs, bs)
    #         k = np.searchsorted(cols, i)
    #         if k >= cols.size or cols[k] != i:
    #             raise RuntimeError("Missing diagonal 3x3 block in kmat.")
    #         Db = data[k]
    #         Db = 0.5 * (Db + Db.T)
    #         diag_inv[i] = np.linalg.inv(Db)

    #     Dinv = sp.bsr_matrix(
    #         (diag_inv, np.arange(nblk, dtype=np.int32), np.arange(nblk + 1, dtype=np.int32)),
    #         shape=(N, N),
    #         blocksize=(bs, bs),
    #     ).tocsr()

    #     # fixed sparsity mask from initial K@P
    #     mask = None
    #     if (not with_fillin) and use_mask:
    #         mask = ((K @ P) != 0).astype(np.int8)

    #     def control(Z):
    #         if mask is None:
    #             return Z
    #         if not sp.issparse(Z):
    #             Z = sp.csr_matrix(Z)
    #         return Z.multiply(mask)

    #     P = control(P)

    #     # Jacobi-preconditioned energy smoothing: P <- P - omega * Dinv * (K P)
    #     if with_fillin or (mask is None):
    #         for _ in range(int(n_sweeps)):
    #             KP = K @ P
    #             P = P - float(omega) * (Dinv @ KP)
    #     else:
    #         for _ in range(int(n_sweeps)):
    #             KP = control(K @ P)
    #             P = control(P - float(omega) * (Dinv @ KP))

    #     P_out = P.toarray()
    #     self._lock_P_cache[cache_key] = P_out
    #     return P_out


    # def prolongate(self, coarse_u: np.ndarray, nxe_coarse: int):
    #     dpn = self.dof_per_node
    #     nxc = nxe_coarse + 1
    #     Nc = nxc * nxc
    #     assert coarse_u.size == dpn * Nc

    #     nxe_f = 2 * nxe_coarse
    #     nxf = nxe_f + 1
    #     Nf = nxf * nxf

    #     if self.prolong_mode == "locking-global":
    #         method = self._locking_aware_prolong_global_mitc_v1
    #         P = method(nxe_coarse, length=1.0)
    #     elif self.prolong_mode == 'locking-local':
    #         P = self._locking_aware_prolong_local_mitc_v5_jacobi(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
    #     elif self.prolong_mode == "standard":
    #         P = self._build_P2_uncoupled3(nxe_coarse)
    #         P = self._apply_bcs_to_P(P, nxe_coarse)
    #     elif self.prolong_mode == "energy-jacobi":
    #         P = self._energy_smooth_jacobi_v1(nxe_coarse, n_sweeps=self.n_lock_sweeps, omega=self.omega)
    #     else:
    #         raise NotImplementedError("locking-local not implemented in this prototype")

    #     fine_u = P @ coarse_u

    #     return fine_u

    # def restrict_defect(self, fine_r: np.ndarray, nxe_fine: int):
    #     dpn = self.dof_per_node
    #     nxf = nxe_fine + 1
    #     Nf = nxf * nxf
    #     assert fine_r.size == dpn * Nf
    #     assert (nxe_fine % 2) == 0

    #     nxe_coarse = nxe_fine // 2
    #     nxc = nxe_coarse + 1
    #     Nc = nxc * nxc

    #     if self.prolong_mode == "locking-global":
    #         method = self._locking_aware_prolong_global_mitc_v1
    #         P = method(nxe_coarse, length=1.0)
    #     elif self.prolong_mode == 'locking-local':
    #         P = self._locking_aware_prolong_local_mitc_v5_jacobi(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
    #     elif self.prolong_mode == "standard":
    #         P = self._build_P2_uncoupled3(nxe_coarse)
    #         P = self._apply_bcs_to_P(P, nxe_coarse)
    #     elif self.prolong_mode == "energy-jacobi":
    #         P = self._energy_smooth_jacobi_v1(nxe_coarse, n_sweeps=self.n_lock_sweeps, omega=self.omega)
    #     else:
    #         raise NotImplementedError("locking-local not implemented in this prototype")

    #     R = P.T

    #     fine_r = fine_r.copy()
    #     self.apply_bcs_2d(fine_r, nxe_fine)

    #     coarse_r = R @ fine_r

    #     return coarse_r
