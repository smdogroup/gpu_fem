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

    K_SHEAR = 5.0 / 6.0
    DRILL_REG = 10.0
    # DRILL_REG = 0.0 # temp debug

    def __init__(
        self,
        prolong_mode: str = "locking-global",  # 'standard'
        lam: float = 1e-5,
        n_lock_sweeps:int=10,
        omega:float=0.5,
        debug:bool=False,
        strain_debug_mode:str='all', # 'all' is not debug, 'tying', 'bending', 'drill' just does one term
    ):
        assert prolong_mode in ["locking-global", "locking-local", "standard", "energy-jacobi"]

        self.dof_per_node = 6
        self.nodes_per_elem = 4
        self.ndof = self.dof_per_node * self.nodes_per_elem
        self.debug = debug
        self.strain_debug_mode = strain_debug_mode

        self.prolong_mode = prolong_mode
        self.clamped = False
        self.lam = float(lam)
        self.n_lock_sweeps = n_lock_sweeps
        self.omega = float(omega)

        # cache for prolong/restrict operators
        self._P1_cache = {}   # key: nxe_coarse -> P1 (csr)
        self._P2_cache = {}   # key: nxe_coarse -> P2 (csr)
        self._P2_u3_cache = {}
        # self._lock_P_cache = {}
        self._kmat_cache = {}
        self._xpts_cache = {}
        self._P_cache = {}

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

                if self.strain_debug_mode in ['all', 'drill']:
                    # old mode (not fast version)
                    # local_mat_col += self._add_jac_col_drill_nodal(
                    #     pt, scale, elem_xpts, fn,
                    #     XdinvT, Tmat, XdinvzT, comp_data,
                    #     p_vars
                    # )

                    # checked it's fine to use either version of drill strain and it still gives mesh convergence!
                    # just wanted to be sure
                    local_mat_col += self._add_jac_col_drill(
                        pt, scale, elem_xpts, fn,
                        XdinvT, Tmat, XdinvzT, comp_data,
                        p_vars
                    )

                if self.strain_debug_mode in ['all', 'bending']:
                    local_mat_col += self._add_jac_col_bending(
                        pt, scale, elem_xpts, fn,
                        XdinvT, Tmat, XdinvzT, comp_data,
                        p_vars
                    )

                if self.strain_debug_mode in ['all', 'tying']:
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

        # print(f"{elem_xpts.reshape((4,3))=}")

        for i in range(4):
            # node pt
            ixi, ieta = i % 2, i // 2
            xi = -1 + 2.0 * ixi
            eta = -1 + 2.0 * ieta
            node_pt = [xi, eta]
            # print(f"{elem_xpts=}")

            dX_dxi, dX_deta = self._interp_fields_grad(node_pt, vpn=3, num_fields=3, vec=elem_xpts)
            # print(f"{dX_dxi=} {dX_deta=}")
            normal = np.cross(dX_dxi, dX_deta)
            normal /= np.linalg.norm(normal)
            fn[3*i:(3*i+3)] += normal

            # xpt = self._interp_fields(node_pt, vpn=3, num_fields=3, vec=elem_xpts)
            # print(f"{dX_dxi=} {dX_deta=} {normal=}")

        # fn_rs = fn.reshape((3,4))
        # print(f"{fn_rs=}")
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

        # print(f"{Xd=}")

        Xdinv = np.linalg.inv(Xd)
        XdinvT = Xdinv @ Tmat
        XdinvzT = -Xdinv @ Xdz @ XdinvT

        # print(f"{XdinvT=} {Tmat=} {XdinvzT=}")
        return XdinvT, Tmat, XdinvzT
    
    def _add_jac_col_drill_nodal(
        self, quad_pt:list, scale:float, 
        elem_xpts:np.ndarray, fn:np.ndarray,
        _XdinvT:np.ndarray, _Tmat:np.ndarray, 
        _XdinvzT:np.ndarray, comp_data:list,
        p_vars:np.ndarray   
    ):
        mat_col = np.zeros(24)

        etn = np.zeros(4)
        for inode in range(4):
            # node pt
            ixi, ieta = inode % 2, inode // 2
            xi = -1 + 2.0 * ixi
            eta = -1 + 2.0 * ieta
            node_pt = [xi, eta]
            # print(f"{elem_xpts=}")

            dX_dxi, dX_deta = self._interp_fields_grad(node_pt, vpn=3, num_fields=3, vec=elem_xpts)        
            n0 = fn[3*inode:(3*inode+3)]
            Xd = np.column_stack((dX_dxi, dX_deta, n0))
            Tmat = self._shell_compute_transform(n0, ref_axis=np.array([1, 0, 0]))
            u0xi, u0eta = self._interp_fields_grad(node_pt, vpn=6, num_fields=3, vec=p_vars)
            u0xn = np.column_stack((u0xi, u0eta, np.zeros(3)))
            u0 = p_vars[6*inode:(6*inode+6)]
            thx, thy, thz = u0[3], u0[4], u0[5]
            C0 = np.array([ # rotation matrix for drill strain
                [1.0, thz, -thy],
                [-thz, 1.0, thx],
                [thy, -thx, 1.0]
            ])
            C = Tmat.T @ C0 @ Tmat
            XdinvT = np.linalg.inv(Xd) @ Tmat
            u0x = Tmat.T @ u0xn @ XdinvT
            # eval drill strain
            etn[inode] = 0.5 * (C[1,0] + u0x[1,0] - C[0,1] - u0x[0,1])
    
        et = self._interp_fields(quad_pt, vpn=1, num_fields=1, vec=etn)
        
        # now compute drill stress
        # st = self._compute_drill_stress(scale, comp_data, et)
        E, nu, thick = comp_data
        G = E / 2.0 / (1.0 + nu)
        As = self.K_SHEAR * G * thick
        A_drill = As * self.DRILL_REG
        st = scale * A_drill * et[0] # equiv to et_bar

        # reverse
        st_vec = np.zeros(1); st_vec[0] = st
        stn = self._interp_fields_transpose(quad_pt, vpn=1, num_fields=1, vec=st_vec)

        for inode in range(4):
            # node pt
            ixi, ieta = inode % 2, inode // 2
            xi = -1 + 2.0 * ixi
            eta = -1 + 2.0 * ieta
            node_pt = [xi, eta]
            # print(f"{elem_xpts=}")

            dX_dxi, dX_deta = self._interp_fields_grad(node_pt, vpn=3, num_fields=3, vec=elem_xpts)        
            n0 = fn[3*inode:(3*inode+3)]
            Xd = np.column_stack((dX_dxi, dX_deta, n0))
            Tmat = self._shell_compute_transform(n0, ref_axis=np.array([1, 0, 0]))
            XdinvT = np.linalg.inv(Xd) @ Tmat            
            
            # now do the transpose version of the drill strains
            # mat_col = self._compute_drill_strain_transpose(pt, Tmat, XdinvT, st)
            u0x_hat = np.zeros_like(u0x); C_hat = np.zeros_like(C)
            u0x_hat[1,0] += 0.5 * stn[inode]
            C_hat[1,0] += 0.5 * stn[inode]
            u0x_hat[0,1] -= 0.5 * stn[inode]
            C_hat[0,1] -= 0.5 * stn[inode]

            C0_hat = Tmat @ C_hat @ Tmat.T
            u0xn_hat = Tmat @ u0x_hat @ XdinvT.T
            u0xi_hat = u0xn_hat[:,0]
            u0eta_hat = u0xn_hat[:,1]

            thx_hat = C0_hat[1,2] - C0_hat[2,1]
            thy_hat = C0_hat[2,0] - C0_hat[0,2]
            thz_hat = C0_hat[0,1] - C0_hat[1,0]

            mat_col[6*inode+3] += thx_hat
            mat_col[6*inode+4] += thy_hat
            mat_col[6*inode+5] += thz_hat
            
            mat_col += self._interp_fields_grad_transpose(
                node_pt, vpn=6, num_fields=3, dxi=u0xi_hat, deta=u0eta_hat
            )

        return mat_col
        
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
        As = self.K_SHEAR * G * thick
        A_drill = As * self.DRILL_REG
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
    
    def _get_tying_strain_matrix(
        self, quad_pt:list,
        elem_xpts:np.ndarray
    ):
        # for multigrid operators
        tying_matrix = np.zeros((5, 24))

        fn = self._get_shell_normals(elem_xpts)
        XdinvT, _, _ = self._get_shell_rotations(
            quad_pt, elem_xpts, fn
        )
        
        for i in range(24):
            p_vars = np.zeros(24)
            p_vars[i] = 1.0

            # just do pforward and hrev part cause linear
            p_d = self._compute_director(p_vars, fn)
            p_ety = self._compute_mitc_tying_strains(elem_xpts, fn, p_vars, p_d)
            p_gty = self._interp_tying_strains(quad_pt, p_ety)
            # sym mat rotate frame
            p_e0ty = XdinvT.T @ p_gty @ XdinvT
            # membrane strains
            p_em = np.array([p_e0ty[0,0], p_e0ty[1,1], 2.0 * p_e0ty[0,1]]) 
            # trv shear strains 
            p_es = 2.0 * np.array([p_e0ty[0,2], p_e0ty[1,2]]) # or just sym sums here which means reverse is copy to each component

            p_tying = np.concatenate([p_em, p_es], axis=0)
            tying_matrix[:,i] = p_tying

        return tying_matrix # return the five tying strains at this quadpt

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

    # ----------------------------
    # Prolongation / Restriction (same as your class, but locking constraints changed)
    # ----------------------------
    def _build_P1_scalar(self, nxe_coarse: int) -> sp.csr_matrix:
        if nxe_coarse in self._P1_cache:
            return self._P1_cache[nxe_coarse]

        nc = nxe_coarse + 1
        nf = 2 * nxe_coarse + 1  # = 2*nc - 1

        rows, cols, vals = [], [], []

        for i in range(nc):
            rows.append(2 * i)
            cols.append(i)
            vals.append(1.0)

        for i in range(nc - 1):
            r = 2 * i + 1
            rows += [r, r]
            cols += [i, i + 1]
            vals += [0.5, 0.5]

        P1 = sp.coo_matrix((vals, (rows, cols)), shape=(nf, nc)).tocsr()
        self._P1_cache[nxe_coarse] = P1
        return P1

    def _build_P2_scalar(self, nxe_coarse: int) -> sp.csr_matrix:
        if nxe_coarse in self._P2_cache:
            return self._P2_cache[nxe_coarse]
        P1 = self._build_P1_scalar(nxe_coarse)
        P2 = sp.kron(P1, P1, format="csr")
        self._P2_cache[nxe_coarse] = P2
        return P2

    def _build_P2_uncoupled3(self, nxe_coarse: int) -> sp.csr_matrix:
        if nxe_coarse in self._P2_u3_cache:
            return self._P2_u3_cache[nxe_coarse]
        P2s = self._build_P2_scalar(nxe_coarse)
        P = sp.kron(P2s, sp.eye(self.dof_per_node, format="csr"), format="csr")
        self._P2_u3_cache[nxe_coarse] = P
        return P
    
    def _apply_bcs_to_P(self, P: sp.csr_matrix, nxe_c: int) -> sp.csr_matrix:
        """
        Enforce Dirichlet BC structure directly on P (fine rows, coarse cols).
        For simply-supported: constrain w on boundary nodes.
        For clamped: constrain w, thx, thy on boundary nodes.
        """
        nxe_f = 2 * nxe_c
        nx_f = nxe_f + 1
        nx_c = nxe_c + 1

        nnodes_f = nx_f * nx_f
        nnodes_c = nx_c * nx_c

        # need clamped rn, haven't setup non-clamped BCs in multigrid yet (cause diff DOF constrained per node)
        assert self.clamped

        # which dofs are constrained at a boundary node
        if self.clamped:
            dofs = (0, 1, 2, 3, 4, 5)   # u, v, w, thx, thy, thz
        else:
            dofs = (0,)        # w only

        fixed_rows_f = []
        for inode in range(nnodes_f):
            i = inode % nx_f
            j = inode // nx_f
            on_edge = (i == 0) or (i == nx_f - 1) or (j == 0) or (j == nx_f - 1)
            if on_edge:
                base = 6 * inode
                for a in dofs:
                    fixed_rows_f.append(base + a)

        fixed_cols_c = []
        for inode in range(nnodes_c):
            i = inode % nx_c
            j = inode // nx_c
            on_edge = (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1)
            if on_edge:
                base = 6 * inode
                for a in dofs:
                    fixed_cols_c.append(base + a)

        fixed_rows_f = np.array(sorted(set(fixed_rows_f)), dtype=int)
        fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=int)

        # IMPORTANT: for Dirichlet dofs, we want the prolongation to output exactly 0,
        # independent of coarse values. So:
        #  - zero those fine rows
        #  - zero those coarse columns (optional but recommended for consistency)
        P = P.tolil()
        P[fixed_rows_f, :] = 0.0
        P[:, fixed_cols_c] = 0.0
        P = P.tocsr()
        P.eliminate_zeros()
        return P

    def apply_bcs_2d(self, u: np.ndarray, nxe: int):
        nx = nxe + 1
        U = u.reshape((nx * nx, 6))
        assert self.clamped # haven't written it for not clamped yet
        for j in range(nx):
            for i in range(nx):
                on_edge = (i == 0) or (i == nx - 1) or (j == 0) or (j == nx - 1)
                if not on_edge:
                    continue
                k = i + nx * j
                if self.clamped:
                    for i in range(6):
                        U[k,i] = 0.0

    def _get_edge_prolong_strain_matrix(self, nxe_c:int):

        # sizes
        nxe_f = 2 * nxe_c
        nelems_f = nxe_f**2
        nelems_c = nxe_c**2

        # ---------------------------------------------------------
        # P_gam: geometry-aware interpolation of tying-strain "samples"
        # from coarse element tying locations -> fine element tying locations.
        #
        # gamma ordering per element (9):
        # 0 g11n @ (xi=0,eta=-1)  bottom-mid
        # 1 g13n @ (xi=0,eta=-1)  bottom-mid
        # 2 g11p @ (xi=0,eta=+1)  top-mid
        # 3 g13p @ (xi=0,eta=+1)  top-mid
        # 4 g22n @ (xi=-1,eta=0)  left-mid
        # 5 g23n @ (xi=-1,eta=0)  left-mid
        # 6 g22p @ (xi=+1,eta=0)  right-mid
        # 7 g23p @ (xi=+1,eta=0)  right-mid
        # 8 g12  @ (xi=0,eta=0)   center
        # ---------------------------------------------------------
        P_gam = np.zeros((9 * nelems_f, 9 * nelems_c), dtype=float)

        hc = 1.0 / float(nxe_c)
        hf = 1.0 / float(nxe_f)

        def clamp_int(v, lo, hi):
            return lo if v < lo else (hi if v > hi else v)

        def elem_id(ex, ey, nxe):
            return ex + ey * nxe

        def bilinear_elems(x, y):
            """
            Return up to 4 surrounding coarse elements and bilinear weights
            for a point (x,y) in [0,1]^2, interpreted in coarse-element cell coords.
            """
            # clamp to domain (avoid x==1 -> i==nxe_c)
            x = min(max(x, 0.0), 1.0 - 1e-15)
            y = min(max(y, 0.0), 1.0 - 1e-15)

            i0 = int(np.floor(x / hc))
            j0 = int(np.floor(y / hc))
            i0 = clamp_int(i0, 0, nxe_c - 1)
            j0 = clamp_int(j0, 0, nxe_c - 1)

            i1 = min(i0 + 1, nxe_c - 1)
            j1 = min(j0 + 1, nxe_c - 1)

            tx = (x - i0 * hc) / hc
            ty = (y - j0 * hc) / hc

            if i1 == i0:
                tx = 0.0
            if j1 == j0:
                ty = 0.0

            w00 = (1.0 - tx) * (1.0 - ty)
            w10 = (tx)       * (1.0 - ty)
            w01 = (1.0 - tx) * (ty)
            w11 = (tx)       * (ty)

            return (i0, j0, w00), (i1, j0, w10), (i0, j1, w01), (i1, j1, w11)

        def yblend_in_elem(y, ey_c):
            """
            For a coarse element (ex,ey_c), compute local y-fraction s in [0,1]
            so we can blend bottom/top samples (g??n vs g??p).
            """
            y0 = ey_c * hc
            s = (y - y0) / hc
            return min(max(s, 0.0), 1.0)

        def xblend_in_elem(x, ex_c):
            """
            For a coarse element (ex_c,ey), compute local x-fraction s in [0,1]
            so we can blend left/right samples (g??n vs g??p).
            """
            x0 = ex_c * hc
            s = (x - x0) / hc
            return min(max(s, 0.0), 1.0)

        for ielem_f in range(nelems_f):
            exf = ielem_f % nxe_f
            eyf = ielem_f // nxe_f

            rf = 9 * ielem_f

            # --- fine tying sample physical locations ---
            # bottom mid (g11n,g13n)
            xb = (exf + 0.5) * hf
            yb = (eyf + 0.0) * hf

            # top mid (g11p,g13p)
            xt = (exf + 0.5) * hf
            yt = (eyf + 1.0) * hf

            # left mid (g22n,g23n)
            xl = (exf + 0.0) * hf
            yl = (eyf + 0.5) * hf

            # right mid (g22p,g23p)
            xr = (exf + 1.0) * hf
            yr = (eyf + 0.5) * hf

            # center (g12)
            xc = (exf + 0.5) * hf
            yc = (eyf + 0.5) * hf

            # -------------------------
            # g11/g13 rows: blend (n,p) in y inside each coarse elem,
            # and Q1 across neighboring coarse elems.
            # -------------------------
            # fine row 0: g11n at (xb,yb)
            for (ix, iy, w) in bilinear_elems(xb, yb):
                if w == 0.0:
                    continue
                e = elem_id(ix, iy, nxe_c)
                rc = 9 * e
                s = yblend_in_elem(yb, iy)  # blend bottom->top in that coarse elem
                P_gam[rf + 0, rc + 0] += w * (1.0 - s)  # coarse g11n
                P_gam[rf + 0, rc + 2] += w * (s)        # coarse g11p

                P_gam[rf + 1, rc + 1] += w * (1.0 - s)  # coarse g13n
                P_gam[rf + 1, rc + 3] += w * (s)        # coarse g13p

            # fine row 2/3: g11p/g13p at (xt,yt)
            for (ix, iy, w) in bilinear_elems(xt, yt):
                if w == 0.0:
                    continue
                e = elem_id(ix, iy, nxe_c)
                rc = 9 * e
                s = yblend_in_elem(yt, iy)
                P_gam[rf + 2, rc + 0] += w * (1.0 - s)
                P_gam[rf + 2, rc + 2] += w * (s)

                P_gam[rf + 3, rc + 1] += w * (1.0 - s)
                P_gam[rf + 3, rc + 3] += w * (s)

            # -------------------------
            # g22/g23 rows: blend (n,p) in x inside each coarse elem,
            # and Q1 across neighboring coarse elems.
            # -------------------------
            # fine row 4/5: left mid (xl,yl)
            for (ix, iy, w) in bilinear_elems(xl, yl):
                if w == 0.0:
                    continue
                e = elem_id(ix, iy, nxe_c)
                rc = 9 * e
                s = xblend_in_elem(xl, ix)  # blend left->right in that coarse elem
                P_gam[rf + 4, rc + 4] += w * (1.0 - s)  # coarse g22n
                P_gam[rf + 4, rc + 6] += w * (s)        # coarse g22p

                P_gam[rf + 5, rc + 5] += w * (1.0 - s)  # coarse g23n
                P_gam[rf + 5, rc + 7] += w * (s)        # coarse g23p

            # fine row 6/7: right mid (xr,yr)
            for (ix, iy, w) in bilinear_elems(xr, yr):
                if w == 0.0:
                    continue
                e = elem_id(ix, iy, nxe_c)
                rc = 9 * e
                s = xblend_in_elem(xr, ix)
                P_gam[rf + 6, rc + 4] += w * (1.0 - s)
                P_gam[rf + 6, rc + 6] += w * (s)

                P_gam[rf + 7, rc + 5] += w * (1.0 - s)
                P_gam[rf + 7, rc + 7] += w * (s)

            # -------------------------
            # g12 center: Q1 over coarse elements (no extra blend)
            # -------------------------
            for (ix, iy, w) in bilinear_elems(xc, yc):
                if w == 0.0:
                    continue
                e = elem_id(ix, iy, nxe_c)
                rc = 9 * e
                P_gam[rf + 8, rc + 8] += w

        return P_gam

    def _get_locking_linear_system(self, nxe_c:int):
        # sizes
        nxe_f = 2 * nxe_c

        nx_f = nxe_f + 1
        nnodes_f = nx_f**2
        nelems_f = nxe_f**2
        N_f = 6 * nnodes_f

        nx_c = nxe_c + 1
        nnodes_c = nx_c**2
        nelems_c = nxe_c**2
        N_c = 6 * nnodes_c

        # from registration of xpts of assembler to element (in run script)
        fine_xpts = self._xpts_cache[nxe_f]
        coarse_xpts = self._xpts_cache[nxe_c]

        # Build telling-strain operator G_f, G_c (dense):
        # 9 tying strains per element
        G_f = np.zeros((9 * nelems_f, N_f), dtype=float)
        for ielem_f in range(nelems_f):
            ex = ielem_f % nxe_f
            ey = ielem_f // nxe_f
            loc_nodes = np.array([
                ex + nx_f * ey,
                (ex + 1) + nx_f * ey,
                ex + nx_f * (ey + 1),
                (ex + 1) + nx_f * (ey + 1),
            ], dtype=int)
            loc_dof = np.array([6 * node + dof for node in loc_nodes for dof in range(6)], dtype=int)
            loc_xpt_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)], dtype=int)
            elem_xpts = fine_xpts[loc_xpt_dof]
            # print(f"lock-aware fine {elem_xpts.shape=}")

            # loop over each of the tying points (edge DOF and center point)
            quad_pt = [0, -1]
            tying_mat0 = self._get_tying_strain_matrix(quad_pt, elem_xpts)
            g11n = tying_mat0[0,:]
            g13n = tying_mat0[4,:]

            quad_pt = [0, 1]
            tying_mat1 = self._get_tying_strain_matrix(quad_pt, elem_xpts)
            g11p = tying_mat1[0,:]
            g13p = tying_mat1[4,:]

            quad_pt = [-1, 0]
            tying_mat2 = self._get_tying_strain_matrix(quad_pt, elem_xpts)
            g22n = tying_mat2[1,:]
            g23n = tying_mat2[3,:]

            quad_pt = [1, 0]
            tying_mat3 = self._get_tying_strain_matrix(quad_pt, elem_xpts)
            g22p = tying_mat3[1,:]
            g23p = tying_mat3[3,:]

            quad_pt = [0, 0]
            tying_mat4 = self._get_tying_strain_matrix(quad_pt, elem_xpts)
            g12 = tying_mat4[2,:]

            r0 = 9 * ielem_f
            G_f[r0 + 0, loc_dof] += g11n
            G_f[r0 + 1, loc_dof] += g13n
            G_f[r0 + 2, loc_dof] += g11p
            G_f[r0 + 3, loc_dof] += g13p
            G_f[r0 + 4, loc_dof] += g22n
            G_f[r0 + 5, loc_dof] += g23n
            G_f[r0 + 6, loc_dof] += g22p
            G_f[r0 + 7, loc_dof] += g23p
            G_f[r0 + 8, loc_dof] += g12


        G_c = np.zeros((9 * nelems_c, N_c), dtype=float)
        for ielem_c in range(nelems_c):
            ex = ielem_c % nxe_c
            ey = ielem_c // nxe_c
            loc_nodes = np.array([
                ex + nx_c * ey,
                (ex + 1) + nx_c * ey,
                ex + nx_c * (ey + 1),
                (ex + 1) + nx_c * (ey + 1),
            ], dtype=int)
            loc_dof = np.array([6 * node + dof for node in loc_nodes for dof in range(6)], dtype=int)
            loc_xpt_dof = np.array([3 * node + dof for node in loc_nodes for dof in range(3)], dtype=int)
            elem_xpts = coarse_xpts[loc_xpt_dof]

            # print(f"lock-aware coarse {elem_xpts.shape=}")

            # loop over each of the tying points (edge DOF and center point)
            quad_pt = [0, -1]
            tying_mat0 = self._get_tying_strain_matrix(quad_pt, elem_xpts)
            g11n = tying_mat0[0,:]
            g13n = tying_mat0[4,:]

            quad_pt = [0, 1]
            tying_mat1 = self._get_tying_strain_matrix(quad_pt, elem_xpts)
            g11p = tying_mat1[0,:]
            g13p = tying_mat1[4,:]

            quad_pt = [-1, 0]
            tying_mat2 = self._get_tying_strain_matrix(quad_pt, elem_xpts)
            g22n = tying_mat2[1,:]
            g23n = tying_mat2[3,:]

            quad_pt = [1, 0]
            tying_mat3 = self._get_tying_strain_matrix(quad_pt, elem_xpts)
            g22p = tying_mat3[1,:]
            g23p = tying_mat3[3,:]

            quad_pt = [0, 0]
            tying_mat4 = self._get_tying_strain_matrix(quad_pt, elem_xpts)
            g12 = tying_mat4[2,:]

            r0 = 9 * ielem_c
            G_c[r0 + 0, loc_dof] += g11n
            G_c[r0 + 1, loc_dof] += g13n
            G_c[r0 + 2, loc_dof] += g11p
            G_c[r0 + 3, loc_dof] += g13p
            G_c[r0 + 4, loc_dof] += g22n
            G_c[r0 + 5, loc_dof] += g23n
            G_c[r0 + 6, loc_dof] += g22p
            G_c[r0 + 7, loc_dof] += g23p
            G_c[r0 + 8, loc_dof] += g12

        # doesn't make much of a difference here..
        P_gam_method = 'injection'
        # P_gam_method = 'edges'

        if P_gam_method == 'injection':
            # Elementwise injection for tying strains (4-per-elem)
            P_gam = np.zeros((9 * nelems_f, 9 * nelems_c), dtype=float)
            for ielem_f in range(nelems_f):
                ex = ielem_f % nxe_f
                ey = ielem_f // nxe_f
                ielem_c = (ex // 2) + (ey // 2) * nxe_c

                rf = 9 * ielem_f
                rc = 9 * ielem_c
                for i in range(9):
                    P_gam[rf + i, rc + i] = 1.0

        elif P_gam_method == 'edges':
            # interpolate more rigidly (bigger stencil)
            P_gam = self._get_edge_prolong_strain_matrix(nxe_c)

        RHS_full = P_gam @ G_c  # (9*nelems_f, 6*nnodes_c)

        # Baseline nodal prolong
        P_0 = self._build_P2_uncoupled3(nxe_c) # csr
        P_0 = self._apply_bcs_to_P(P_0, nxe_c)
        lam = float(self.lam)

        # Coarse BC columns (same logic as your v2)
        assert self.clamped # haven't written for not clamped case yet
        constrained_dofs = (0, 1, 2, 3, 4, 5) if self.clamped else (0,)

        fixed_cols_c = []
        for inode in range(nnodes_c):
            i = inode % nx_c
            j = inode // nx_c
            if (i == 0) or (i == nx_c - 1) or (j == 0) or (j == nx_c - 1):
                base = 6 * inode
                for a in constrained_dofs:
                    fixed_cols_c.append(base + a)
        fixed_cols_c = np.array(sorted(set(fixed_cols_c)), dtype=int)
        all_cols_c = np.arange(6 * nnodes_c, dtype=int)
        free_cols_c = np.setdiff1d(all_cols_c, fixed_cols_c, assume_unique=False)

        # Fine BC rows with beam-style E-constraint
        fixed_rows_f = []
        for inode in range(nnodes_f):
            i = inode % nx_f
            j = inode // nx_f
            if (i == 0) or (i == nx_f - 1) or (j == 0) or (j == nx_f - 1):
                base = 6 * inode
                for a in constrained_dofs:
                    fixed_rows_f.append(base + a)
        fixed_rows_f = np.array(sorted(set(fixed_rows_f)), dtype=int)

        # -----------------------------
        # v5: impose coarse BCs by zeroing constrained coarse columns
        # in BOTH RHS term and P0 (so system has N_c cols but BC cols are forced to 0)
        # -----------------------------
        if fixed_cols_c.size > 0:
            RHS_full[:, fixed_cols_c] = 0.0
            # also guarantee P0 has those cols zero (in case _apply_bcs_to_P doesn't)
            P_0 = P_0.tolil(copy=True)
            P_0[:, fixed_cols_c] = 0.0
            P_0 = P_0.tocsr()

        # Build normal equations without augmentation (FULL columns)
        solve_rows_f = np.arange(N_f, dtype=int)  # keep all rows
        A = G_f[:, solve_rows_f]           # (4*nelems_f, N_f)
        B = RHS_full                       # (4*nelems_f, N_c)
        P0_all = P_0[solve_rows_f, :].toarray()  # (N_f, N_c), cols multiple of 3

        M   = A.T @ A + lam * np.eye(solve_rows_f.size)
        rhs = A.T @ B + lam * P0_all

        # -----------------------------
        # enforce fine BCs by row/col elimination on M, set rhs row to P0 row (ALL cols)
        # -----------------------------
        if fixed_rows_f.size > 0:
            M[fixed_rows_f, :] = 0.0
            M[:, fixed_rows_f] = 0.0
            for d in fixed_rows_f:
                M[d, d] = 1.0
            rhs[fixed_rows_f, :] = P0_all[fixed_rows_f, :]

        # same some states to element class for DEBUGGING in locking sandbox 
        # need to get it out of this class
        self.G_f = G_f
        self.G_c = G_c
        self.P_gam = P_gam
        self.P_0 = P_0
        self.M = M
        self.RHS = rhs
        self.free_cols_c = free_cols_c
        self.fixed_cols_c = fixed_cols_c
        self.solve_rows_f = solve_rows_f
        # self.P_0_free = P0_free
        self.Mb = None

    def _locking_aware_prolong_global_mitc_v1(self, nxe_c: int, length: float = 1.0):
        """
        Locking-aware prolongation where constraints are on MITC tying strains:
          [ gx(0,-b), gx(0,+b), gy(-a,0), gy(+a,0) ] = 0   per element
        => 4 constraints per element.
        """
        
        self._get_locking_linear_system(nxe_c)
        M = self.M
        rhs = self.RHS
        
        # direct solve
        P = np.linalg.solve(M, rhs)
        return P
    
    # def _locking_aware_prolong_local_mitc_v5_jacobi(
    #         self,
    #         nxe_c: int,
    #         length: float = 1.0,
    #         n_sweeps: int = 10,
    #         omega: float = 0.5,
    #         with_fillin: bool = True,
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
    #     return P


    def _energy_smooth_jacobi_v1(
        self,
        nxe_c: int,
        length: float = 1.0,
        n_sweeps: int = 10,
        omega: float = 0.7,
        with_fillin: bool = True,
        use_mask: bool = True,
    ):
        """
        Standard K-matrix energy smoothing in Jacobi-preconditioned space:
            P <- P - omega * D^{-1} (K P)

        - K is taken from: self._kmat_cache[nxe_c]   (you provide it)
        - Optional fixed sparsity: mask = pattern(K@P) computed once initially.
        - Cache key is ONLY nxe_c (simple).
        """

        import numpy as np
        import scipy.sparse as sp

        # simple cache: ONLY keyed by nxe_c
        # cache_key = ("energy_smooth_jacobi_v1", int(nxe_c))
        # nxe_f = nxe_c * 2
        # cache_key = int(nxe_f)
        # if cache_key in self._lock_P_cache:
        #     return self._lock_P_cache[cache_key]

        # baseline P0
        P0 = self._build_P2_uncoupled3(nxe_c)
        P0 = self._apply_bcs_to_P(P0, nxe_c)
        P = P0.tocsr()

        # kmat from cache
        if not hasattr(self, "_kmat_cache") or (int(nxe_c) not in self._kmat_cache):
            raise RuntimeError("Expected self._kmat_cache[nxe_c] to exist for energy smoothing.")
        K = self._kmat_cache[int(nxe_c)]
        K = K.tocsr() if sp.isspmatrix(K) else sp.csr_matrix(K)

        # Jacobi block inverse (3x3 nodal blocks)
        bs = 3
        N = K.shape[0]
        if (N % bs) != 0 or K.shape[1] != N:
            raise ValueError(f"kmat must be square with size multiple of 3, got {K.shape}")
        nblk = N // bs

        Kb = K.tobsr(blocksize=(bs, bs))
        diag_inv = np.zeros((nblk, bs, bs), dtype=float)
        for i in range(nblk):
            s, e = Kb.indptr[i], Kb.indptr[i + 1]
            cols = Kb.indices[s:e]
            data = Kb.data[s:e]  # (nblocks_in_row, bs, bs)
            k = np.searchsorted(cols, i)
            if k >= cols.size or cols[k] != i:
                raise RuntimeError("Missing diagonal 3x3 block in kmat.")
            Db = data[k]
            Db = 0.5 * (Db + Db.T)
            diag_inv[i] = np.linalg.inv(Db)

        Dinv = sp.bsr_matrix(
            (diag_inv, np.arange(nblk, dtype=np.int32), np.arange(nblk + 1, dtype=np.int32)),
            shape=(N, N),
            blocksize=(bs, bs),
        ).tocsr()

        # fixed sparsity mask from initial K@P
        mask = None
        if (not with_fillin) and use_mask:
            mask = ((K @ P) != 0).astype(np.int8)

        def control(Z):
            if mask is None:
                return Z
            if not sp.issparse(Z):
                Z = sp.csr_matrix(Z)
            return Z.multiply(mask)

        P = control(P)

        # Jacobi-preconditioned energy smoothing: P <- P - omega * Dinv * (K P)
        if with_fillin or (mask is None):
            for _ in range(int(n_sweeps)):
                KP = K @ P
                P = P - float(omega) * (Dinv @ KP)
        else:
            for _ in range(int(n_sweeps)):
                KP = control(K @ P)
                P = control(P - float(omega) * (Dinv @ KP))

        P_out = P.toarray()
        return P_out

    def _assemble_prolongation(self, nxe_fine:int):
        nxe_coarse = nxe_fine // 2
        nxe_f = 2 * nxe_coarse

        if nxe_f in self._P_cache:
            return self._P_cache[nxe_f]

        if self.prolong_mode == "locking-global":
            method = self._locking_aware_prolong_global_mitc_v1
            P = method(nxe_coarse, length=1.0)
        elif self.prolong_mode == 'locking-local':
            P = self._locking_aware_prolong_local_mitc_v5_jacobi(nxe_coarse, length=1.0, n_sweeps=self.n_lock_sweeps, omega=self.omega)
        elif self.prolong_mode == "standard":
            P = self._build_P2_uncoupled3(nxe_coarse)
            P = self._apply_bcs_to_P(P, nxe_coarse)
        elif self.prolong_mode == "energy-jacobi":
            P = self._energy_smooth_jacobi_v1(nxe_coarse, n_sweeps=self.n_lock_sweeps, omega=self.omega, 
                                              with_fillin=True,
                                            #   with_fillin=False,
                                              )
        else:
            raise NotImplementedError("locking-local not implemented in this prototype")
        
        self._P_cache[nxe_f] = sp.csr_matrix(P)
        return P

    def prolongate(self, coarse_u: np.ndarray, nxe_coarse: int):
        dpn = self.dof_per_node
        nxc = nxe_coarse + 1
        Nc = nxc * nxc
        assert coarse_u.size == dpn * Nc

        nxe_f = 2 * nxe_coarse
        nxf = nxe_f + 1
        # Nf = nxf * nxf
        P = self._assemble_prolongation(nxe_f)

        fine_u = P @ coarse_u

        return fine_u

    def restrict_defect(self, fine_r: np.ndarray, nxe_fine: int):
        dpn = self.dof_per_node
        nxf = nxe_fine + 1
        Nf = nxf * nxf
        assert fine_r.size == dpn * Nf
        assert (nxe_fine % 2) == 0

        # nxe_coarse = nxe_fine // 2
        # nxc = nxe_coarse + 1
        # Nc = nxc * nxc
        
        P = self._assemble_prolongation(nxe_fine)
        R = P.T

        fine_r = fine_r.copy()
        # self.apply_bcs_2d(fine_r, nxe_fine)

        coarse_r = R @ fine_r

        return coarse_r
