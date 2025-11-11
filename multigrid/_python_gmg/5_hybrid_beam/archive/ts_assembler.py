__all__ = ["TSAssembler"]

import numpy as np
import scipy as sp
from ._eb_elem import *
from ._ts_elem import *

class TSAssembler:
    def __init__(self, nxe:int, nxh:int, E:float, b:float, L:float, rho:float, 
                 qmag:float, ys:float, rho_KS:float, dense:bool=False, load_fcn=None):
        self.nxe = nxe
        self.nnxh = nxe // nxh
        self.nxh = nxh
        self.E = E
        self.b = b
        self.L = L
        self.rho = rho
        self.rho_KS = rho_KS
        self.qmag = qmag
        self.ys = ys
        self.load_fcn = load_fcn
        if load_fcn is None:
            self.load_fcn = lambda x : np.sin(4.0 * np.pi * x / L)

        self.Kmat = None
        self.force = None
        self.u = None
        # adjoint only required for stress function
        self.psis = None

        # simply supported BCss
        self.num_elements = nxe
        self.num_nodes = nxe + 1
        self.num_dof = 2 * self.num_nodes
        self.xscale = L / nxe
        self.dx = self.xscale
        self.bcs = [0, 2 * (self.num_nodes-1)]
        self.conn = [[ielem, ielem+1] for ielem in range(self.num_elements)]

        self._dense = dense

        # compute rowPtr, colPtr
        self.rowPtr = [0]
        self.colPtr = []
        self.nnzb = 0
        for inode in range(self.num_nodes):
            temp = [ind for elem_conn in self.conn if inode in elem_conn for ind in elem_conn]
            temp = np.unique(np.array(temp))

            self.nnzb += temp.shape[0]
            self.rowPtr += [self.nnzb]
            self.colPtr += list(temp)
        self.rowPtr = np.array(self.rowPtr)
        self.colPtr = np.array(self.colPtr)
        # print(f"{self.rowPtr=}")
        # print(f"{self.colPtr=}")

    def _compute_mat_vec(self, helem_vec):
        # copy states out
        E = self.E; b = self.b; rho = self.rho; nxe = self.nxe; L = self.L; rho_KS = self.rho_KS

        self.data = np.zeros((self.nnzb, 2, 2), dtype=helem_vec.dtype)

        # define the element loads here
        xvec = [(ielem+0.5) * self.dx for ielem in range(self.num_elements)]
        # qvec = [self.qmag * np.sin(4.0 * np.pi * xval / L) for xval in xvec]
        qvec = [self.qmag * self.load_fcn(xval) for xval in xvec]

        # compute Kelem without EI scaling
        elem_xpts = np.array([0.0]*3 + [self.dx, 0.0, 0.0])
        qvars = np.array([0.0] * 12)
        ref_axis = np.array([0.0, 1.0, 0.0])
        nu=0.3
        G = E / 2.0 / (1 + nu)
        material = Material(E=E, rho=rho, nu=nu, G=G, k_s=5.0 / 6.0, ys=self.ys)
        CK = get_constitutive_data(material, b, helem_vec[0]) # assuming constant thick right now (just simple demos here to compare multigrid soo.. not need full solver)
        Kelem_nom = get_stiffness_matrix(elem_xpts, qvars, ref_axis, CK)

        Kelem_nom *= 0.1 # ? why does this fix rescaling?
        # print(f"{Kelem_nom=}")
        # exit()

        twod_beam_dof = np.array([2, 4, 8, 10]) # w and thetay
        Kelem_nom = Kelem_nom[twod_beam_dof,:][:,twod_beam_dof]
        felem_nom = np.array([1.0, 0.0, 1.0, 0.0]) * self.dx #* self.dx**2
        # plt.imshow(Kelem_nom)
        # plt.show()

        num_dof = 2 * self.num_nodes
        if self._dense:
            Kmat = np.zeros((num_dof, num_dof), dtype=helem_vec.dtype)
        # want to start as sparse matrix now
        else:
            Kmat = None

        force = np.zeros((num_dof,))

        for ielem in range(self.num_elements): 
            local_conn = np.array(self.dof_conn[ielem])

            if self._dense:
                np.add.at(Kmat, (local_conn[:,None], local_conn[None,:]), Kelem_nom)
            else: # sparse
                # add into sparse data 
                Kelem = Kelem_nom
                # print(f'{Kelem=}')
                # plt.imshow(Kelem)
                # plt.show()

                # loop through rowPtr, colPtr data structure
                for row_node in [ielem, ielem+1]:
                    start = self.rowPtr[row_node]
                    end = self.rowPtr[row_node+1]
                    inode = row_node - ielem
                    for p in range(start, end):
                        col_node = self.colPtr[p]
                        if col_node in [ielem, ielem+1]:
                            # print(f"{ielem=} conn={[ielem, ielem+1]} {p=}")
                            jnode = col_node - ielem
                            # print(f"{p=} {inode=} {jnode=}")
                            Kelem_loc = Kelem[2*inode:(2*inode+2)][:,2*jnode:(2*jnode+2)]
                            self.data[p,:,:] += Kelem_loc                            
            
            q = qvec[ielem]
            np.add.at(force, local_conn, q * felem_nom)

        # now apply simply supported BCs
        bcs = [0, 2 * (self.num_nodes-1)]

        # apply dirichlet w=0 BCs
        if self._dense:
            for bc in bcs:
                Kmat[bc,:] = 0.0
                Kmat[:, bc] = 0.0
                Kmat[bc, bc] = 1.0
        else: # sparse
            # just explicitly do it for now
            self.data[0:2,0,:] = 0.0
            self.data[[0,2],:,0] = 0.0
            self.data[0,0,0] = 1.0

            bc = 2 * (self.num_nodes-1)
            self.data[self.nnzb-2:,0,:] = 0.0
            self.data[self.nnzb-3,:,0] = 0.0
            self.data[self.nnzb-1,:,0] = 0.0
            self.data[self.nnzb-1,0,0] = 1.0

        for bc in bcs:
            force[bc] = 0.0
            
        # convert to sparse matrix (or could have stored as sparse originally)
        # Kmat = sp.sparse.csr_matrix(Kmat)

        if not self._dense:
            Kmat = sp.sparse.bsr_matrix(
                (self.data, self.colPtr, self.rowPtr), 
                shape=(2*self.num_nodes, 2*self.num_nodes))
            
            # convert to csr since doesn't support bsr in scipy spsolve
            Kmat = Kmat.tocsr()

        # store in object
        self.Kmat = Kmat
        self.force = force
        return Kmat, force
    
    def _compute_psiresid(self, hvec):
        # update kmat, compute psi^T * R
        self._compute_mat_vec(hvec)
        resid = np.dot(self.Kmat.toarray(), self.u)
        return np.dot(self.psis, resid)
    
    def _compute_dRdx(self, hvec):
        # copy states out
        E = self.E; b = self.b; rho = self.rho; nxe = self.nxe; L = self.L; rho_KS = self.rho_KS

        # gradient to compute
        Rgrad = [0.0]*self.num_elements

        # compute Kelem without EI scaling
        Kelem_nom = get_kelem(self.xscale)
        felem_nom = get_felem(self.xscale)

        for ielem in range(self.num_elements):
            # do the gradient purely element wise (so very efficient to compute)
            
            # get local psi vector
            local_conn = np.array(self.dof_conn[ielem])
            psi_local = self.psis[local_conn]

            # get local disp vector
            u_local = self.u[local_conn]

            # get any local bcs
            local_node_conn = self.conn[ielem]

            # compute local Kelem thickness derivative
            EI = E * b * hvec[ielem]**3 / 12.0
            dKelem = Kelem_nom * EI * 3 / hvec[ielem]

            # apply local bcs to dKelem matrix
            start_node = local_node_conn[0]
            for node in local_node_conn:
                if node in self.bcs:
                    node_off = node - start_node
                    dKelem[2*node_off,:] = 0.0
                    dKelem[:,2*node_off] = 0.0
                    dKelem[2*node_off, 2*node_off] = 1.0
            
            # now compute quadratic product for gradient
            Rgrad[ielem] = np.dot(psi_local, np.dot(dKelem, u_local))
        # return np.array(Rgrad)
        # then convert to reduced dx gradient
        return np.array(self.helem_to_hred_vec(Rgrad))
    
    def get_helem_vec(self, hred):
        helem_vec = np.zeros((self.num_elements,), dtype=hred.dtype)
        for ix in range(self.nxe):
            ired = ix // self.nnxh
            helem_vec[ix] = hred[ired]
        return helem_vec
    
    def helem_to_hred_vec(self, dhelem_vec):
        _hred = np.zeros((self.nxh,))
        for ix in range(self.nxe):
            ired = ix // self.nnxh
            _hred[ired] += dhelem_vec[ix]
        return _hred

    def helem_to_hred_max_vec(self, dhelem_vec):
        _hred = np.zeros((self.nxh,))
        for ix in range(self.nxe):
            ired = ix // self.nnxh
            _hred[ired] = np.max([_hred[ired], dhelem_vec[ix]])
        return _hred

    def solve_forward(self, hred):
        # now solve the linear system
        helem_vec = self.get_helem_vec(hred)
        self._compute_mat_vec(helem_vec)

        if self._dense:
            self.u = np.linalg.solve(self.Kmat, self.force)
        else:
            # print(f"{self.Kmat.toarray()=}")
            self.u = sp.sparse.linalg.spsolve(self.Kmat, self.force)

        return self.u

    def solve_adjoint(self, hred):
        helem_vec = self.get_helem_vec(hred)
        # now solve the adjoint system
        self._compute_mat_vec(helem_vec)

        # adjoint solve only required for stress function (mass is non-adjoint no state var dependence)
        KT = self.Kmat.T
        stress_rhs = self._compute_dstressdu(helem_vec)

        if self._dense:
            self.psis = np.linalg.solve(KT, -stress_rhs)
        else:
            self.psis = sp.sparse.linalg.spsolve(KT, -stress_rhs)
        return self.psis

    @property
    def dof_conn(self):
        return [[2 * ix+_ for _ in range(4)] for ix in range(self.nxe)]

    def _compute_stresses_sq(self, helem_vec):
        # copy states out
        E = self.E; b = self.b; rho = self.rho; nxe = self.nxe; L = self.L; rho_KS = self.rho_KS
        xscale = L / nxe

        # compute stresses
        stress_vec = []
        for ielem in range(nxe):
            local_conn = self.dof_conn[ielem]
            local_disp = self.u[local_conn]
            hess_vec = []
            for ibasis in range(4):
                xi = 0.0
                hess_vec += [get_hess(ibasis, xi, xscale) * local_disp[ibasis]]
            hess_elem = sum(hess_vec)
            stress_elem = E * helem_vec[ielem]/2 * hess_elem / self.ys
            # if ielem == 0:
            #     print(f"{xscale=} {local_disp=} {stress_elem=}")
            stress_vec += [stress_elem**2] # compute stress^2 < 1 here
        stress_vec = np.array(stress_vec) 
        return stress_vec
    
    """
    -------------------------
    matrix-free methods here
    -------------------------
    """
    def get_mass(self, hred):
        helem_vec = self.get_helem_vec(hred)
        # copy states out
        E = self.E; b = self.b; rho = self.rho; nxe = self.nxe; L = self.L; rho_KS = self.rho_KS

        # compute mass
        mass_vec = [E * b * helem_vec[ielem] * rho for ielem in range(nxe)]
        return sum(mass_vec)
    
    def get_mass_gradient(self):
        # copy states out
        E = self.E; b = self.b; rho = self.rho; nxe = self.nxe; L = self.L; rho_KS = self.rho_KS

        # compute mass gradient
        dmass = np.array([E * b * rho] * nxe)
        dmass_red = self.helem_to_hred_vec(dmass)
        return dmass_red
    
    def get_analytic_opt_thicknesses(self, hred, Mred):
        # f_i = P_i / h_i^2 => h_i = sqrt(P_i / f_i) is optimal design
        Pvec = 6.0 * Mred / self.b / self.ys # where Mred are the constant bending moments
        fail_index = self.get_comp_fail_indexes(hred, Mred)
        # print(f"{fail_index=}")

        # check fails
        hred_opt = np.sqrt(Pvec) # as if fails = 1
        # print(f"{hred_opt[:10]=}")
        fails_opt = self.get_comp_fail_indexes(hred_opt, Mred)
        # print(f"{fails_opt[:10]=}")
        # print(f"{hred_opt=}")
        mopt = self.get_mass(hred_opt)
        # print(f"{mopt=}")
        return hred_opt
    
    def _get_opt_feasibility(self, hred, hred_opt, fails):
        # helper function
        # compute the analytic solution of hred first (adapted from self.get_comp_fail_indexes)

        hdiff = hred - hred_opt
        # print(f"{hdiff=}")

        # optimality (compare mass to opt mass)
        m = self.get_mass(hred)
        mstar = self.get_mass(hred_opt)
        # print(f"{m=} {mstar=}")
        opt = abs((m - mstar) / mstar)

        # feasibility, compute |elem_relu(f - 1)| / |1| (only penalizes f > 1 here)
        relu = lambda x : x if x > 0 else 0.0
        c_relu = np.array([relu(1 - f) for f in fails])
        feas = np.linalg.norm(c_relu) / np.linalg.norm(np.ones(hred.shape))
        return opt, feas

    def get_opt_feasibility(self, hred, Mred):
        # get the optimality and feasibility of a design
        hred_opt = self.get_analytic_opt_thicknesses(hred, Mred)

        fails = self.get_comp_fail_indexes(hred, Mred)
        # print(f"{fails=}")
        opt, feas = self._get_opt_feasibility(hred, hred_opt, fails)
        return opt, feas

    def compute_component_bend_moments(self, hred):
        # compute the component - constant bending moments M_i^{max} for matrix-free (once before optimization since it's constant)
        self.solve_forward(hred)

        helem_vec = self.get_helem_vec(hred)
        elem_moments = np.abs(self._compute_element_bending_moments(helem_vec))
        comp_max_moments = self.helem_to_hred_max_vec(elem_moments)
        
        return comp_max_moments
    
    def get_comp_fail_indexes(self, hred_vec, Mred_vec):
        # for matrix-free opt => constraints c_i

        # f_i = M_i * (0.5 * h_i) / I / sigma_ys = 6 * M_i / (b * sigma_ys * h_i^2) = P_i / h_i^2
        Pvec = 6.0 * Mred_vec / self.b / self.ys
        fail_indexes = Pvec / np.power(hred_vec, 2.0)
        return fail_indexes
    
    def get_comp_fail_indexes_jac(self, hred_vec, Mred_vec):
        # for matrix-free opt A = c_x diag matrix

        # f_i = M_i * (0.5 * h_i) / I / sigma_ys = 6 * M_i / (b * sigma_ys * h_i^2) = P_i / h_i^2
        Pvec = 6.0 * Mred_vec / self.b / self.ys
        fail_indexes = Pvec / hred_vec**2

        # then jacobian will just be -2 * fail_indexes / hred_vec
        return -2.0 * fail_indexes / hred_vec # the vec => diag matrix
    
    def get_comp_fail_indexes_lamhess(self, hred_vec, Mred_vec, lambda_vec):
        # for matrix-free opt lam^T c_{xx} diag matrix

        # f_i = M_i * (0.5 * h_i) / I / sigma_ys = 6 * M_i / (b * sigma_ys * h_i^2) = P_i / h_i^2
        Pvec = 6.0 * Mred_vec / self.b / self.ys
        fail_indexes = Pvec / hred_vec**2

        diag_hess = -6.0 * fail_indexes / hred_vec**2
        return diag_hess * lambda_vec
    
    def _compute_element_bending_moments(self, helem_vec):
        # copy states out
        E = self.E; b = self.b; rho = self.rho; nxe = self.nxe; L = self.L; rho_KS = self.rho_KS
        xscale = L / nxe
        
        moments = []

        # compute stresses
        for ielem in range(nxe):
            local_conn = self.dof_conn[ielem]
            local_disp = self.u[local_conn]
            hess_vec = []
            for ibasis in range(4):
                xi = 0.0
                hess_vec += [get_hess(ibasis, xi, xscale) * local_disp[ibasis]]
            hess_elem = sum(hess_vec)

            # get the bending moment M = EI * w''
            I = b * helem_vec[ielem]**3 / 12.0
            moment = E * I * hess_elem
            moments += [moment]

        moments = np.array(moments)
        # print(f"{self.u=}")
        # print(f"{moments=}")
        return moments
    
    """
    -------------------------
    matrix-free methods end here
    -------------------------

    ------------------
    FSD methods begin
    ------------------
    """
    
    def _compute_comp_stresses(self, hvec):
        helem_vec = self.get_helem_vec(hvec)
        elem_stresses = np.sqrt(self._compute_stresses_sq(helem_vec)) # that was sqrt stresses..
        return self.helem_to_hred_max_vec(elem_stresses)

    
    def _compute_ks_stress(self, hvec):
        rho_KS = self.rho_KS
        stress_vec = self._compute_stresses_sq(hvec)
        true_max = np.max(stress_vec)
        max_stress = true_max + 1.0/rho_KS * np.log(np.sum(np.exp(rho_KS * (stress_vec - true_max) )))
        return max_stress
    
    def _compute_dstressdu(self, helem_vec):
        # copy states out
        E = self.E; b = self.b; rho = self.rho; nxe = self.nxe; L = self.L; rho_KS = self.rho_KS
        xscale = L / nxe

        stress_vec = self._compute_stresses_sq(helem_vec)
        true_max = np.max(stress_vec)

        # now compute derivatives here
        dstress_i = np.exp(rho_KS * (stress_vec - true_max))
        dstress_i /= np.sum(dstress_i) # partition of unity

        num_dof = 2 * (nxe + 1)
        du_global = np.zeros((num_dof,))
        for ielem in range(nxe):
            local_conn = self.dof_conn[ielem]
            local_disp = self.u[local_conn]
            hess_vec = []
            du_vec = []
            for ibasis in range(4):
                xi = 0.0
                hess_vec += [get_hess(ibasis, xi, xscale) * local_disp[ibasis]]
                du_vec += [get_hess(ibasis, xi, xscale)]
            hess_elem = sum(hess_vec)
            stress_elem = E * helem_vec[ielem]/2 * hess_elem / self.ys
            du_vec = np.array(du_vec) * E / self.ys * helem_vec[ielem]/2 * dstress_i[ielem] * 2.0 * stress_elem

            du_global[local_conn] += du_vec
        
        return du_global
    
    def _compute_dstressdx(self, helem_vec):
        # copy states out
        E = self.E; b = self.b; rho = self.rho; nxe = self.nxe; L = self.L; rho_KS = self.rho_KS
        xscale = L / nxe

        stress_vec = self._compute_stresses_sq(helem_vec)
        true_max = np.max(stress_vec)

        # now compute derivatives here
        dstress_i = np.exp(rho_KS * (stress_vec - true_max))
        dstress_i /= np.sum(dstress_i) # partition of unity

        dh_vec = [0.0] * nxe
        for ielem in range(nxe):
            local_conn = self.dof_conn[ielem]
            local_disp = self.u[local_conn]
            hess_vec = []
            du_vec = []
            for ibasis in range(4):
                xi = 0.0
                hess_vec += [get_hess(ibasis, xi, xscale) * local_disp[ibasis]]
                du_vec += [get_hess(ibasis, xi, xscale)]
            hess_elem = sum(hess_vec)
            stress_elem = E / self.ys * helem_vec[ielem]/2 * hess_elem
            dh_vec[ielem] = 2.0 * stress_elem * E / self.ys /2.0 * hess_elem * dstress_i[ielem]
        
        # return dh_vec
        return np.array(self.helem_to_hred_vec(dh_vec))

    def get_functions(self, hred):
        helem_vec = self.get_helem_vec(hred)
        # copy states out
        E = self.E; b = self.b; rho = self.rho; nxe = self.nxe; L = self.L; rho_KS = self.rho_KS

        # compute mass
        mass_vec = [E * b * helem_vec[ielem] * rho for ielem in range(nxe)]
        mass = sum(mass_vec)

        stress_vec = self._compute_stresses_sq(helem_vec)
        true_max = np.max(stress_vec)
        max_stress = true_max + 1.0/rho_KS * np.log(np.sum(np.exp(rho_KS * (stress_vec - true_max) )))

        return np.array([mass, max_stress])
    
    def get_function_gradients(self, hred):
        helem_vec = self.get_helem_vec(hred)
        # copy states out
        E = self.E; b = self.b; rho = self.rho; nxe = self.nxe; L = self.L; rho_KS = self.rho_KS

        # compute mass gradient
        dmass = np.array([E * b * rho] * nxe)
        dmass_red = self.helem_to_hred_vec(dmass)

        # stress gradient
        dstress = np.array(self._compute_dstressdx(helem_vec))
        dstress = np.reshape(dstress, newshape=(hred.shape[0],))
        dRdx_term = self._compute_dRdx(helem_vec)

        # dRdx_term = np.reshape(dRdx_term, newshape=(hvec.shape[0], 1))
        dstress += dRdx_term

        # print(f"{dmass_red.shape=} {dstress.shape=}")

        return np.array([dmass_red, dstress])
    
    def complex_step_test(self, hvec, h:float=1e-30):
        p = np.random.rand(self.nxh)
        hvec = np.array(hvec)

        # adjoint value
        self.solve_forward(hvec)
        self.solve_adjoint(hvec)
        # funcs = self.get_functions(hvec)
        func_grads = self.get_function_gradients(hvec)
        
        print(f"{func_grads.shape=}")

        nfunc = 2
        adj_prods = [
            np.dot(func_grads[ifunc,:], p) for ifunc in range(nfunc)
        ]

        # complex-step product
        hvec2 = hvec + h * 1j * p
        self.solve_forward(hvec2)
        funcs2 = self.get_functions(hvec2)
        cs_prods = [
            np.imag(func_val) / h for func_val in funcs2
        ]

        print(f"{adj_prods=}")
        print(f"{cs_prods=}")

    def test_dRdx(self, hvec, h:float=1e-30):
        # this one works!
        p = np.random.rand(self.nxh)
        hvec = np.array(hvec)

        # adjoint value
        self.solve_forward(hvec)
        self.solve_adjoint(hvec)
        adj_prod = np.dot(p, self._compute_dRdx(hvec))

        # complex-step product
        hvec2 = hvec + h * 1j * p
        new_resid = self._compute_psiresid(hvec2)
        cs_prod = np.imag(new_resid) / h

        print("check dRdx")
        print(f"\t{adj_prod=}")
        print(f"\t{cs_prod=}")

    def test_dsdu(self, hvec, h:float=1e-30):
        # this one works!
        p = np.random.rand(2*self.num_nodes)

        # pre solve for u
        self.solve_forward(hvec)

        # adjoint value
        du_vec = self._compute_dstressdu(hvec)
        adj_prod = np.dot(p, du_vec)

        # complex-step product
        self.u = self.u.astype(np.complex128)
        self.u += + h * 1j * p
        new_stress = self._compute_ks_stress(hvec)
        cs_prod = np.imag(new_stress) / h

        print("check dstress/du")
        print(f"\t{adj_prod=}")
        print(f"\t{cs_prod=}")

    def test_dsdx(self, hvec, h:float=1e-30):
        # this one works!
        p = np.random.rand(self.nxh)

        # pre solve for u
        self.solve_forward(hvec)

        # adjoint value
        dx_vec = self._compute_dstressdx(hvec)
        adj_prod = np.dot(p, dx_vec)

        # complex-step product
        hvec2 = hvec + h * 1j * p
        new_stress = self._compute_ks_stress(hvec2)
        cs_prod = np.imag(new_stress) / h

        print("check dstress/dx")
        print(f"\t{adj_prod=}")
        print(f"\t{cs_prod=}")

    @property
    def xvec(self) -> list:
        return [i*self.dx for i in range(self.num_nodes)]

    def plot_disp(self):
        xvec = self.xvec
        # print(f"{self.u=}")
        w = self.u[0::2]
        plt.figure()
        plt.plot(xvec, w)
        plt.plot(xvec, np.zeros((self.num_nodes,)), "k--")
        plt.xlabel("x")
        plt.ylabel("w(x)")
        plt.show()     

    def plot_thickness(self, hred):
        xvec = self.xvec
        helem_vec = self.get_helem_vec(hred)
        # hvec += [hvec[-1]]
        plt.figure()
        plt.plot(xvec[:-1], helem_vec)
        plt.plot(xvec[:-1], np.zeros((self.nxe,)), "k--")
        plt.xlabel("x")
        plt.ylabel("thick(x)")
        plt.show()     