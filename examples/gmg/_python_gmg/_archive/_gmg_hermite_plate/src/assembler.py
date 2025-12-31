__all__ = ["PlateAssembler", "PlateLoads", "PlateFemGeom", "IsotropicMaterial"]

import numpy as np
import scipy as sp
from ._plate_elem import *
from ._numba import *
from ._helpers import PlateLoads, PlateFemGeom, IsotropicMaterial, plot_init
from matplotlib.ticker import FormatStrFormatter

class PlateAssembler:
    """
    lightweight version of the plate element from log_struct_opt repo with only sparse and numba options..
    for testing geometric multigrid methods
    """
    def __init__(
        self, 
        material:IsotropicMaterial, 
        plate_fem_geom:PlateFemGeom,
        plate_loads:PlateLoads,
        rho_KS:float, 
        can_print:bool=False
    ):
        
        # unpack data from the objects
        E, nu, rho, ys = material.E, material.nu, material.rho, material.ys
        nxe, nye, nxh, nyh, a, b = plate_fem_geom.nxe, plate_fem_geom.nye, plate_fem_geom.nxh, plate_fem_geom.nyh, plate_fem_geom.a, plate_fem_geom.b
        qmag, load_fcn = plate_loads.qmag, plate_loads.load_fcn

        self.nxe = nxe
        self.nye = nye

        # nxh by nyh components (separate from elements)
        self.nxh = nxh
        self.nyh = nyh
        self.ncomp = nxh * nyh
        # nnxh by nnyh number of elements per component in each direction
        self.nnxh = nxe // nxh
        self.nnyh = nye // nyh

        self.E = E
        self.nu = nu
        self.a = a
        self.b = b
        self.rho = rho
        self.rho_KS = rho_KS
        self.qmag = qmag
        self.ys = ys
        self.load_fcn = load_fcn
        self.can_print = can_print

        # solve states
        self.Kmat = None
        self.force = None
        self.u = None
        # adjoint only required for stress function
        self.psis = None

        # prelim mesh info
        self.num_elements = nxe * nye
        self.nnx = nxe + 1; self.nny = nye + 1
        self.num_nodes = (nxe + 1) * (nye+1)
        self.num_dof = 3 * self.num_nodes
        self.xscale = a / nxe
        self.yscale = b / nye
        self.dx = self.xscale
        self.dy = self.yscale

        self.dof_per_node = 3

        # define bcs
        self.bcs = []
        for iy in range(self.nny):
            for ix in range(self.nnx):
                inode = iy * self.nnx + ix
                if ix == 0 or ix == self.nnx - 1 or iy == 0 or iy == self.nny-1:
                    self.bcs += [3 * inode]
                if iy == 0 or iy == self.nny - 1: # x edge with y=const
                    self.bcs += [3 * inode + 1] # dw/dx = 0
                if ix == 0 or ix == self.nnx - 1: # y edge with x=const
                    self.bcs += [3 * inode + 2] # dw/dy = 0
        self.bcs = np.array(self.bcs)

        # also define finer grid bcs for multigrid (TODO : should we also apply rot bcs on boundary? for multigrid)
        self.fine_bcs = []
        for iy in range(2 * self.nny - 1):
            for ix in range(2 * self.nnx - 1):
                inode = iy * (2 * self.nnx - 1) + ix
                w_idof = 3 * inode
                if ix == 0 or ix == 2 * self.nnx - 2 or iy == 0 or iy == 2 * self.nny - 2:
                    self.fine_bcs += [w_idof]
                if iy == 0 or iy == 2 * self.nny - 2: # x edge with y=const
                    self.fine_bcs += [3 * inode + 1] # dw/dx = 0
                if ix == 0 or ix == 2 * self.nnx - 2: # y edge with x=const
                    self.fine_bcs += [3 * inode + 2] # dw/dy = 0
        self.fine_bcs = np.array(self.fine_bcs)
        # print(F"{np.max(self.fine_bcs)=} {self.num_fine_dof=}")
    
        # define element connectivity (node dof)
        self.conn = []
        for iy in range(nye):
            for ix in range(nxe):
                istart = ix + self.nnx * iy
                istart2 = istart + self.nnx

                self.conn += [[istart, istart+1, istart2, istart2+1]]

        # define the node locations here
        self.xpts = []
        xoffset = -a/2 # center at origin
        yoffset = -b/2
        for iy in range(self.nny):
            for ix in range(self.nnx):

                xval = xoffset + ix * self.dx
                yval = yoffset + iy * self.dy 
                zval = 0.0
                self.xpts += [xval, yval, zval]

        # define the element loads here
        self.elem_loads = []
        for iy in range(nye):
            for ix in range(nxe):
                xval = (ix+0.5)*self.dx
                yval = (iy+0.5)*self.dy
                self.elem_loads += [load_fcn(xval, yval)]

        self.nelem_per_comp = self.num_elements // self.ncomp

        # compute CSR rowp, cols (don't think I need BSR for simple plate in just bending, prob overkill and BSR not well supported in python anyways)
        self.rowp = None
        self.cols = None
        self.rows_scipy = None
        self._compute_CSR_pattern()

        # compute elem to component DV map
        self.elem_comp_map = None
        self._compute_elem_to_comp_map()

        # get the integer elem => CSR assembly map (for faster assembly)
        self.elem_to_csr_map = None
        self._compute_elem_to_CSR_assembly_map()

        # similar bc cols CSR map (for fast bc cols application, bc rows is easier)
        self.bc_cols_csr_map = None
        self._compute_bc_cols_map()

        # store EI=1 (un-scaled Kelem and felem for faster assembly)
        self.Kelem_nom = get_kelem(1.0, self.xscale, self.yscale)
        self.Kelem_nom_flat = self.Kelem_nom.flatten()
        self.felem_nom = get_felem(self.xscale, self.yscale)

    @classmethod
    def aluminum_unitsquare_trigload(
        cls,
        num_elements:int, # num elements, num_components must be perfect squares and divide evenly
        num_components:int,
        rho_KS:float=100.0, 
        qmag:float=2e-2,
        can_print:bool=False
    ):
        # main test case I'm using for snopt, FSD, INK comparison rn
        return cls(
            material=IsotropicMaterial.aluminum(),
            plate_fem_geom=PlateFemGeom.unit_square(num_elements, num_components),
            plate_loads=PlateLoads.game_of_life_trig_load(qmag=qmag),
            rho_KS=rho_KS,
            can_print=can_print,
        )

    @property
    def dof_conn(self):
        return np.array([[3*ind+_ for ind in elem_conn for _ in range(3)] for elem_conn in self.conn])
    
    def _compute_CSR_pattern(self):
        """jelper method upon construction compute the CSR matrix rowp, cols pattern (nofill) """
        import time
        t0 = time.time()
        if self.can_print: print("compute CSR pattern")

        # code found in _numba.py in plate folder
        nnz, rowp, cols, rows_scipy = _compute_CSR_pattern_numba(
            self.num_nodes, self.num_elements, np.array(self.conn)
        )
        self.nnz = nnz
        self.rowp = rowp
        self.cols = cols
        self.rows_scipy = rows_scipy

        dt = time.time() - t0
        if self.can_print: print(f"CSR pattern computed in {dt=:.4e} sec")

    def _compute_elem_to_comp_map(self):
        self.elem_comp_map = np.zeros((self.num_elements,), dtype=np.int32)

        for iy in range(self.nye):
            iy_comp = iy // self.nnyh
            for ix in range(self.nxe):
                ix_comp = ix // self.nnxh
                icomp = iy_comp * self.nxh + ix_comp
                ielem = iy * self.nxe + ix

                self.elem_comp_map[ielem] = icomp

    def _compute_elem_to_CSR_assembly_map(self):
        """upon construction pre-compute an integer map for the assembly"""
        import time
        t0 = time.time()
        dof_per_elem = 12
            
        # code found in _numba.py in plate folder
        self.elem_to_csr_map = _compute_elem_to_CSR_assembly_map_numba(
            self.num_elements, dof_per_elem, self.dof_conn, self.rowp, self.cols)

        dt = time.time() - t0
        if self.can_print: print(f"CSR assembly map computed in {dt=:.4e} sec")
        return
    
    def _compute_bc_cols_map(self):
        """on construction pre-compute for each bc, which rows does the bc col appear in (col bcs slower to apply without this)"""
        import time
        t0 = time.time() 

        # this code is found in _numba.py
        bc_counts = _compute_colbc_row_counts_numba(self.bcs, self.num_dof, self.rowp, self.cols)
        bc_offsets, colbc_rows = _compute_bc_cols_csr_map_numba(self.bcs, self.num_dof, self.rowp, self.cols, bc_counts)

        self.bc_cols_csr_offsets = bc_offsets
        self.bc_cols_csr_rows = colbc_rows

        dt = time.time() - t0
        if self.can_print: print(f"CSR bc cols map computed in {dt=:.4e} sec")
        return

    """
    end of constructor or constructor helper methods
    ------------------------------------------------------
    start of general red-space solve utils section
        includes primal, adjoint methods used in both INK and SNOPT KS-aggreg
    ------------------------------------------------------
    """

    def get_helem_vec(self, hcomp_vec):
        return np.array([hcomp_vec[self.elem_comp_map[ielem]] for ielem in range(self.num_elements)])
    
    def _compute_mat_vec(self, hcomp_vec):
        """compute Kmat and F (lhs and rhs)"""
        helem_vec = self.get_helem_vec(hcomp_vec)
        self._compute_sparse_mat_vec(helem_vec)

    def _compute_sparse_mat_vec(self, helem_vec):
        """sparse CSR matrix assembly (and RHS)"""

        # to hold the csr data (for making scipy CSR Kmat at end of method)
        csr_data = np.zeros((self.nnz), dtype=helem_vec.dtype)       
        dof_per_node = self.dof_per_node 
        num_dof = dof_per_node * self.num_nodes
        global_force = np.zeros((num_dof,))

        kelem_scales = self.E * helem_vec**3 / 12.0 / (1 - self.nu**2) # = D the flexural modulus
        felem_scales = np.array(self.elem_loads)
        _helper_numba_assembly_serial(
            self.num_elements, self.elem_to_csr_map, self.dof_conn,
            csr_data, self.Kelem_nom_flat, kelem_scales,
            global_force, self.felem_nom, felem_scales)

        # now apply simply supported BCs
        bcs = self.bcs

        # apply dirichlet w=0 BCs
        # first to zero out rows for each bc (except diag entry)
        for bc in bcs:
            glob_row = bc
            for csr_ind in range(self.rowp[glob_row], self.rowp[glob_row+1]):
                glob_col = self.cols[csr_ind]
                csr_data[csr_ind] = glob_col == glob_row
        
        # now zero out cols
        for i_bc,bc in enumerate(bcs):
            # use the bc_cols_map to get which rows include this bc col (otherwise we need like a quadruple for loop and it's very slow 
            # (so use pre-computed map on construction)
            # included_rows = self.bc_cols_csr_map[bc] # without bc numba
            included_rows = self.bc_cols_csr_rows[self.bc_cols_csr_offsets[i_bc] : self.bc_cols_csr_offsets[i_bc + 1]]

            for glob_row in included_rows:
                for csr_ind in range(self.rowp[glob_row], self.rowp[glob_row+1]):
                    glob_col = self.cols[csr_ind]
                    if glob_col == bc:
                        csr_data[csr_ind] = glob_col == glob_row

        # zero out bcs in force vector
        global_force[bcs] = 0.0

        # store in object
        self.Kmat = sp.sparse.csr_matrix((csr_data, (self.rows_scipy, self.cols)), shape=(num_dof, num_dof))
        self.force = global_force

        return
    
    def helem_to_hcomp_vec(self, dhelem_vec):
        dhcomp_vec = np.zeros((self.ncomp,))
        for ielem in range(self.num_elements):
            icomp = self.elem_comp_map[ielem]
            dhcomp_vec[icomp] += dhelem_vec[ielem]
        return dhcomp_vec

    def solve_forward(self, hcomp_vec):
        # now solve the linear system
        self._compute_mat_vec(hcomp_vec)

        # direct sparse linear solve 
        self.u = sp.sparse.linalg.spsolve(self.Kmat, self.force)
        return self.u

    def solve_adjoint(self, hcomp_vec):
        # now solve the adjoint system
        self._compute_mat_vec(hcomp_vec)

        # adjoint solve only required for stress function (mass is non-adjoint no state var dependence)
        KT = self.Kmat.T
        adj_rhs = self._get_dkstot_du(hcomp_vec)

        # direct sparse linear solve 
        self.psis = sp.sparse.linalg.spsolve(KT, -adj_rhs)
        return self.psis

    def _get_elem_fails(self, helem_vec):
        """helper function that computes the failure index in each element"""

        # pre-compute hessian data
        hess_basis = np.zeros((12, 3))
        for ibasis in range(12):
            for idof in range(3):
                hess_basis[ibasis, idof] = get_hessians(ibasis, xi=0.0, eta=0.0, xscale=self.dx, yscale=self.dy)[idof]
        Dvec = self.E * helem_vec**3 / 12.0 / (1 - self.nu**2)

        # this code is found in _numba.py
        elem_fails = _get_elem_fails_numba(
            self.num_elements, Dvec, self.dof_conn, self.u,
            hess_basis, self.nu, helem_vec, self.ys)
        
        return elem_fails
    
    def _get_elem_fails_DVsens(self, delem_fails, helem_vec, elem_fails):
        dhelem_vec = [0.0] * self.num_elements
        for ielem in range(self.num_elements):           
            # shouldn't the coefficient be -2.0 here? for some reason it works with -1.0 
            dhelem_vec[ielem] += elem_fails[ielem] * 1.0 / helem_vec[ielem] * delem_fails[ielem]
        return np.array(self.helem_to_hcomp_vec(dhelem_vec))
    
    def _get_elem_fails_SVsens(self, delem_fails, helem_vec):
        """generic helper method to compute dfail/du given the df/dfail backproped partials
           this allows this method to be modular for one overall KSfail (SNOPT) vs KSfail by component (INK)"""
    
        # pre-compute hessian data
        hess_basis = np.zeros((12, 3))
        for ibasis in range(12):
            for idof in range(3):
                hess_basis[ibasis, idof] = get_hessians(ibasis, xi=0.0, eta=0.0, xscale=self.dx, yscale=self.dy)[idof]
        Dvec = self.E * helem_vec**3 / 12.0 / (1 - self.nu**2)
        du_global = _get_elem_fails_SVsens_numba(
            self.num_dof, self.num_elements, Dvec, self.dof_conn, self.u,
            hess_basis, self.nu, helem_vec, self.ys, delem_fails)
            
        return du_global
    
    def _compute_psi_Ku_prod(self, hcomp_vec):
        # update kmat, compute psi^T * fint = psi^T * Ku
        # u and psi optional inputs only for full space case
        self._compute_mat_vec(hcomp_vec)
        resid = self.Kmat.dot(self.u)
        return np.dot(self.psis, resid)
    
    def _compute_adj_dRdx(self, hcomp_vec):
        """compute the adj resid product psi^T dR/dx (partial deriv on R)"""
        helem_vec = self.get_helem_vec(hcomp_vec)
        
        Dvec = self.E * helem_vec**3 / 12.0 / (1 - self.nu**2)
        # code in _numba.py
        dthick_elem = _compute_adj_dRdx_numba(self.num_elements, self.Kelem_nom, self.dof_conn,
                                self.psis, self.u, Dvec, helem_vec, self.bcs)

        return np.array(self.helem_to_hcomp_vec(dthick_elem))
    
    def get_mass(self, hcomp_vec):
        helem_vec = self.get_helem_vec(hcomp_vec)
        # copy states out
        a = self.a; b = self.b; rho = self.rho

        # compute mass
        mass_vec = [a * b * helem_vec[ielem] * rho for ielem in range(self.num_elements)]
        mass = sum(mass_vec)
        return mass
    
    def get_mass_gradient(self):
        # copy states out
        a = self.a; b = self.b; rho = self.rho
        dmass = np.array([rho * a * b] * self.num_elements)
        dmass_red = self.helem_to_hcomp_vec(dmass)
        return dmass_red
    
    def _ks_func(self, vec):
        """helper generic ks func on a vector"""
        rhoKS = self.rho_KS
        true_max = np.max(vec)
        ks_max = true_max + 1.0 / rhoKS * np.log(np.sum(np.exp(rhoKS * (vec - true_max))))
        return ks_max
    
    def _ks_grad(self, vec):
        """helper generic derivative through the generic ks func"""
        rhoKS = self.rho_KS
        true_max = np.max(vec)

        dvec = np.exp(rhoKS * (vec - true_max))
        dvec /= np.sum(dvec)
        return dvec
    
    """
    -------------------------------------------------------------------
        end of general utils for any method (like primal, adjoint, elem_fails)
    -------------------------------------------------------------------
    start of multigrid section
    -------------------------------------------------------------------
    """

    @property
    def num_fine_dof(self) -> int:
        """length of finer grid vecs in multigrid"""
        return 3 * (self.nxe * 2 + 1)**2

    def to_fine(self, coarse_vec):
        """go to finer mesh than this one (this is coarse mesh)"""
        return self._coarse_fine_helper(coarse_vec, fine_out=True)
    
    def to_coarse(self, fine_vec):
        """go from finer mesh to this one (this is coarse mesh)"""
        return self._coarse_fine_helper(fine_vec, fine_out=False)

    def _coarse_fine_helper(self, in_vec, fine_out:bool=True):
        """go to a finer mesh than this one (so coarse is this mesh size).."""
        assert(self.nxe == self.nye) # TBD only do square grid right now easier

        in_vec = in_vec.copy()

        if fine_out:
            out_vec = np.zeros(self.num_fine_dof)
        else:
            out_vec = np.zeros(self.num_dof)
        nxe_fine = self.nxe * 2
        nx_fine = nxe_fine + 1
        nx_coarse = self.nxe + 1

        # need to rescale by mesh size change first so that the interp goes well
        # if fine_out:
        #     # r = 4.0
        #     r = 3.0
        #     # r = 2.0
        #     # r = 0.5
        #     in_vec[1::3] *= r
        #     in_vec[2::3] *= r

        nnodes_fine = self.num_fine_dof // 3

        for inode_fine in range(nnodes_fine):
            ix_fine, iy_fine = inode_fine % nx_fine, inode_fine // nx_fine
            ix_coarse, iy_coarse = ix_fine // 2, iy_fine // 2
            inode_coarse = nx_coarse * iy_coarse + ix_coarse

            for idof in range(3): # each DOF
                idof_fine = 3 * inode_fine + idof
                idof_coarse = 3 * inode_coarse + idof
                if fine_out:
                    idof_in, idof_out = idof_coarse, idof_fine
                else:
                    idof_in, idof_out = idof_fine, idof_coarse
            
                if ix_fine % 2 == 0 and iy_fine % 2 == 0: # matches node of coarse grid
                    out_vec[idof_out] += in_vec[idof_in]
                    # weights_vec[idof_coarse] += 1.0

                elif ix_fine % 2 == 1 and iy_fine % 2 == 1:
                    xi, eta = 0.0, 0.0

                    # fine node in the middle of an element
                    ixe_coarse, iye_coarse = ix_coarse, iy_coarse
                    ielem_coarse = self.nxe * iye_coarse + ixe_coarse
                    local_conn = self.dof_conn[ielem_coarse]
                    for ibasis in range(12):
                        if fine_out:
                            idof_in = local_conn[ibasis]
                        else:
                            idof_out = local_conn[ibasis]
                        
                        if idof == 0:
                            coeff = hermite_cubic_2d(ibasis, xi, eta)
                        elif idof == 1:
                            coeff = get_gradient(ibasis, xi, eta, xscale=1.0, yscale=1.0)[0] # no xscale (since w_x DOF is actually w_xi in hermite cubic)
                        else: # idof == 2
                            coeff = get_gradient(ibasis, xi, eta, xscale=1.0, yscale=1.0)[1]

                        out_vec[idof_out] += in_vec[idof_in] * coeff
                    
                elif ix_fine % 2 == 0:
                    # on element x boundary (two elements) but middle in y edge of element
                    eta = 0.0
                    for i, ixe_coarse in enumerate([ix_coarse-1, ix_coarse]):
                        xi = 1.0 if i == 0 else -1.0 # right or left side

                        if ixe_coarse < 0:
                            ixe_coarse, xi = 0, -1.0
                        elif ixe_coarse > self.nxe - 1:
                            ixe_coarse, xi = self.nxe - 1, 1.0

                        iye_coarse = iy_coarse
                        ielem_coarse = self.nxe * iye_coarse + ixe_coarse
                        local_conn = self.dof_conn[ielem_coarse]
                        for ibasis in range(12):
                            if fine_out:
                                idof_in = local_conn[ibasis]
                            else:
                                idof_out = local_conn[ibasis]
                            
                            if idof == 0:
                                coeff = hermite_cubic_2d(ibasis, xi, eta)
                            elif idof == 1:
                                coeff = get_gradient(ibasis, xi, eta, xscale=1.0, yscale=1.0)[0] # no xscale (since w_x DOF is actually w_xi in hermite cubic)
                            else: # idof == 2
                                coeff = get_gradient(ibasis, xi, eta, xscale=1.0, yscale=1.0)[1]

                            out_vec[idof_out] += in_vec[idof_in] * coeff * 0.5 # since two elements considered

                else: # need to interp in this element
                    # on element y boundary (two elements) but middle in x edge of element
                    xi = 0.0
                    for i, iye_coarse in enumerate([iy_coarse-1, iy_coarse]):
                        eta = 1.0 if i == 0 else -1.0 # top or bottom of element

                        if iye_coarse < 0:
                            iye_coarse, eta = 0, -1.0
                        elif iye_coarse > self.nxe - 1:
                            iye_coarse, eta = self.nxe - 1, 1.0

                        ixe_coarse = ix_coarse
                        ielem_coarse = self.nxe * iye_coarse + ixe_coarse
                        local_conn = self.dof_conn[ielem_coarse]
                        for ibasis in range(12):
                            if fine_out:
                                idof_in = local_conn[ibasis]
                            else:
                                idof_out = local_conn[ibasis]
                            
                            if idof == 0:
                                coeff = hermite_cubic_2d(ibasis, xi, eta)
                            elif idof == 1:
                                coeff = get_gradient(ibasis, xi, eta, xscale=1.0, yscale=1.0)[0] # no xscale (since w_x DOF is actually w_xi in hermite cubic)
                            else: # idof == 2
                                coeff = get_gradient(ibasis, xi, eta, xscale=1.0, yscale=1.0)[1]

                            out_vec[idof_out] += in_vec[idof_in] * coeff * 0.5 # since two elements considered

        # enforce fine bcs
        if fine_out:
            out_vec[self.fine_bcs] = 0.0
        else:
            out_vec[self.bcs] = 0.0

        # rescale rot DOF since they are w_xi and w_eta and scale by changed mesh size with hermite cubic here
        # this is something particular only to hermite cubic (not shells where all DOF are physical quantities)
        # to coarse works well, but to fine isn't scaling that well?

        if not fine_out: # coarsen
            out_vec *= 0.25 # cause accumulates from 4 values going down


        # if fine_out: # means we are increasing mesh size, scale down by 2x the rot DOF
        #     out_vec[1::3] *= 0.5
        #     # out_vec[2::3] *= 0.5
        # else: # decreasing mesh size, scale up the ROT dof
        #     out_vec[1::3] *= 2.0
        #     # out_vec[2::3] *= 2.0

        return out_vec


    """
    -------------------------------------------------------------------
        end of multigrid section
    -------------------------------------------------------------------
    start of SNOPT KS-aggregation section, uses one overall ksfail constraint (not by component)
    -------------------------------------------------------------------
    """
    
    def get_ks_fail(self, hcomp_vec):
        """KS-aggregate failure index of the whole structure for SNOPT, ks-aggreg approach"""
        helem_vec = self.get_helem_vec(hcomp_vec)
        elem_fails = self._get_elem_fails(helem_vec)
        ks_fail = self._ks_func(elem_fails)
        return ks_fail
    
    def _get_dkstot_du(self, hcomp_vec):
        """SV partial derivs for KS-aggregate fail index of whole structure"""
        helem_vec = self.get_helem_vec(hcomp_vec)
        elem_fails = self._get_elem_fails(helem_vec)
        dfails = self._ks_grad(elem_fails)
        du_global = self._get_elem_fails_SVsens(dfails, helem_vec)
        return du_global
    
    def _get_dkstot_dx(self, hcomp_vec):
        """DV partial derivatives for KSfail of whole structure, for SNOPT KS-aggreg"""        
        # backprop from scalar stress to stress vec 
        helem_vec = self.get_helem_vec(hcomp_vec)
        elem_fails = self._get_elem_fails(helem_vec)
        dfails = self._ks_grad(elem_fails)
        dhcomp_vec = self._get_elem_fails_DVsens(dfails, helem_vec, elem_fails)
        return dhcomp_vec

    def get_functions(self, hcomp_vec):
        """get functions for SNOPT with one aggreg ksfail"""
        mass = self.get_mass(hcomp_vec)
        ksfail = self.get_ks_fail(hcomp_vec)
        return np.array([mass, ksfail])
    
    def get_function_gradients(self, hcomp_vec):
        """get function gradients for SNOPT with one aggreg ksfail"""
        dmass = self.get_mass_gradient()

        # TODO : check shapes here..
        dks_fail = self._get_dkstot_dx(hcomp_vec)
        dks_fail += self._compute_adj_dRdx(hcomp_vec)

        return np.array([dmass, dks_fail])
    
    """
    ------------------------------------------------
    FSD methods here
    ------------------------------------------------
    """
    def _get_comp_true_fails(self, elem_fails):
        """for FSD method (true max in each component of fails, no KS) """
        # can be sensitive to stress-singularities with this approach
        comp_fails = np.zeros((self.ncomp,), dtype=elem_fails.dtype)

        for ielem in range(self.num_elements):
            icomp = self.elem_comp_map[ielem]
            comp_fails[icomp] = np.max([comp_fails[icomp], elem_fails[ielem]])
        # print(f"{stress_red=}")
        return comp_fails
    
    def get_comp_true_fails(self, hcomp_vec):
        """
        for FSD Method
        """
        helem_vec = self.get_helem_vec(hcomp_vec)
        elem_fails = self._get_elem_fails(helem_vec)
        true_comp_fails = self._get_comp_true_fails(elem_fails)
        return true_comp_fails
    
    """
        end of FSD methods
    ------------------------------------------------------
    start of plot utils section
    ------------------------------------------------------
    """

    @property
    def xvec(self) -> list:
        return [i*self.dx for i in range(self.num_nodes)]    
    
    def _plot_field_on_ax(self, field, ax, log_scale:bool=True, cmap='viridis', surface:bool=False, elem_to_node_convert:bool=True):
        """helper method to plot a scalar field as a contour"""
        x = self.xpts[0::3]
        y = self.xpts[1::3]

        X = np.reshape(x, (self.nnx, self.nny))
        Y = np.reshape(y, (self.nnx, self.nny))
        if elem_to_node_convert:
            H = np.reshape(field, (self.nxe, self.nye))
            H = self._elem_to_node_arr(H)
        else: 
            H = np.reshape(field, (self.nnx, self.nny))

        if log_scale: 
            H = np.log10(H)
        
        # 'seismic', 'twilight', 'magma', 'plasma', 'cividis'
        cmaps = ['viridis', 'turbo', 'RdBu_r']
        if isinstance(cmap, int): # allows you to put an int in
            cmap = cmaps[cmap]
        
        if surface:
            cf = ax.plot_surface(X, Y, H, cmap=cmap)
        else:
            cf = ax.contourf(X, Y, H, cmap=cmap, levels=100)
            self._plot_dv_grid(ax) # plot the DV boundaries on the plot

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return cf

    def plot_disp(self, filename:str=None, figsize=(10, 7), dpi:int=100, format_str="%.1e"):
        """plot the transverse disp w(x,y)"""

        plot_init()
        fig, ax = plt.subplots(figsize=figsize)
        cf = self._plot_field_on_ax(
            field=self.u[0::3],
            ax=ax,
            log_scale=False,
            cmap='turbo',
            surface=False,
            elem_to_node_convert=False,
        )
        cb = fig.colorbar(cf, ax=ax, format=FormatStrFormatter(format_str)) # can change string format as needed
        cb.set_label("w(x,y)")
        fig.set_dpi(dpi)

        if filename is None:
            plt.show() 
        else:
            plt.savefig(filename, dpi=dpi)    
            plt.close('all')

    def plot_vector(self, vec, idof:int=0, filename:str=None, figsize=(10, 7), dpi:int=100, format_str="%.1e"):
        """plot the transverse disp w(x,y)"""

        plot_init()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        cf = self._plot_field_on_ax(
            field=vec[idof::3], # plot only w DOF first
            ax=ax,
            log_scale=False,
            cmap='turbo',
            surface=True,
            elem_to_node_convert=False,
        )
        cb = fig.colorbar(cf, ax=ax, format=FormatStrFormatter(format_str)) # can change string format as needed
        cb.set_label("w(x,y)")
        fig.set_dpi(dpi)

        if filename is None:
            plt.show() 
        else:
            plt.savefig(filename, dpi=dpi)    
            plt.close('all')
    
    def plot_vectors(self, vec1, vec2, filename:str=None, figsize=(10, 7), dpi:int=100, format_str="%.1e"):
        """plot the transverse disp w(x,y)"""

        plot_init()
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # Create a 2x3 grid of subplots

        for i in range(2):
            for j in range(3):
                vec = vec1 if i == 0 else vec2
                ax = fig.add_subplot(2, 3, i*3 + j + 1, projection='3d')

                cf = self._plot_field_on_ax(
                    field=vec[j::3], # plot only w DOF first
                    ax=ax,
                    log_scale=False,
                    cmap='turbo',
                    surface=True,
                    elem_to_node_convert=False,
                )
                cb = fig.colorbar(cf, ax=ax, format=FormatStrFormatter(format_str)) # can change string format as needed
                # cb.set_label("w(x,y)")
        fig.set_dpi(dpi)

        if filename is None:
            plt.show() 
        else:
            plt.savefig(filename, dpi=dpi)    
            plt.close('all')

    def _elem_to_node_arr(self, arr):
        """convert """

        # strategy #2
        from scipy.interpolate import RegularGridInterpolator
        xorig = np.linspace(-self.a/2, self.a/2, self.nxe)
        yorig = np.linspace(-self.b/2, self.b/2, self.nye)
        # print(f"{xorig.shape=} {yorig.shape=} {arr.shape=}")
        # 'linear', 'cubic'
        interp_func = RegularGridInterpolator((xorig, yorig), arr, method='linear')

        xnew = np.linspace(-self.a/2, self.a/2, self.nnx)
        ynew = np.linspace(-self.b/2, self.b/2, self.nny)
        grid_points = np.array(np.meshgrid(xnew, ynew)).T.reshape(-1, 2)
        arr_v2_vec = interp_func(grid_points)
        arr_v2 = arr_v2_vec.reshape(self.nnx, self.nny)
        # print(f'{arr_v2.shape=}')

        return arr_v2
    
    def _plot_dv_grid(self, ax):
        """plot black boundaries for each DV region"""
        grid_rows, grid_cols = self.nxh, self.nyh # the actual DV regions
        row_step = self.a / grid_rows
        col_step = self.b / grid_cols

        for i in range(grid_rows):
            for j in range(grid_cols):
                ax.add_patch(plt.Rectangle((-self.a/2 + j * col_step, -self.b/2 + i * row_step), col_step, row_step, 
                                        edgecolor='white', facecolor='none', linewidth=1))

    """
    ------------------------------------------------------
        end of plot utils section
    ------------------------------------------------------
    """

    def finite_diff_test(self, hcomp_vec, h:float=1e-6):
        # central finite diff test for all func grads used in regular KS-aggregated opt (one overall KS failure index like in SNOPT)
        p = np.random.rand(self.ncomp)
        hcomp_vec = np.array(hcomp_vec)

        # adjoint value
        self.solve_forward(hcomp_vec)
        self.solve_adjoint(hcomp_vec)
        # funcs = self.get_functions(hcomp_vec)
        func_grads = self.get_function_gradients(hcomp_vec)
        
        nfunc = 2
        adj_prods = [
            np.dot(func_grads[ifunc,:], p) for ifunc in range(nfunc)
        ]

        # FD [rpdict]
        hcomp_vec2 = hcomp_vec + h * p
        self.solve_forward(hcomp_vec2)
        funcs2 = self.get_functions(hcomp_vec2)

        hcomp_vecn1 = hcomp_vec - h * p
        self.solve_forward(hcomp_vecn1)
        funcsn1 = self.get_functions(hcomp_vecn1)
        fd_prods = [
            (funcs2[i] - funcsn1[i]) / 2 / h for i in range(len(funcs2))
        ]

        print(f"{adj_prods=}")
        print(f"{fd_prods=}")