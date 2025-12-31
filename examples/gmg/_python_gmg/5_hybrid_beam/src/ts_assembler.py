__all__ = ["TimoshenkoAssembler"]

import numpy as np
import scipy as sp
from ._ts_elem import *

class TimoshenkoAssembler:
    def __init__(self, nxe:int, nxh:int, E:float, b:float, L:float, rho:float, 
                 qmag:float, ys:float, rho_KS:float, dense:bool=False, load_fcn=None, use_nondim:bool=False):
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
        self.use_nondim = use_nondim

        self.dof_per_node = 2 # need this for multigrid

        self.Kmat = None
        self.force = None
        self.u = None
        # adjoint only required for stress function
        self.psis = None

        self.red_int = True # can change this in python if you really need to

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
        qvec = [self.qmag * self.load_fcn(xval) for xval in xvec]

        # compute Kelem without EI scaling
        EI = E * b * helem_vec[0]**3 / 12.0
        nu=0.3
        G = E / 2.0 / (1 + nu)
        A = b * helem_vec[0]
        GA = G * A
        # kGA = 5.0 / 6.0 * G * A
        # print(f"{EI=:.2e} {helem_vec[0]=:.2e} {kGA=:.2e}")
        # red_int = True
        # red_int = False

        # nondim transform
        self.gam = gam = np.sqrt(EI / (5.0 / 6.0) / GA)

        # temp debug
        # EI = 0.0

        Kelem_nom = get_kelem(J=self.dx / 2.0, EI=EI, GA=GA, use_reduced_integration_for_shear=self.red_int, use_nondim=self.use_nondim)
        felem_nom = get_felem(self.dx / 2.0)
        if self.use_nondim:
            felem_nom *= gam

        # Kelem_nom = get_kelem(J=self.dx, EI=EI, GA=GA, use_reduced_integration_for_shear=red_int)
        # felem_nom = get_felem(self.dx)

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
            # q *= 0.5 # since each node has two contributions adding to it (normalizing from local to global basis functions basically)
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
    
    def get_helem_vec(self, hred):
        helem_vec = np.zeros((self.num_elements,), dtype=hred.dtype)
        for ix in range(self.nxe):
            ired = ix // self.nnxh
            helem_vec[ix] = hred[ired]
        return helem_vec

    def solve_forward(self, hred):
        # now solve the linear system
        helem_vec = self.get_helem_vec(hred)
        self._compute_mat_vec(helem_vec)

        if self._dense:
            self.u = np.linalg.solve(self.Kmat, self.force)
        else:
            # print(f"{self.Kmat.toarray()=}")
            self.u = sp.sparse.linalg.spsolve(self.Kmat, self.force)

        if self.use_nondim:
            self.u[0::2] *= self.gam

        return self.u

    @property
    def dof_conn(self):
        return [[2 * ix+_ for _ in range(4)] for ix in range(self.nxe)]

    @property
    def xvec(self) -> list:
        return [i*self.dx for i in range(self.num_nodes)]

    def plot_disp(self):
        xvec = self.xvec
        # print(f"{self.u=}")
        w = self.u[0::2]
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(xvec, w)
        plt.plot(xvec, np.zeros((self.num_nodes,)), "k--")
        plt.xlabel("x")
        plt.ylabel("w(x)")
        plt.show()       


    # multgrid code
    # ---------------

    def prolongate(self, coarse_disp):
        # assume coarse disp is for half as many elements

        # coarse size
        ndof_coarse = coarse_disp.shape[0]
        nnodes_coarse = ndof_coarse // 2
        nelems_coarse = nnodes_coarse - 1
        coarse_xscale = self.L / nelems_coarse

        # fine size
        nelems_fine = 2 * nelems_coarse 
        assert(nelems_fine == self.num_elements)
        nnodes_fine = nelems_fine + 1
        ndof_fine = 2 * nnodes_fine

        # allocate final array
        fine_disp = np.zeros(ndof_fine)
        fine_weights = np.zeros(ndof_fine) # for global partition of unity normalization

        # loop through coarse elements
        for ielem_c in range(nelems_coarse):

            # get the coarse element DOF
            coarse_elem_dof = np.array([2 * _node + _dof for _node in [ielem_c, ielem_c + 1] for _dof in range(2)])
            coarse_elem_disps = coarse_disp[coarse_elem_dof]
            
            # interpolate the w DOF first using FEA basis
            # start_inode_c = ielem_c
            # start_inode_f = 2 * ielem_c
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                xi = -1.0 + 1.0 * i

                w, th = interp_lagrange(xi, coarse_elem_disps)

                fine_disp[2 * inode_f] += w
                fine_disp[2 * inode_f + 1] += th

                fine_weights[2 * inode_f] += 1.0
                fine_weights[2 * inode_f + 1] += 1.0

        # normalize by fine weights now
        fine_disp /= fine_weights

        # apply bcs..
        fine_disp[0] = 0.0
        fine_disp[-2] = 0.0

        return fine_disp

    def restrict_defect(self, fine_defect):
        # from fine defect to this assembler as coarse defect

        # fine size
        ndof_fine = fine_defect.shape[0]
        nnodes_fine = ndof_fine // 2
        nelems_fine = nnodes_fine - 1

        # coarse size
        nelems_coarse = nelems_fine // 2
        assert(nelems_coarse == self.num_elements)
        nnodes_coarse = nelems_coarse + 1
        ndof_coarse = 2 * nnodes_coarse

        # allocate final array
        coarse_defect = np.zeros(ndof_coarse)
        fine_weights = np.zeros(ndof_fine) # for global partition of unity normalization

        # compute first the fine weights (I do this better way on GPU).. and other codes, this is lightweight implementation, don't care here
        for ielem_c in range(nelems_coarse):
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                fine_weights[2 * inode_f] += 1.0
                fine_weights[2 * inode_f + 1] += 1.0

        # begin by apply bcs to fine defect in (usually not necessary)
        fine_defect[0] = 0.0
        fine_defect[-2] = 0.0

        # loop through coarse elements to compute restricted defect
        for ielem_c in range(nelems_coarse):
            coarse_elem_dof = np.array([2 * _node + _dof for _node in [ielem_c, ielem_c + 1] for _dof in range(2)])
            
            # interpolate the w DOF first using FEA basis
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                xi = -1.0 + 1.0 * i
                nodal_in = fine_defect[2 * inode_f : (2 * inode_f + 2)] / fine_weights[2 * inode_f : (2 * inode_f + 2)]
                coarse_out = interp_lagrange_transpose(xi, nodal_in)
                coarse_defect[coarse_elem_dof] += coarse_out

            
        # apply bcs.. to coarse defect also
        coarse_defect[0] = 0.0
        coarse_defect[-2] = 0.0

        return coarse_defect