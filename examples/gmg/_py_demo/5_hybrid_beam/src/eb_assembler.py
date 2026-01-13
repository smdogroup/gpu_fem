__all__ = ["EBAssembler"]

import numpy as np
import scipy as sp
from ._eb_elem import *

class EBAssembler:
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

        self.dof_per_node = 2 # need this for multigrid

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

    @property
    def dof_conn(self):
        return [[2 * ix+_ for _ in range(4)] for ix in range(self.nxe)]

    def _compute_mat_vec(self, helem_vec):
        # copy states out
        E = self.E; b = self.b; rho = self.rho; nxe = self.nxe; L = self.L; rho_KS = self.rho_KS

        self.data = np.zeros((self.nnzb, 2, 2), dtype=helem_vec.dtype)

        # define the element loads here
        xvec = [(ielem+0.5) * self.dx for ielem in range(self.num_elements)]
        qvec = [self.qmag * self.load_fcn(xval) for xval in xvec]

        # compute Kelem without EI scaling
        Kelem_nom = get_kelem(self.xscale / 2.0)
        felem_nom = get_felem(self.xscale / 2.0)

        num_dof = 2 * self.num_nodes
        if self._dense:
            Kmat = np.zeros((num_dof, num_dof), dtype=helem_vec.dtype)
        # want to start as sparse matrix now
        else:
            Kmat = None

        force = np.zeros((num_dof,))

        for ielem in range(self.num_elements): 
            local_conn = np.array(self.dof_conn[ielem])

            EI = E * b * helem_vec[ielem]**3 / 12.0
            if self._dense:
                np.add.at(Kmat, (local_conn[:,None], local_conn[None,:]), EI * Kelem_nom)
            else: # sparse
                # add into sparse data 
                Kelem = EI * Kelem_nom

                # if ielem == 0:
                #     plt.imshow(Kelem)
                #     plt.show()
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
            # q *= 0.5 # to normalize from local to global basis functions
            # felem_nom[1] = 0.0
            # felem_nom[3] = 0.0
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

        return self.u
    
    @property
    def xvec(self) -> list:
        return [i*self.dx for i in range(self.num_nodes)]

    def plot_disp(self, idof:int=0):
        xvec = self.xvec
        # print(f"{self.u=}")
        w = self.u[idof::2]
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
        fine_xscale = coarse_xscale * 0.5

        # allocate final array
        fine_disp = np.zeros(ndof_fine)
        fine_weights = np.zeros(ndof_fine) # for global partition of unity normalization

        # loop through coarse elements
        for ielem_c in range(nelems_coarse):

            # get the coarse element DOF
            coarse_elem_dof = np.array([2 * _node + _dof for _node in [ielem_c, ielem_c + 1] for _dof in range(2)])
            # print(F"{ielem_c=} {coarse_elem_dof=}")
            coarse_elem_disps = coarse_disp[coarse_elem_dof]
            
            # interpolate the w DOF first using FEA basis
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                xi = -1.0 + 1.0 * i
                w = interp_hermite_disp(xi, coarse_elem_disps, fine_xscale=fine_xscale)
                # even though we're interpolating from fine to coarse elements with hermite cubic
                # the in-element dw/dxi interp better if scaled down to fine dw/dxi size (hard to explain but works for hermtie cubic and hugely improves conv back to omega = 1 from omega = 0.25 on two grid case)
                
                th = interp_lagrange_rotation(xi, coarse_elem_disps)
                # th = interp_hermite_rotation(xi, coarse_elem_disps, coarse_xscale=coarse_xscale)

                # print(f"{ielem_c=} interp to {inode_f=} with {xi=} and {w=}")

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
        fine_xscale = self.xscale * 0.5

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
                coarse_out = interp_hermite_disp_transpose(xi, nodal_in[0], fine_xscale=fine_xscale)

                # lagrange interp of rotations actually works better interestingly (element allows small rotation in each element to min energy, so weird theta interp with hermite grad)
                coarse_out += interp_lagrange_rotation_transpose(xi, nodal_in[1])
                # coarse_out += interp_hermite_rotation_transpose(xi, nodal_in[1], coarse_xscale=self.xscale)
                # coarse_out[np.array([1,3])] = 0.0 # temp debug

                coarse_defect[coarse_elem_dof] += coarse_out

            
        # apply bcs.. to coarse defect also
        coarse_defect[0] = 0.0
        coarse_defect[-2] = 0.0

        return coarse_defect
