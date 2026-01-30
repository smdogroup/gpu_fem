import numpy as np
from .basis import second_order_quadrature

# based on Oesterle paper, "A shear deformable, rotation-free isogeometric shell formulation"
# https://www.sciencedirect.com/science/article/pii/S004578251630202X


class HierarchicIsogeometricDispElement9:
    """hierarchic displacement element HIGD with 2nd order IGA basis to allow 2nd derivatives in weak form"""
    def __init__(self, reduced_integrated:bool=False):              
        self.dof_per_node = 2
        self.nodes_per_elem = 9
        self.ndof = 18
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

        for ipt in range(9):
            ii, jj = ipt % 3, ipt // 3
            xi = pts[ii]; eta = pts[jj]
            wt = wts[ii] * wts[jj]

            # get J from xpts


            # get xi, eta derivs with basis


            # TODO : bending strains
            # k11 = -(wb + ws2)_{,11}
            # k22 = -(wb + ws1)_{,22}
            # 2*k12 = -(2*wb + ws1 + ws2)_{,12}

            # TODO : trv shear strains
            # 2*e13 = ws1,1
            # 2*e23 = ws2,2

    def get_felem(
        self,
        mag:float,
        elem_xpts:np.ndarray,
        left_bndry:bool,
        right_bndry:bool,
        bot_bndry:bool,
        top_bndry:bool,
    ):
        # not reduced integrated
        pts, wts = second_order_quadrature()

    def _build_restr_matrix(self, nxe_c:int):
        pass # 2d Kronecker product of 1d version see 1d higd elem

    def prolongate(self, coarse_soln:np.ndarray):
        pass

    def restrict_defect(self, fine_defect:np.ndarray):
        pass