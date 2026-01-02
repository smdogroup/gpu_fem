from _src import *
import numpy as np
import unittest

np.random.seed(1234)

class TestANSShell(unittest.TestCase):
    h = 0.134
    STRAIN_STRS = ['exx', 'eyy', 'exy', 'kxx', 'kyy', 'kxy', 'gxz', 'gyz']

    def _get_quadpt_strains(self, xi, eta, elem_vars):
        """get shell strains at this quadpt"""
        xpts = np.zeros(27)
        for node in range(9):
            inode, jnode = node % 3, node // 3
            xpts[3*node], xpts[3*node+1], xpts[3*node+2] = self.h * inode, self.h * jnode, 0.0

        shell_xi_axis = np.array([1.0, 0.0, 0.0])
        _strains = get_quadpt_strains(shell_xi_axis, xi, eta, xpts, elem_vars)
        return _strains
    
    def _get_quadpt_strains_transpose(self, xi, eta, dstrains):
        """get shell strains at this quadpt"""
        xpts = np.zeros(27)
        for node in range(9):
            inode, jnode = node % 3, node // 3
            xpts[3*node], xpts[3*node+1], xpts[3*node+2] = self.h * inode, self.h * jnode, 0.0

        shell_xi_axis = np.array([1.0, 0.0, 0.0])
        _dvars = get_strains_transpose(shell_xi_axis, xi, eta, xpts, dstrains)
        return _dvars

    @staticmethod
    def _rel_err(truth, pred):
        if truth != 0.0:
            return abs((truth - pred) / truth)
        else: 
            return 0.0

    def _strain_verification_helper(self, strains_in:np.ndarray):

        # get input strains (const)
        exx, eyy, exy = strains_in[0], strains_in[1], strains_in[2]
        kxx, kyy, kxy = strains_in[3], strains_in[4], strains_in[5]
        gxz, gyz = strains_in[6], strains_in[7]

        _kxy = kxy * 0.5 # half the strain here?

        # random curvatures for w so that alpha, beta enforced
        w_xx, w_xy, w_yy = np.random.rand(), np.random.rand(), np.random.rand()

        # only introduce w in bending case
        if gxz != 0.0 or gyz != 0.0:
            w_xx *= 0.0
            w_xy *= 0.0
            w_yy *= 0.0

        # bending strain conv from alpha,beta to thx, thy
        gxz_x, gxz_y = kxx + w_xx, _kxy + w_xy
        gyz_x, gyz_y = _kxy + w_xy, kyy + w_yy

        # NOTE : can't have both bending + trv shear together here => otherwise alpha, beta changing but constant?
        is_bending = kxx != 0.0 or kyy != 0.0 or kxy != 0.0
        is_trv_shear = gxz != 0.0 or gyz != 0.0
        assert(not(is_bending and is_trv_shear))

        elem_vars = np.zeros(45)
        for node in range(9):
            inode, jnode = node % 3, node // 3
            x, y = self.h * inode, self.h * jnode

            # u,v,w
            elem_vars[5*node] = x * exx + 0.5 * exy * y
            elem_vars[5*node+1] = y * eyy + 0.5 * exy * x
            elem_vars[5*node+2] = w_xx * 0.5 * x**2 + w_xy * x * y + w_yy * 0.5 * y**2

            # thx, thy
            elem_vars[5*node+3] = gxz + gxz_x * x + gxz_y * y
            elem_vars[5*node+4] = gyz + gyz_x * x + gyz_y * y
        
        xi, eta = np.random.rand(), np.random.rand()
        strains_out = self._get_quadpt_strains(xi, eta, elem_vars)

        rel_err = np.max(np.array([self._rel_err(strains_in[_i], strains_out[_i]) for _i in range(8)]))

        return strains_out, rel_err
    
    def _single_strain_helper(self, strain_ind, test_ind:int, name:str):
        strains_in = np.zeros(8)
        strains_in[strain_ind] = np.random.rand() 
        strains_out, rel_err = self._strain_verification_helper(strains_in)
        e_in, e_out = strains_in[strain_ind], strains_out[strain_ind]

        print(f"{test_ind} ANS shell, {name} strain test: {e_in=:.3e} {e_out=:.3e} => {rel_err=:.3e}", flush=True)
        return rel_err
    
    def test_all_strains(self):
        for ind in range(8):
            rel_err = self._single_strain_helper(strain_ind=ind, test_ind=ind, name=self.STRAIN_STRS[ind])
            assert(rel_err < 1e-4)
        return
    
    def test_strain_disp_transpose_compat(self):
        """compute B matrix and B^T matrix at a single quadpt and check compatibility (so that Kelem will be energy func still)"""
        xi, eta = np.random.rand(), np.random.rand()

        # form B matrix by mult all 45 ei by it..
        B = np.zeros((8, 45))
        for i in range(45):
            p_vars = np.zeros(45)
            p_vars[i] = 1.0
            p_strains = self._get_quadpt_strains(xi, eta, p_vars)
            B[:, i] = p_strains[:]
        BT = B.T

        max_rel_err = 0.0
        for i in range(8): # compare BT backprops..
            dstrains = np.zeros(8)
            dstrains[i] = 1.0

            BT_true = np.dot(BT, dstrains)
            BT_pred = self._get_quadpt_strains_transpose(xi, eta, dstrains)

            rel_err = np.linalg.norm(BT_true - BT_pred) / np.linalg.norm(BT_true)
            name = self.STRAIN_STRS[i]
            print(f"{i+1} {name} strain : {rel_err=}")
            if rel_err > 1e-4:
                print(f"\t{BT_true=} {BT_pred=}", flush=True)
            assert(rel_err < 1e-4)

        # B_BT = BT[:,:3] @ B[:3,:]
        # plt.imshow(B_BT)
        # plt.show()

        print(f"B vs B^T test passed with {max_rel_err=:.2e}")

        return max_rel_err


if __name__ == "__main__":
    unittest.main()