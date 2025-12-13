#include "linalg/linalg.h"
#include "base/utils.h"

/*
Try the same problem but I scale the matrix by 1e8 kind of like kmat, see if needs preconditioner or something
*/

int main() {
    using Mat = BsrMat<DeviceVec<double>>;

    // create my own BsrData and BsrMat object
    int orig_rowPtr[] = {0, 3, 6, 8, 10};
    int orig_colPtr[] = {0, 1, 3, 0, 1, 2, 1, 2, 0, 3};
    int nnodes = 4;
    int nnzb = 10;
    int block_dim = 1;
    double fillin = 10.0;

    bool print = true;
    BsrData bsr_data = BsrData(nnodes, block_dim, nnzb, orig_rowPtr, orig_colPtr);
    bsr_data.symbolic_factorization(fillin, print);
    BsrData d_bsr_data = bsr_data.createDeviceBsrData();

    // nz from kernel matrix k(x_i, x_j) = x_i * x_j = (i+1)*(j+1)
    // plus diagol 2 * I matrix to make pos definite

    // added fillin zeros also
    double values[] = {3, 2, 4, 2, 6, 6, 0, 6, 11, 0, 4,  0, 0, 18};
    int nnzb_fillin = 14;
    HostVec<double> h_values(nnzb_fillin, values);
    h_values.scale(1e8);
    auto d_values = h_values.createDeviceVec();

    Mat kmat = BsrMat(d_bsr_data, d_values);

    double rhs[] = {-1, -2, 3, 4};
    HostVec<double> h_rhs(4, rhs);
    auto d_rhs = h_rhs.createDeviceVec();

    double true_soln[] = {1.20689655, -2.2183908, 1.48275862, -0.04597701};
    HostVec<double> h_true_soln(4, true_soln);
    h_true_soln.scale(1e-8);
    HostVec<double> temp(4), soln(4);
    auto d_temp = temp.createDeviceVec();
    auto d_soln = soln.createDeviceVec();

    CUSPARSE::direct_LU_solve_old<double>(kmat, d_rhs, d_soln);
    auto max_resid = CUSPARSE::get_resid<double>(kmat, d_rhs, d_soln);

    auto h_soln = d_soln.createHostVec();
    printf("h_soln: ");
    printVec<double>(4, h_soln.getPtr());

    printf("h_true_soln: ");
    printVec<double>(4, h_true_soln.getPtr());

    getVecRelError<double>(h_soln, h_true_soln);
    printf("cusparse linear solve max error: %.8e\n", max_resid);

    return 0;
};