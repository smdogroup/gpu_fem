#include "base/utils.h"
#include "linalg/linalg.h"

/*
  debug solve on the CPU here
 */

int main() {
    using Mat = BsrMat<HostVec<double>>;

    // create my own BsrData and BsrMat object
    int orig_rowPtr[] = {0, 3, 6, 8, 10};
    int orig_colPtr[] = {0, 1, 3, 0, 1, 2, 1, 2, 0, 3};
    int nnodes = 4;
    int nnzb = 10;
    int block_dim = 1;

    bool print = true;
    BsrData bsr_data =
        BsrData(nnodes, block_dim, nnzb, orig_rowPtr, orig_colPtr, print);

    // nz from kernel matrix k(x_i, x_j) = x_i * x_j = (i+1)*(j+1)
    // plus diagol 2 * I matrix to make pos definite

    // added fillin zeros also
    double _values[] = {3, 2, 4, 2, 6, 6, 6, 11, 4, 18};
    int nnzb_fillin = 10;
    HostVec<double> values(nnzb_fillin, _values);

    Mat kmat = BsrMat(bsr_data, values);

    double _rhs[] = {-1, -2, 3, 4};
    HostVec<double> rhs(4, _rhs);

    double _true_soln[] = {1.20689655, -2.2183908, 1.48275862, -0.04597701};
    HostVec<double> true_soln(4, _true_soln);
    HostVec<double> temp(4), soln(4);

    EIGEN::iterative_CG_solve<double>(kmat, rhs, soln);
    // auto max_resid = EIGEN::get_resid<double>(kmat, rhs, soln);

    printf("soln: ");
    printVec<double>(4, soln.getPtr());

    printf("true_soln: ");
    printVec<double>(4, true_soln.getPtr());

    getVecRelError<double>(soln, true_soln);
    // printf("eigen linear solve max error: %.8e\n", max_resid);

    return 0;
};