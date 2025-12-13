#include "linalg/linalg.h"
#include "base/utils.h"

/*
Try 4x4 block size here hand-crafted
*/

int main() {
    using Mat = BsrMat<DeviceVec<double>>;

    // create my own BsrData and BsrMat object
    int orig_rowPtr[] = {0, 3, 6, 8, 10};
    int orig_colPtr[] = {0, 1, 3, 0, 1, 2, 1, 2, 0, 3};
    int nnodes = 4;
    int nnzb = 10;
    int block_dim = 4;
    double fillin = 10.0;

    bool print = true;
    BsrData bsr_data = BsrData(nnodes, block_dim, nnzb, orig_rowPtr, orig_colPtr, fillin, print);
    BsrData d_bsr_data = bsr_data.createDeviceBsrData();

    // nz from kernel matrix k(x_i, x_j) = x_i * x_j = (i+1)*(j+1)
    // plus diagonal 2 * I matrix to make pos definite

    // added fillin zeros also
    int nnzb_fillin = 14;
    int nvals = 16 * nnzb_fillin;
    double *values = new double[nvals];
    int *rowPtr = bsr_data.rowPtr;
    int *colPtr = bsr_data.colPtr;

    // now allocate values in same way as python script 3_true_soln.py
    for (int block_row = 0; block_row < nnodes; block_row++) {
        for (int colPtr_ind = rowPtr[block_row]; colPtr_ind < rowPtr[block_row+1]; colPtr_ind++) {
            int block_col = colPtr[colPtr_ind];
            int val_ind = 16 * colPtr_ind;

            for (int inz = 0; inz < 16; inz++) {
                int local_row = inz / 4;
                int local_col = inz % 4;
                int global_row = 4 * block_row + local_row;
                int global_col = 4 * block_col + local_col;

                values[val_ind + inz] = (global_row + 1) * (global_col + 1);
                if (global_row == global_col) {
                    values[val_ind + inz] += 64.0;
                }

                // make sure fillin spots are zero (hack => since will put kernel matrix there too)
                if (block_row == 3 && block_col == 1) {
                    values[val_ind + inz] = 0.0;
                }
                if (block_row == 3 && block_col == 2) {
                    values[val_ind + inz] = 0.0;
                }
                if (block_row == 1 && block_col == 3) {
                    values[val_ind + inz] = 0.0;
                }
                if (block_row == 2 && block_col == 3) {
                    values[val_ind + inz] = 0.0;
                }

            }
        }
    }
    
    HostVec<double> h_values(16 * nnzb_fillin, values);
    auto d_values = h_values.createDeviceVec();

    Mat kmat = BsrMat(d_bsr_data, d_values);

    double *rhs = new double[16];
    for (int i = 0; i < 16; i++) {
        rhs[i] = i;
    }
    HostVec<double> h_rhs(16, rhs);
    auto d_rhs = h_rhs.createDeviceVec();

    double true_soln[] = {-0.07040604, -0.12518708, -0.17996812, -0.23474916,  0.10428516,
        0.12826719,  0.15224922,  0.17623126, -0.05286689, -0.05700487,
       -0.06114286, -0.06528085,  0.03731687,  0.04138933,  0.04546178,
        0.04953423};
    HostVec<double> h_true_soln(16, true_soln);
    // h_true_soln.scale(1e-8);
    HostVec<double> temp(16), soln(16);
    auto d_temp = temp.createDeviceVec();
    auto d_soln = soln.createDeviceVec();

    CUSPARSE::direct_LU_solve_old<double>(kmat, d_rhs, d_soln);
    auto max_resid = CUSPARSE::get_resid<double>(kmat, d_rhs, d_soln);

    auto h_soln = d_soln.createHostVec();
    printf("h_soln: ");
    printVec<double>(16, h_soln.getPtr());

    printf("h_true_soln: ");
    printVec<double>(16, h_true_soln.getPtr());

    getVecRelError<double>(h_soln, h_true_soln);
    printf("cusparse linear solve max error: %.8e\n", max_resid);

    return 0;
};