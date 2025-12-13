#include "linalg/linalg.h"
#include <fstream>
#include <iostream>
#include <vector>

void assemble_kmat(int nxe, int num_nodes, int block_dim, int num_bcs, int *bcs,
                   int *rowPtr, int *colPtr, double *kelem, double *&values) {
    int nx = nxe + 1;
    int ielem = -1;
    int block_dim2 = block_dim * block_dim;

    // printf("rowPtr: ");
    printVec<int>(num_nodes + 1, rowPtr);
    // printf("colPtr: ");
    int nnzb = rowPtr[num_nodes];
    printVec<int>(nnzb, colPtr);

    // printf("nx = %d, block_dim = %d, num_bcs = %d\n", nx, block_dim,
    // num_bcs);

    for (int iye = 0; iye < nxe; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            ielem++;
            int inode = nx * iye + ixe;
            printf("inode = %d\n", inode);
            // get the connectivity for this element
            int nodes[] = {inode, inode + 1, inode + nx + 1, inode + nx};

            // now we add kelem_dense into values array for global kmat
            for (int elem_block = 0; elem_block < 16; elem_block++) {
                int elem_block_row = elem_block / 4;
                int elem_block_col = elem_block % 4;
                // elem block row, col correspond to node level
                // map to global nodes or block locations
                int this_block_row = nodes[elem_block_row];
                int this_block_col = nodes[elem_block_col];

                // printf("elem_block_row %d, elem_block_col %d, block_row %d,
                // block_col %d\n", elem_block_row, elem_block_col,
                // this_block_row, this_block_col);

                // now need to find the location of this block in values array
                for (int block_row = 0; block_row < num_nodes; block_row++) {
                    for (int colPtr_ind = rowPtr[block_row];
                         colPtr_ind < rowPtr[block_row + 1]; colPtr_ind++) {
                        int block_col = colPtr[colPtr_ind];
                        int val_ind = block_dim2 * colPtr_ind;

                        // make sure this matches the right nodes spots from
                        // connectivity we're supposed to add into
                        if (block_row == this_block_row &&
                            block_col == this_block_col) {

                            // check block locations
                            // printf("ielem %d, block_row %d, block_col %d at
                            // location val_ind %d\n", ielem, block_row,
                            // block_col, colPtr_ind);

                            for (int inz = 0; inz < block_dim2; inz++) {
                                int local_row = inz / block_dim;
                                int local_col = inz % block_dim;

                                int global_row =
                                    block_dim * block_row + local_row;
                                int global_col =
                                    block_dim * block_col + local_col;

                                // check if bcs on this global_row (note this is
                                // not efficient code, just going for
                                // correctness here for debugging)
                                bool is_bc = false;
                                int bc;
                                for (int ibc = 0; ibc < num_bcs; ibc++) {
                                    bc = bcs[ibc];
                                    // does it make a difference in direct
                                    // solver if we zero out bc columns or not?
                                    // if (bc == global_row) { # try not doing
                                    // col here
                                    if (bc == global_row || bc == global_col) {
                                        is_bc = true;
                                        break;
                                    }
                                }

                                int elem_row =
                                    block_dim * elem_block_row + local_row;
                                int elem_col =
                                    block_dim * elem_block_col + local_col;

                                // printf("adding value of %.8e into spot
                                // values[%d]\n", kelem_dense[24 * elem_row +
                                // elem_col], val_ind + inz);

                                // printf("ielem %d adding value %.8e into grow
                                // %d, gcol %d, elem_block %d, erow %d, ecol %d,
                                // but is bc %d\n", ielem, kelem[36 * elem_block
                                // + inz], global_row, global_col, elem_block,
                                // elem_row, elem_col, (int)is_bc);
                                if (!is_bc) {
                                    // kelem is stored in dense fashion here
                                    // (not BSR)
                                    values[val_ind + inz] +=
                                        kelem[12 * elem_row + elem_col];
                                    // BSR style
                                    // values[val_ind + inz] += kelem[block_dim2
                                    // * elem_block + inz];
                                } else if (global_row == global_col) {
                                    values[val_ind + inz] = 1.0;
                                } else {
                                    values[val_ind + inz] = 0.0;
                                }

                                // if (is_bc) {
                                //     printf("bc %d, block_row %d, block_col
                                //     %d, "
                                //            "grow %d, gcol %d\n",
                                //            bc, block_row, block_col,
                                //            global_row, global_col);
                                // }
                            }
                        }
                    }
                }
            } // end of elem_block loop

        } // end of ixe loop
    }     // end of iye loop
}

template <typename T>
void write_to_csv(const T *array, size_t size, const std::string &filename) {
    std::ofstream out(filename);
    if (!out) {
        throw std::ios_base::failure("Failed to open file for writing");
    }
    for (size_t i = 0; i < size; ++i) {
        out << array[i];
        if (i != size - 1) {
            out << ",";
        }
    }
    out << "\n";
    out.close();
}

double *get_red_kelem_dense() {
    double *kelem_dense = new double[144];
    double red_kelem_dense[] = {
        1.30876e+08,  -4.67415e+06, -5.60897e+07, -1.30876e+08, 4.67415e+06,
        -5.60897e+07, 6.54380e+07,  -4.67415e+06, -4.20673e+07, -6.54380e+07,
        4.67415e+06,  -4.20673e+07, -4.67415e+06, 4.67517e+06,  -7.01109e+06,
        4.67415e+06,  2.33685e+06,  7.01123e+06,  4.67415e+06,  -2.33696e+06,
        -7.01123e+06, -4.67415e+06, -4.67506e+06, 7.01109e+06,  -5.60897e+07,
        -7.01109e+06, 4.20678e+07,  5.60897e+07,  -7.01123e+06, 1.40222e+07,
        -4.20673e+07, 7.01123e+06,  3.50562e+07,  4.20673e+07,  7.01109e+06,
        7.01084e+06,  -1.30876e+08, 4.67415e+06,  5.60897e+07,  1.30876e+08,
        -4.67415e+06, 5.60897e+07,  -6.54380e+07, 4.67415e+06,  4.20673e+07,
        6.54380e+07,  -4.67415e+06, 4.20673e+07,  4.67415e+06,  2.33685e+06,
        -7.01123e+06, -4.67415e+06, 4.67517e+06,  7.01109e+06,  -4.67415e+06,
        -4.67506e+06, -7.01109e+06, 4.67415e+06,  -2.33696e+06, 7.01123e+06,
        -5.60897e+07, 7.01123e+06,  1.40222e+07,  5.60897e+07,  7.01109e+06,
        4.20678e+07,  -4.20673e+07, -7.01109e+06, 7.01084e+06,  4.20673e+07,
        -7.01123e+06, 3.50562e+07,  6.54380e+07,  4.67415e+06,  -4.20673e+07,
        -6.54380e+07, -4.67415e+06, -4.20673e+07, 1.30876e+08,  4.67415e+06,
        -5.60897e+07, -1.30876e+08, -4.67415e+06, -5.60897e+07, -4.67415e+06,
        -2.33696e+06, 7.01123e+06,  4.67415e+06,  -4.67506e+06, -7.01109e+06,
        4.67415e+06,  4.67517e+06,  7.01109e+06,  -4.67415e+06, 2.33685e+06,
        -7.01123e+06, -4.20673e+07, -7.01123e+06, 3.50562e+07,  4.20673e+07,
        -7.01109e+06, 7.01084e+06,  -5.60897e+07, 7.01109e+06,  4.20678e+07,
        5.60897e+07,  7.01123e+06,  1.40222e+07,  -6.54380e+07, -4.67415e+06,
        4.20673e+07,  6.54380e+07,  4.67415e+06,  4.20673e+07,  -1.30876e+08,
        -4.67415e+06, 5.60897e+07,  1.30876e+08,  4.67415e+06,  5.60897e+07,
        4.67415e+06,  -4.67506e+06, 7.01109e+06,  -4.67415e+06, -2.33696e+06,
        -7.01123e+06, -4.67415e+06, 2.33685e+06,  7.01123e+06,  4.67415e+06,
        4.67517e+06,  -7.01109e+06, -4.20673e+07, 7.01109e+06,  7.01084e+06,
        4.20673e+07,  7.01123e+06,  3.50562e+07,  -5.60897e+07, -7.01123e+06,
        1.40222e+07,  5.60897e+07,  -7.01109e+06, 4.20678e+07};
    for (int i = 0; i < 144; i++) {
        kelem_dense[i] = red_kelem_dense[i];
    }
    return kelem_dense;
}

int main() {
    using Mat = BsrMat<HostVec<double>>;

    // 2x2 plate mesh => 3x3 nodes and 9 num_nodes, 4 elements
    int nxe = 2, nx = nxe + 1;
    int num_elements = nxe * nxe;
    int num_nodes = nx * nx;

    // row and colPtr from by hand matrix assembly
    int orig_rowPtr[] = {0, 4, 10, 14, 20, 29, 35, 39, 45, 49};
    int orig_colPtr[] = {
        0, 1, 3, 4,                // 4
        0, 1, 2, 3, 4, 5,          // 6
        1, 2, 4, 5,                // 4
        0, 1, 3, 4, 6, 7,          // 6
        0, 1, 2, 3, 4, 5, 6, 7, 8, // 9
        1, 2, 4, 5, 7, 8,          // 6
        3, 4, 6, 7,                // 4
        3, 4, 5, 6, 7, 8,          // 6
        4, 5, 7, 8                 // 4
    };
    int nnzb = 49; // = 4 * 4 + 6 * 4 + 1 * 9
    int block_dim = 3;
    int block_dim2 = block_dim * block_dim;
    int ndof = num_nodes * block_dim;

    // kelem for w, thx, thy dof only is 12x12 = 144 nonzeros (dense)
    double *red_kelem_dense = get_red_kelem_dense();

    // all nodes 0-8 have w = 0 constr except node 4 (dof 12) and node 0 has thx
    // = thy = 0 as well to ensure thx, thy PDEs are ill-posed
    int num_bcs = 10;
    int bcs[] = {0, 1, 2, 3, 6, 9, 15, 18, 21, 24};

    // do fillin with cholmod in suitesparse
    bool print = false;
    BsrData bsr_data =
        BsrData(num_nodes, block_dim, nnzb, orig_rowPtr, orig_colPtr, print);

    // rowPtr, colPtr should be unchanged since they were already filled in
    // before by cholmod / suitesparse
    int *rowPtr = bsr_data.rowPtr;
    int *colPtr = bsr_data.colPtr;

    // now assemble the matrix of the 4x4 elem or 5x5 nodal plate mesh myself
    // using kelem_dense into BSR data structure
    int nnzb2 = bsr_data.nnzb;
    int nvals = block_dim2 * nnzb2;
    // printf("nvals = %d\n", nvals);
    double *_values = new double[nvals];
    memset(_values, 0.0, nvals * sizeof(double));

    // let's assemble the matrix now in BSR format
    // convert it to dense format, print out to python and heatmap check
    assemble_kmat(nxe, num_nodes, block_dim, num_bcs, &bcs[0], rowPtr, colPtr,
                  red_kelem_dense, _values);

    // now copy data to device for cusparse
    HostVec<double> values(nvals, _values);

    Mat kmat = BsrMat(bsr_data, values);

    double *_rhs = new double[ndof];
    memset(_rhs, 0.0, ndof * sizeof(double));
    _rhs[3 * 4] = 1.0; // f_z load at middle node in plate

    HostVec<double> rhs(ndof, _rhs);

    // double *true_soln = get_python_soln();
    // HostVec<double> h_true_soln(ndof, true_soln);
    HostVec<double> temp(ndof), soln(ndof);

    // auto bsrData = kmat.getBsrData();
    // DeviceVec<int> d_rowPtr(bsrData.nnodes + 1, bsrData.rowPtr);
    // auto h_rowPtr = d_rowPtr.createHostVec();
    // DeviceVec<int> d_colPtr(bsrData.nnzb, bsrData.colPtr);
    // auto h_colPtr = d_colPtr.createHostVec();

    // // printf("kmat pre solve: \n");
    // // printVec<double>(nvals, values);
    // write_to_csv<double>(h_kmat.getPtr(), h_kmat.getSize(),
    //                      "csv/plate_kmat.csv");
    // write_to_csv<int>(h_rowPtr.getPtr(), h_rowPtr.getSize(),
    //                   "csv/plate_rowPtr.csv");
    // write_to_csv<int>(h_colPtr.getPtr(), h_colPtr.getSize(),
    //                   "csv/plate_colPtr.csv");

    // write_to_csv<double>(h_loads.getPtr(), h_loads.getSize(),
    // "csv/plate_loads.csv"); write_to_csv<double>(h_soln.getPtr(),
    // h_soln.getSize(), "csv/plate_soln.csv");

    EIGEN::iterative_CG_solve<double>(kmat, rhs, soln);
    // auto max_resid = EIGEN::get_resid<double>(kmat, d_rhs, d_soln);

    // true solution from examples/cusparse_solve/4_plate_soln.py
    double _true_soln[] = {
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.03602546e-08,  7.63638404e-09,  0.00000000e+00,  3.91671045e-12,
        -1.52736269e-08, 0.00000000e+00,  2.54606903e-09,  -7.63525545e-09,
        5.09217100e-09,  2.54567738e-09,  7.63592723e-09,  0.00000000e+00,
        2.54528567e-09,  -7.63615095e-09, 0.00000000e+00,  5.08848194e-09,
        -1.52736457e-08, 0.00000000e+00,  -1.52709894e-08, 7.63547004e-09,
        0.00000000e+00,  5.08613195e-09,  2.36960227e-12};
    HostVec<double> true_soln(ndof, _true_soln);

    printf("soln: ");
    printVec<double>(ndof, soln.getPtr());

    printf("true_soln: ");
    printVec<double>(ndof, true_soln.getPtr());

    getVecRelError<double>(soln, true_soln);
    // printf("cusparse linear solve max error: %.8e\n", max_resid);

    return 0;
};