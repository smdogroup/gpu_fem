#pragma once
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

// util
// Helper function to convert BSR to Eigen triplets
template <typename T>
void BSRToTriplets(int num_nodes, int block_dim, const int *rowPtr,
                   const int *colPtr, const T *values,
                   std::vector<Eigen::Triplet<T>> &triplets) {

    for (int block_row = 0; block_row < num_nodes; ++block_row) {
        for (int j = rowPtr[block_row]; j < rowPtr[block_row + 1]; ++j) {
            int block_col = colPtr[j];
            for (int bi = 0; bi < block_dim; ++bi) {
                for (int bj = 0; bj < block_dim; ++bj) {
                    // Compute global row and column indices
                    int globalRow = block_row * block_dim + bi;
                    int globalCol = block_col * block_dim + bj;

                    // Index in the flat `values` array
                    int flatIndex =
                        (j * block_dim * block_dim) + (bi * block_dim + bj);

                    // printf("values[%d] = %.8e at (%d,%d)\n", flatIndex,
                    //        values[flatIndex], globalRow, globalCol);

                    // Add non-zero value to the triplet list
                    triplets.emplace_back(globalRow, globalCol,
                                          values[flatIndex]);
                }
            }
        }
    }
}

/*
EIGEN package for linear solvers
on the CPU (for debugging and comparison)
*/
namespace EIGEN {

template <typename T>
void iterative_CG_solve(BsrMat<HostVec<T>> &mat, HostVec<T> &rhs,
                        HostVec<T> &soln, const bool can_print = false) {

    auto bsr_data = mat.getBsrData();
    int num_nodes = bsr_data.nnodes;
    int block_dim = bsr_data.block_dim;
    int *rowPtr = bsr_data.rowPtr;
    int *colPtr = bsr_data.colPtr;
    T *values = mat.getPtr();
    int num_global = num_nodes * block_dim;

    // Convert BSR to triplets (may not be efficient to copy data here)
    // but just using this as a debug solver here..
    std::vector<Eigen::Triplet<T>> triplets;
    BSRToTriplets<T>(num_nodes, block_dim, rowPtr, colPtr, values, triplets);

    // Build the Eigen sparse matrix
    Eigen::SparseMatrix<T> A(num_global, num_global);
    A.setFromTriplets(triplets.begin(), triplets.end());

    // Display the sparse matrix
    if (can_print) {
        std::cout << "Matrix A:\n" << Eigen::MatrixXd(A) << std::endl;
    }

    // Define the right-hand side vector b (that points to rhs memory)
    Eigen::Map<Eigen::VectorXd> b(rhs.getPtr(), rhs.getSize());

    // Solve Ax = b using Conjugate Gradient
    Eigen::ConjugateGradient<Eigen::SparseMatrix<T>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed!" << std::endl;
        return;
    }

    Eigen::VectorXd x = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Solve failed!" << std::endl;
        return;
    }

    // now copy data into soln from x Eigen::VectorXd object
    std::memcpy(soln.getPtr(), x.data(), x.size() * sizeof(T));
}

}; // namespace EIGEN
