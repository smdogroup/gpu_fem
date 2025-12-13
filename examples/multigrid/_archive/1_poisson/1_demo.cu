#include "include/pde.h"

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

int main() {
    // make poisson solver and try and run it..

    using T = float; // double

    // int nxe = 2;
    // int nxe = 3;
    int nxe = 4;
    // int nxe = 8;
    // int nxe = 16;

    // int nxe = 10;
    // int nxe = 30;
    // int nxe = 64;
    // int nxe = 100;
    // int nxe = 300;
    // int nxe = 500;
    // int nxe = 1024; // about 1M DOF (works)
    // int nxe = 4000; // about 16M DOF (works)
    // int nxe = 6000; // about 36M DOF (works)
    // int nxe = 8000; // about 64M DOF (works)
    // int nxe = 10000; // 100M DOF (fails on 3060 Ti)
    // int nxe = 30000; // 1 billion DOF (fails on 3060 Ti)

    bool red_black_order = false;
    // bool red_black_order = true; // use for red-black GS
    auto solver = PoissonSolver<T>(nxe, red_black_order);

    // return 0;

    // // try printing out data structures to check..
    int N = solver.N;
    int n_print = 300;

    // printf("h_rowp: ");
    int *h_rowp = solver.d_csr_rowp.createHostVec().getPtr();
    // printVec<int>(min(n_print, N+1), h_rowp);

    printf("csr_nnz %d\n", (int)solver.csr_nnz);
    printf("h_rows: ");
    int *h_rows = solver.d_csr_rows.createHostVec().getPtr();
    printVec<int>(min(n_print, (int)solver.csr_nnz), h_rows);

    printf("h_cols: ");
    int *h_cols = solver.d_csr_cols.createHostVec().getPtr();
    printVec<int>(min(n_print, (int)solver.csr_nnz), h_cols);

    T *h_x0 = solver.d_soln.createHostVec().getPtr();

    // // now printout lhs and rhs
    // printf("h_lhs:");
    T *h_lhs = solver.d_lhs.createHostVec().getPtr();
    // printVec<T>(min(n_print, (int)solver.csr_nnz), h_lhs);

    // printf("h_rhs:");
    T *h_rhs = solver.d_rhs.createHostVec().getPtr();
    // printVec<T>(min(n_print, N), h_rhs);
    // return 0;

    // printout lhs and rhs of each row::
    printf("N = %d DOF linear system\n", N);
    for (int row = 0; row < N; row++) {
        // first printout lhs
        printf("row %d : lhs ", row);
        for (int jp = h_rowp[row]; jp < h_rowp[row+1]; jp++) {
            int col = h_cols[jp];
            T val = h_lhs[jp];

            printf("%.2e[%d] ", val, col);
        }
        printf("  * (x0 = %.2e) ", h_x0[row]);
        printf("= (rhs = %.2e)\n", h_rhs[row]);
    }

    // // check lhs diag inv
    // printf("h_diag_inv:");
    // T *h_diag_inv = solver.d_diag_inv.createHostVec().getPtr();
    // printVec<T>(min(n_print, N), h_diag_inv);

    // return;
    

    if (red_black_order) {
        // int n_iter = 100;
        int n_iter = 1000;
        // int n_iter = 3;
        bool print = true;
        solver.redBlackGaussSeidelDefect(n_iter, print, 1);
    } else {
        // int n_iter = 100;
        int n_iter = 1000;
        T omega = 2.0 / 3.0;
        bool print = true;
        solver.dampedJacobiSolve(n_iter, omega, print);
    }

    // get true soln error (including discretization)
    // maybe some error in lhs or rhs here..
    T err_nrm = solver.getSolnError();
    printf("true soln error = %.4e\n", err_nrm);

    T resid_nrm = solver.getResidNorm();
    printf("resid nrm = %.4e\n", resid_nrm);

    // write soln to csv file ..
    T *h_soln = solver.d_soln.createHostVec().getPtr();
    write_to_csv<T>(solver.N, h_soln, "8x8_dof_soln.csv");

    // free memory
    solver.free();

    return 0;
};