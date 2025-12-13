/**
Evaluate performance of linear solve, MELD, etc. on plate case
*/

#include "../plate/_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "coupled/meld.h"
#include <chrono>

// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

template <int NN_PER_BLOCK = 32>
void time_MELD(int nxs, int nxa, bool print = true, bool debug = false, bool write_vtk = false) {

    using T = double;
    
    // setup 
    T Lx = 1.0, Ly = 1.0;
    int nnx_a = nxa, nny_a = nxa;
    int nnx_s = nxs, nny_s = nxs;

    auto xs0 = makeGridMesh<T>(nnx_s, nny_s, Lx, Ly, 0.01);
    auto xa0 = makeGridMesh<T>(nnx_a, nny_a, Lx, Ly, 0.01);

    // prescribed displacements
    auto h_us = makeInPlaneShearDisp<T>(xs0, 20.0);

    // convert to device vecs
    auto d_xa0 = xa0.createDeviceVec();
    auto d_xs0 = xs0.createDeviceVec();
    auto d_us = h_us.createDeviceVec();
    
    // create MELD
    T beta = 10.0;
    int sym = -1; // no symmetry yet I believe
    int nn = 128;
    double Hreg = 1e-4;
    auto meld = MELD<T, NN_PER_BLOCK>(d_xs0, d_xa0, beta, nn, sym, Hreg);
    meld.initialize();

    // transfer disps
    auto start_disp = std::chrono::high_resolution_clock::now();
    auto d_ua = meld.transferDisps(d_us);
    auto end_disp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> disp_time = end_disp - start_disp;
    auto h_ua = d_ua.createHostVec();

    // visualize one of the meshes in paraview
    // printGridToVTK<T>(nnx_s, nny_s, xs0, h_us, "xs.vtk");
    if (write_vtk) printGridToVTK<T>(nnx_a, nny_a, xa0, h_ua, "xa.vtk");

    // now setup loads and do load transfer
    auto h_fa = makeInPlaneShearDisp<T>(xa0, 10.0);
    auto d_fa = h_fa.createDeviceVec();

    auto start_load = std::chrono::high_resolution_clock::now();
    auto d_fs = meld.transferLoads(d_fa);
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_time = end_load - start_load;
    auto h_fs = d_fs.createHostVec();
    
    // printf("h_fs:");
    // printVec<double>(h_fs.getSize(), h_fs.getPtr());

    // printGridToVTK<T>(nnx_a, nny_a, xa0, h_fa, "out/fa.vtk");
    if (write_vtk) printGridToVTK<T>(nnx_s, nny_s, xs0, h_fs, "fs.vtk");

    // compute total aero forces
    T fA_tot[3], fS_tot[3];
    int na = h_fa.getSize() / 3;
    int ns = h_fs.getSize() / 3;
    memset(fA_tot, 0.0, 3 * sizeof(T));
    memset(fS_tot, 0.0, 3 * sizeof(T));

    for (int ia = 0; ia < na; ia++) {
        for (int idim = 0; idim < 3; idim++) {
            fA_tot[idim] += h_fa[3 * ia + idim];
        }
    }

    for (int is = 0; is < ns; is++) {
        for (int idim = 0; idim < 3; idim++) {
            fS_tot[idim] += h_fs[3 * is + idim];
        }
    }

    if (debug) {
        printf("fA_tot:");
        printVec<double>(3, &fA_tot[0]);

        printf("fS_tot:");
        printVec<double>(3, &fS_tot[0]);
    }

    // compute total work done
    T W_A = 0.0, W_S = 0.0;
    for (int ia = 0; ia < na; ia++) {
        for (int idim = 0; idim < 3; idim++) {
            W_A += h_fa[3 * ia + idim] * h_ua[3 * ia + idim];
        }
    }
    for (int is = 0; is < ns; is++) {
        for (int idim = 0; idim < 3; idim++) {
            W_S += h_fs[3 * is + idim] * h_us[3 * is + idim];
        }
    }

    if (debug) printf("W_A %.4e, W_S %.4e\n", W_A, W_S);

    // print runtimes, later do it in CSV format
    if (debug) printf("transfer disps in %.4e sec, transfer loads in %.4e sec\n", disp_time.count(), load_time.count());

    // this is what will be written out to vtk
    // nstruct_elems, transfer_disp_time (sec), transfer_load_time (sec)
    int nstruct_nodes = nxs * nxs;
    if (print) printf("%d, %d, %.4e, %.4e\n", nstruct_nodes, NN_PER_BLOCK, disp_time.count(), load_time.count());

    return;
};

int main() {
    // test meld
    printf("nstruct_nodes, NN_per_block, transfer_disp_time(s), transfer_loads_time(s)\n");
    int nxs = 20, nxa = 13;
    for (int i = 0; i < 7; i++) {
        // les than 32 is slower bc of warp size
        // time_MELD<16>(nxs, nxa, true);

        time_MELD<32>(nxs, nxa, true);

        // higher NN_PER_BLOCK is not running yet (CUDA illegal mem access..)
        // time_MELD<64>(nxs, nxa, true);
        // time_MELD<128>(nxs, nxa, true);
        nxs *= 2;
        nxa *= 2;
    }

    // test linear static
    
}