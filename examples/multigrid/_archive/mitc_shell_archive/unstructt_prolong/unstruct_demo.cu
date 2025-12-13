/* develop fast block-GS multicolor using small matrix for testing purposes.. */
// 7 node test matrix with 2x2 block dim for multicoloring (aka 14 DOF).. that I've made up

// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

// local multigrid imports
#include "../../include/grid.h"
#include "../../include/fea.h"
#include "../../include/mg.h"
#include <string>
#include <chrono>

#include "coupled/locate_point.h"
#include "helper.h"

// #include <cusparse_v2.h>
// #include "cublas_v2.h"
// #include "cuda_utils.h"
// #include "linalg/vec.h"
// #include "solvers/linear_static/_cusparse_utils.h"

int main() {

    // fine grid input
    int nxe = 32;
    
    // shells
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;
    const SMOOTHER smoother = MULTICOLOR_GS_FAST;
    using Prolongation = StructuredProlongation<PLATE>;

    using GRID = ShellGrid<Assembler, Prolongation, smoother>;
    using MG = ShellMultigrid<GRID>;

    auto start0 = std::chrono::high_resolution_clock::now();

    // generate two grids (fine and coarse
    std::vector<GRID> grids;

    // fine mesh distortions
    int m_fine = 3, n_fine = 3;
    T x_frac_f = 0.25, y_frac_f = 0.25, shear_frac_f = 0.7;

    // coarse mesh distortions
    int m_coarse = 2, n_coarse = 2;
    T x_frac_c = 0.5, y_frac_c = 0.4, shear_frac_c = 0.4;

    double SR = 50.0;

    // make each grid (two grids)
    int nxe_min = nxe / 2;
    int i_level = 0;
    for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
        // make the assembler
        int c_nye = c_nxe;
        double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
        int nxe_per_comp = c_nxe / 4, nye_per_comp = c_nye/4; // for now (should have 25 grids)

        // distortion 
        bool is_fine = c_nxe == nxe;
        int m = is_fine ? m_fine : m_coarse;
        int n = is_fine ? n_fine : n_coarse;
        T x_frac = is_fine ? x_frac_f : x_frac_c;
        T y_frac = is_fine ? y_frac_f : y_frac_c;
        T shear_frac = is_fine ? shear_frac_f : shear_frac_c;

        auto assembler = createPlateDistortedAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp, 
            m, n, x_frac, y_frac, shear_frac);
        double Q = 1.0; // load magnitude
        T *my_loads = getPlateLoads<T, Physics>(c_nxe, c_nye, Lx, Ly, Q);
        printf("making grid with nxe %d\n", c_nxe);

        // make the grid
        bool full_LU = true; // temp so we can see analyses
        // bool full_LU = c_nxe == nxe_min; // smallest grid is direct solve
        grids.push_back(*GRID::buildFromAssembler(assembler, my_loads, full_LU, true));
        i_level++;
    }

    // now run an analysis real quick.. writeout the solution..
    bool solve = true;
    if (solve) {
        for (int i = 0; i < 2; i++) {
            grids[i].direct_solve();

            auto h_soln = grids[i].d_soln.createPermuteVec(6, grids[i].d_perm).createHostVec();
            printToVTK<Assembler, HostVec<T>>(grids[i].assembler, h_soln, "grid_" + std::to_string(i) + ".vtk");
        }
    }

    // -------------------------------------------------------
    // 1) prelim prolongation maps and data here..
    T *h_xpts_fine = grids[0].assembler.getXpts().createHostVec().getPtr();
    T *h_xpts_coarse = grids[1].assembler.getXpts().createHostVec().getPtr();
    int nnodes_fine = grids[0].assembler.get_num_nodes();
    int nnodes_coarse = grids[1].assembler.get_num_nodes();

    int min_bin_size = 10;
    auto *locator = new LocatePoint<T>(h_xpts_coarse, nnodes_coarse, min_bin_size);

    int nn = 6; // number of coarse node nearest neighbors
    int *nn_conn = new int[nn * nnodes_fine];
    // temp work arrays for each point
    int *indx = new int[nn];
    T *dist = new T[nn];

    for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
        T loc_xfine[3];
        memcpy(loc_xfine, &h_xpts_fine[3 * inode_f], 3 * sizeof(T)); // is this part necessary?

        locator->locateKClosest(nn, indx, dist, loc_xfine);

        for (int k = 0; k < nn; k++) {
            nn_conn[nn * inode_f + k] = indx[k];
        }
    }

    // check this..
    // printf("nn conn: ");
    // printVec<int>(30, nn_conn);

    // -------------------------------------------------------
    // 2) get coarse elements for each coarse node
    auto d_coarse_conn = grids[1].assembler.getConn();
    int *h_coarse_conn = d_coarse_conn.createHostVec().getPtr();
    int *cnode_elem_cts = new int[nnodes_coarse];
    memset(cnode_elem_cts, 0.0, nnodes_coarse * sizeof(int));
    int *cnode_elem_ptr = new int[nnodes_coarse + 1];
    int num_coarse_elems = grids[1].assembler.get_num_elements();
    int n_coarse_node_elems = 4 * num_coarse_elems;
    for (int ielem = 0; ielem < num_coarse_elems; ielem++) {
        for (int iloc = 0; iloc < 4; iloc++) {
            int cnode = h_coarse_conn[4 * ielem + iloc];
            cnode_elem_cts[cnode] += 1;
        }
    }
    // printf("cnode elem cts: ");
    // printVec<int>(nnodes_coarse, cnode_elem_cts);

    // like rowp here (says where and how many elems this node is a part of)
    cnode_elem_ptr[0] = 0;
    for (int inode = 0; inode < nnodes_coarse; inode++) {
        cnode_elem_ptr[inode+1] = cnode_elem_ptr[inode] + cnode_elem_cts[inode];
    }

    // now we put which elems each coarse node is connected to (like cols array here)
    // reset row cts to 0, so you can trick which local elem you're writing in..
    memset(cnode_elem_cts, 0.0, nnodes_coarse * sizeof(int));
    int *cnode_elems = new int[n_coarse_node_elems];
    for (int ielem = 0; ielem < num_coarse_elems; ielem++) {
        for (int iloc = 0; iloc < 4; iloc++) {
            int cnode = h_coarse_conn[4 * ielem + iloc];
            int ind = cnode_elem_ptr[cnode] + cnode_elem_cts[cnode];
            cnode_elems[ind] = ielem;
            cnode_elem_cts[cnode] += 1;
        }
    }
    
    // printf("cnode elem ptr: ");
    // printVec<int>(nnodes_coarse + 1, cnode_elem_ptr);
    // printf("cnode elems (cols): ");
    // printVec<int>(n_coarse_node_elems, cnode_elems);

    // ---------------------------------------------------------------
    // 2.5) debug test the xi, eta interp on fine node + coarse elem pairs (where I know it's contained)
    int in_f = 34, ie_c = 0;
    // int in_f = 136, ie_c = 17;

    T *fine_xpt = &h_xpts_fine[3 * in_f];
    T c_elem_xpts[12];
    get_elem_xpts<T>(ie_c, h_coarse_conn, h_xpts_coarse, c_elem_xpts);

    printf("fine_xpt of node %d: ", in_f);
    printVec<T>(3, fine_xpt);
    printf("coarse_xpts of elem %d: ", ie_c);
    printVec<T>(12, c_elem_xpts);

    T xis[3];
    
    bool print = false;
    // bool print = true;
    get_comp_coords<T, Basis>(c_elem_xpts, fine_xpt, xis, print);
    printf("pred xis: ");
    printVec<T>(3, xis);

    // return 0;

    // ----------------------------------------------------------------
    // 3) get the coarse element(s) for each fine node (that it's contained in)

    // for wingbox, may need multiple coarse elements (up to 2 I think) due to edges of wingbox like OML-SOB

    int *fine_nodes_celem_cts = new int[nnodes_fine];
    memset(fine_nodes_celem_cts, 0, nnodes_fine * sizeof(int));
    int *fine_nodes_celems = new int[2 * nnodes_fine];
    memset(fine_nodes_celems, -1, 2 * nnodes_fine * sizeof(int));
    T *fine_node_xis = new T[4 * nnodes_fine];
    int ntot_elems = 0;
    // up to two coarse elements for each fine node (puts -1 if no extra element..)

    for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
    // for (int inode_f = 0; inode_f < 1; inode_f++) { // FOR DEBUG
        T *fine_node_xpts = &h_xpts_fine[3 * inode_f];

        // should be ~24 elems examined per coarse node (can parallelize this on GPU if needed)
        for (int i_nn = 0; i_nn < nn; i_nn++) {
            int inode_c = nn_conn[nn * inode_f + i_nn];
            
            for (int jp = cnode_elem_ptr[inode_c]; jp < cnode_elem_ptr[inode_c+1]; jp++) {
                int ielem_c = cnode_elems[jp];

                T coarse_elem_xpts[12];
                get_elem_xpts<T>(ielem_c, h_coarse_conn, h_xpts_coarse, coarse_elem_xpts);

                T xi[3];
                get_comp_coords<T, Basis>(coarse_elem_xpts, fine_node_xpts, xi);

                // determine whether xi[3] is in bounds or not; xi & eta in [-1,1] and zeta in [-2,2] for max thick (or can ignore zeta..)
                bool node_in_elem = xis_in_elem<T>(xi);
                if (node_in_elem) {
                    int nelems_prev = fine_nodes_celem_cts[inode_f];
                    int start = max(0, nelems_prev - 1);
                    int prev_elem = fine_nodes_celems[2 * inode_f + start];
                    bool new_elem = ielem_c != prev_elem;

                    if (new_elem) {
                        fine_nodes_celem_cts[inode_f]++;
                        ntot_elems++;
                        fine_nodes_celems[2 * inode_f + nelems_prev] = ielem_c;
                        fine_node_xis[4 * inode_f + 2 * nelems_prev] = xi[0];
                        fine_node_xis[4 * inode_f + 2 * nelems_prev + 1] = xi[1];
                    }
                }

                // DEBUG
                // std::string result = node_in_elem ? "yes" : "no";
                // printf("fine node %d in coarse elem %d : %s\n", inode_f, ielem_c, result.c_str());
                // printf("\txis: ");
                // printVec<T>(3, xi);
            }
        }
    }

    // printf("-------\nfine_nodes_celem_cts: ");
    // printVec<int>(nnodes_fine, fine_nodes_celem_cts);
    // printf("-------\nfine nodes celems: ");
    // printVec<int>(2 * nnodes_fine, fine_nodes_celems);

    // now convert from cts to rowp, cols style as diff # elems per fine node
    int *fine_node2elem_ptr = new int[nnodes_fine + 1];
    fine_node2elem_ptr[0] = 0;
    int *fine_node2elem_elems = new int[ntot_elems];
    T *fine_node2elem_xis = new T[2 * ntot_elems];

    for (int inode = 0; inode < nnodes_fine; inode++) {
        int ct = fine_nodes_celem_cts[inode];
        fine_node2elem_ptr[inode + 1] = fine_node2elem_ptr[inode] + ct;
        int start = fine_node2elem_ptr[inode];

        for (int i = 0; i < ct; i++) {
            int src_block = 2 * inode + i, dest_block = start + i;
            fine_node2elem_elems[dest_block] = fine_nodes_celems[src_block];
            fine_node2elem_xis[2 * dest_block] = fine_node_xis[2 * src_block];
            fine_node2elem_xis[2 * dest_block + 1] = fine_node_xis[2 * src_block + 1];
        }
    }

    // printf("-------\nfine_node2elem_ptr: ");
    // printVec<int>(nnodes_fine + 1, fine_node2elem_ptr);
    // printf("-------\nfine_node2elem_elems: ");
    // printVec<int>(ntot_elems, fine_node2elem_elems);
    // printf("-------\nfine_node2elem_xis: ");
    // printVec<T>(2 * ntot_elems, fine_node2elem_xis);

    // put these maps on the device now
    int *d_n2e_ptr = HostVec<int>(nnodes_fine + 1, fine_node2elem_ptr).createDeviceVec().getPtr();
    int *d_n2e_elems = HostVec<int>(ntot_elems, fine_node2elem_elems).createDeviceVec().getPtr();
    T *d_n2e_xis = HostVec<T>(2 * ntot_elems, fine_node2elem_xis).createDeviceVec().getPtr();

    // -------------------------------------------------------
    // 4) TBD
    // now try prolongation.. with unstruct mesh style (even though technically they are very different distorted struct meshes, still need same general method)
    // do this on the host first..

    // get some perm maps and other things we need for prolong
    int *d_fine_iperm = grids[0].d_iperm;
    int *d_coarse_iperm = grids[1].d_iperm;
    T *d_coarse_soln = grids[1].d_soln.getPtr(); // permuted form
    int fine_ndof = grids[0].N;
    auto d_fine_prolong_soln = DeviceVec<T>(fine_ndof); // will be permuted form
    auto d_fine_wts = DeviceVec<T>(fine_ndof); // will be permuted form

    // printf("here\n");
    // return 0;

    dim3 block(32);
    int nblocks = (nnodes_fine + 31) / 32;
    dim3 grid(nblocks);

    k_unstruct_prolongate<T,Basis><<<grid, block>>>(d_coarse_soln, d_coarse_iperm, d_coarse_conn.getPtr(), d_n2e_ptr, d_n2e_elems, 
        d_n2e_xis, nnodes_fine, d_fine_iperm, d_fine_prolong_soln.getPtr(), d_fine_wts.getPtr());
    CHECK_CUDA(cudaDeviceSynchronize());

    // printf("done with unstruct prolong kernel\n");
    // return 0;

    // normalize
    int nblocks2 = (fine_ndof + 31) / 32;
    dim3 grid2(nblocks2);

    k_vec_normalize<T><<<grid2, block>>>(fine_ndof, d_fine_prolong_soln.getPtr(), d_fine_wts.getPtr());

    // now print the prolongated soln.. after iperm
    auto h_prolong_soln = d_fine_prolong_soln.createPermuteVec(6, grids[0].d_perm).createHostVec();
    printToVTK<Assembler, HostVec<T>>(grids[0].assembler, h_prolong_soln, "grid_prolong.vtk");
    

    return 0;
};