#pragma once
#include <fstream>
#include <iostream>
#include <vector>

#include "coupled/_coupled.h"
#include "linalg/_linalg.h"

template <typename T, class Assembler>
DeviceVec<T> getSurfLoads(Assembler &assembler, T load_mag) {
    // find nodes within tolerance of x=10.0
    int num_nodes = assembler.get_num_nodes();
    int num_vars = assembler.get_num_vars();
    HostVec<T> h_loads(num_vars);
    auto d_xpts = assembler.getXpts();
    auto h_xpts = d_xpts.createHostVec();
    double *h_loads_ptr = h_loads.getPtr();
    for (int inode = 0; inode < num_nodes; inode++) {
        // only apply pressure loads on top and bottom surface
        // otherwise MELD has weird load transfer on spars
        h_loads_ptr[6 * inode + 2] = load_mag;
        // if (h_xpts[3 * inode + 2] == 2.5 || h_xpts[3 * inode + 2] == 5.0) {
        //     h_loads_ptr[6 * inode + 2] = load_mag;
        // }
    }
    auto loads = h_loads.createDeviceVec();
    assembler.apply_bcs(loads);
    return loads;
}

template <typename T, class StructSolver, class AeroSolver, class Transfer, class Assembler>
void testCoupledDriver(StructSolver struct_solver, AeroSolver aero_solver, Transfer transfer,
                       Assembler assembler) {
    using CoupledDriver =
        FuntofemCoupledAnalysis<T, DeviceVec<T>, StructSolver, AeroSolver, Transfer>;
    int num_coupled_steps = 2;
    bool demo = true;  // demo settings for nonlinear struct (just resets it)
    CoupledDriver driver =
        CoupledDriver(struct_solver, aero_solver, transfer, num_coupled_steps, demo);
    driver.solve_forward();

    int ns = struct_solver.get_num_nodes();
    auto us = struct_solver.getStructDisps();
    auto h_us = us.createHostVec();
    auto h_us_ext = MELD<T>::template expandVarsVec<3, 6>(ns, h_us);
    printToVTK<Assembler, HostVec<T>>(assembler, h_us_ext,
                                      "uCRM_us.vtk");  // zero everywhere? weird, fix this..
}

template <typename T, class StructSolver, class AeroSolver, class Transfer, class Assembler>
void testCoupledDriverManual(StructSolver struct_solver, AeroSolver aero_solver, Transfer transfer,
                             Assembler assembler, Assembler _assembler_aero,
                             bool aero_point_data = false) {
    int ns = assembler.get_num_nodes();
    int na = _assembler_aero.get_num_nodes();

    printf("here10\n");

    // break out the coupled loop manually for testing
    auto us = DeviceVec<T>(3 * ns);
    auto h_us = us.createHostVec();
    auto h_us_ext = MELD<T>::template expandVarsVec<3, 6>(ns, h_us);
    printToVTK<Assembler, HostVec<T>>(assembler, h_us_ext, "uCRM_us-0.vtk");  // zero, good

    printf("1st transfer disps call\n");
    auto ua = transfer.transferDisps(us);
    auto h_ua = ua.createHostVec();
    auto h_ua_ext = MELD<T>::template expandVarsVec<3, 6>(na, h_ua);

    if (aero_point_data) {
        printToVTK_points<Assembler, HostVec<T>>(_assembler_aero, h_ua_ext,
                                                 "uCRM_ua-0.vtk");  // small, near zero good
    } else {
        printToVTK<Assembler, HostVec<T>>(_assembler_aero, h_ua_ext,
                                          "uCRM_ua-0.vtk");  // small, near zero good
    }

    auto fa = aero_solver.getAeroLoads();
    auto h_fa = fa.createHostVec();
    auto h_fa_ext = MELD<T>::template expandVarsVec<3, 6>(na, h_fa);
    if (aero_point_data) {
        printToVTK_points<Assembler, HostVec<T>>(_assembler_aero, h_fa_ext,
                                                 "uCRM_fa-0.vtk");  // small, near zero good
    } else {
        printToVTK<Assembler, HostVec<T>>(_assembler_aero, h_fa_ext,
                                          "uCRM_fa-0.vtk");  // small, near zero good
    }

    auto fs = transfer.transferLoads(fa);
    auto fs_ext = fs.addRotationalDOF();
    auto h_fs_ext = fs_ext.createHostVec();
    printToVTK<Assembler, HostVec<T>>(
        assembler, h_fs_ext,
        "uCRM_fs-0.vtk");  // z force only, could be smoother though (increased beta), good

    struct_solver.solve(fs_ext);
    auto us1 = struct_solver.getStructDisps();
    auto h_us1 = us1.createHostVec();
    auto h_us1_ext = MELD<T>::template expandVarsVec<3, 6>(ns, h_us1);
    printToVTK<Assembler, HostVec<T>>(assembler, h_us1_ext,
                                      "uCRM_us-1.vtk");  // zero everywhere? weird, fix this..

    // second loop
    printf("2nd transfer disps call\n");
    auto ua1 = transfer.transferDisps(us1);
    auto h_ua1 = ua1.createHostVec();
    auto h_ua1_ext = MELD<T>::template expandVarsVec<3, 6>(na, h_ua1);
    if (aero_point_data) {
        printToVTK_points<Assembler, HostVec<T>>(_assembler_aero, h_ua1_ext,
                                                 "uCRM_ua-1.vtk");  // small, near zero good
    } else {
        printToVTK<Assembler, HostVec<T>>(_assembler_aero, h_ua1_ext,
                                          "uCRM_ua-1.vtk");  // small, near zero good
    }

    auto fa1 = aero_solver.getAeroLoads();
    auto h_fa1 = fa1.createHostVec();
    auto h_fa1_ext = MELD<T>::template expandVarsVec<3, 6>(na, h_fa1);
    if (aero_point_data) {
        printToVTK_points<Assembler, HostVec<T>>(_assembler_aero, h_fa1_ext,
                                                 "uCRM_fa-1.vtk");  // small, near zero good
    } else {
        printToVTK<Assembler, HostVec<T>>(_assembler_aero, h_fa1_ext,
                                          "uCRM_fa-1.vtk");  // small, near zero good
    }

    auto fs1 = transfer.transferLoads(fa1);
    auto fs1_ext = fs1.addRotationalDOF();
    auto h_fs1_ext = fs1_ext.createHostVec();
    printToVTK<Assembler, HostVec<T>>(assembler, h_fs1_ext, "uCRM_fs-1.vtk");

    struct_solver.solve(fs1_ext);
    auto us2 = struct_solver.getStructDisps();
    auto h_us2 = us2.createHostVec();
    // printf("us1:");
    // printVec<T>(10, h_us1.getPtr());
    // printf("us2:");
    // printVec<T>(10, h_us2.getPtr());
    auto h_us2_ext = MELD<T>::template expandVarsVec<3, 6>(ns, h_us2);
    printToVTK<Assembler, HostVec<T>>(assembler, h_us2_ext, "uCRM_us-2.vtk");
}

template <class Assembler, bool narrow_outboard = true>
Assembler makeAeroSurfMesh(int nx = 101, int ny = 101, bool print = true) {
    // V1 is outside the wing, but much larger than it.. doesn't narrow down the same way
    using T = typename Assembler::T;
    using Basis = typename Assembler::Basis;
    using Geo = typename Assembler::Geo;
    using Data = typename Assembler::Data;

    double zsob_mid = 3.25;
    // double zsob_mid = 2.5;

    int ncomp = 8;
    int N = nx * ny * ncomp;
    HostVec<T> xa0(3 * N);
    int Nh = nx * ny;
    int num_nodes = N;
    for (int inode = 0; inode < Nh; inode++) {
        int ix = inode % nx;
        int iy = inode / ny;
        T xfrac = ix * 1.0 / (nx - 1);
        T yfrac = iy * 1.0 / (ny - 1);
        // printf("ix %d, iy %d, xfrac %.4e, yfrac %.4e\n", ix, iy, xfrac, yfrac);

        // lower skin plane from root to SOB
        xa0[3 * inode] = 25 + (30.5 - 25) * xfrac;
        xa0[3 * inode + 1] = 3.2 * yfrac;
        xa0[3 * inode + 2] = 2.5 + (zsob_mid - 2.5) * yfrac;

        // upper skin plane from root to SOB
        xa0[3 * inode + 3 * Nh] = xa0[3 * inode];
        xa0[3 * inode + 1 + 3 * Nh] = xa0[3 * inode + 1];
        xa0[3 * inode + 2 + 3 * Nh] = 5.0;

        // root to SB at TE
        xa0[3 * inode + 6 * Nh] = 30.5;
        xa0[3 * inode + 6 * Nh + 1] = 3.2 * xfrac;
        xa0[3 * inode + 6 * Nh + 2] = 2.5 + (5.0 - 2.5) * yfrac;

        // root to SB at LE
        xa0[3 * inode + 9 * Nh] = 25.0;
        xa0[3 * inode + 9 * Nh + 1] = 3.2 * xfrac;
        xa0[3 * inode + 9 * Nh + 2] = 2.5 + (5.0 - 2.5) * yfrac;

        // SB to tip at LE
        xa0[3 * inode + 12 * Nh] = 25.0 + (49.0 - 25.0) * xfrac;
        xa0[3 * inode + 12 * Nh + 1] = 3.2 + (36 - 3.2) * xfrac;
        if constexpr (narrow_outboard) {
            // 2.5 to 5.0 at SOB, 4.3 to 4.55 at tip
            double a = 4.3 - zsob_mid;
            xa0[3 * inode + 12 * Nh + 2] =
                zsob_mid + (5.0 - zsob_mid) * yfrac + xfrac * (a + (4.55 - a - 5.0) * yfrac);
        } else {
            xa0[3 * inode + 12 * Nh + 2] = zsob_mid + (5.0 - zsob_mid) * yfrac;
        }

        // SB to tip at TE
        xa0[3 * inode + 15 * Nh] = 30.5 + (49.5 - 30.5) * xfrac;
        xa0[3 * inode + 15 * Nh + 1] = 3.2 + (36 - 3.2) * xfrac;
        if constexpr (narrow_outboard) {
            // 2.5 to 5.0 at SOB, 4.3 to 4.55 at tip
            double a = 4.3 - zsob_mid;
            xa0[3 * inode + 15 * Nh + 2] =
                zsob_mid + (5.0 - zsob_mid) * yfrac + xfrac * (a + (4.55 - a - 5.0) * yfrac);
        } else {
            xa0[3 * inode + 15 * Nh + 2] = zsob_mid + (5.0 - zsob_mid) * yfrac;
        }

        // lower skin plane from SOB to tip
        xa0[3 * inode + 18 * Nh] = (25.0 + 5.5 * yfrac) + xfrac * (24 + -5.0 * yfrac);
        xa0[3 * inode + 1 + 18 * Nh] = 3.2 + (36 - 3.2) * xfrac;
        if constexpr (narrow_outboard) {
            // 2.5 bottom at SOB to 4.3 at tip
            xa0[3 * inode + 2 + 18 * Nh] = zsob_mid + (4.3 - zsob_mid) * xfrac;
        } else {
            xa0[3 * inode + 2 + 18 * Nh] = zsob_mid;
        }

        // upper skin plane from SOB to tip
        xa0[3 * inode + 21 * Nh] = (25.0 + 5.5 * yfrac) + xfrac * (24 + -5.0 * yfrac);
        xa0[3 * inode + 1 + 21 * Nh] = 3.2 + (36 - 3.2) * xfrac;
        if constexpr (narrow_outboard) {
            // 2.5 bottom at SOB to 4.55 at tip
            xa0[3 * inode + 2 + 21 * Nh] = 5.0 - 0.45 * xfrac;
        } else {
            xa0[3 * inode + 2 + 21 * Nh] = 5.0;
        }
    }

    // make the element connectivity
    int nxe = nx - 1;
    int nye = ny - 1;
    int num_elements_h = nxe * nye;
    int num_elements = num_elements_h * ncomp;
    int N1 = Basis::num_nodes * num_elements;
    int32_t *elem_conn = new int[N1];
    for (int iye = 0; iye < nye; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;
            // TODO : issue with defining conn out of order like this, needs to
            // be sorted now?""
            int nodes[] = {nx * iye + ixe, nx * iye + ixe + 1, nx * (iye + 1) + ixe,
                           nx * (iye + 1) + ixe + 1};
            for (int inode = 0; inode < Basis::num_nodes; inode++) {
                for (int icomp = 0; icomp < ncomp; icomp++) {
                    elem_conn[Basis::num_nodes * (ielem + icomp * num_elements_h) + inode] =
                        nodes[inode] + icomp * Nh;
                }
            }
        }
    }

    // printf("checkpoint 3 - post elem_conn\n");

    HostVec<int32_t> geo_conn(N1, elem_conn);
    HostVec<int32_t> vars_conn(N1, elem_conn);

    T E = 1e7, nu = 0.3, thick = 0.1;
    HostVec<Data> physData(num_elements, Data(E, nu, thick));

    std::vector<int> my_bcs;
    my_bcs.push_back(0);  // not actually using bcs here, don't really care
    HostVec<int> bcs(my_bcs.size());
    // deep copy here
    for (int ibc = 0; ibc < my_bcs.size(); ibc++) {
        bcs[ibc] = my_bcs.at(ibc);
    }

    if (print) printf("num_nodes %d, num_elements %d\n", num_nodes, num_elements);
    // printf("xpts:");
    // printVec<T>(10, xa0.getPtr());

    int num_components = 0;
    HostVec<int> elem_components(num_components);

    Assembler assembler(num_nodes, num_nodes, num_elements, geo_conn, vars_conn, xa0, bcs, physData,
                        num_components, elem_components);

    return assembler;
}

struct ForceEntry {
    double x, y, z;     // Coordinates
    double fx, fy, fz;  // Forces
};

template <class Assembler>
Assembler makeFun3dAeroSurfMeshFromDat(std::string filename, double **xyz_forces,
                                       bool print = false) {
    // V1 is outside the wing, but much larger than it.. doesn't narrow down the same way
    using T = typename Assembler::T;
    using Basis = typename Assembler::Basis;
    using Geo = typename Assembler::Geo;
    using Data = typename Assembler::Data;

    // read FUN3D dat file "fun3d_forces.dat"
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file\n";
    }

    printf("here1\n");

    std::vector<ForceEntry> forceData;
    ForceEntry entry;

    while (infile >> entry.x >> entry.y >> entry.z >> entry.fx >> entry.fy >> entry.fz) {
        forceData.push_back(entry);
    }

    infile.close();

    printf("here2\n");

    // now print entries to double check / debug
    // for (size_t i = 0; i < std::min(forceData.size(), size_t(5)); ++i) {
    //     std::cout << "Point: (" << forceData[i].x << ", " << forceData[i].y << ", "
    //               << forceData[i].z << ") ";
    //     std::cout << "Force: (" << forceData[i].fx << ", " << forceData[i].fy << ", "
    //               << forceData[i].fz << ")\n";
    // }

    int num_nodes = forceData.size();
    printf("num_nodes = %d\n", num_nodes);
    HostVec<T> xa0(3 * num_nodes);
    *xyz_forces = new T[3 * num_nodes];
    for (int i = 0; i < num_nodes; i++) {
        xa0[3 * i] = forceData[i].x;
        xa0[3 * i + 1] = forceData[i].y;
        xa0[3 * i + 2] = forceData[i].z;
        (*xyz_forces)[3 * i] = forceData[i].fx;
        (*xyz_forces)[3 * i + 1] = forceData[i].fy;
        (*xyz_forces)[3 * i + 2] = forceData[i].fz;
    }
    printf("here3\n");

    // printf("checkpoint 3 - post elem_conn\n");
    // don't need elements for this, not doing struct solve
    int N1 = 4, num_elements = 1;
    int *elem_conn = new int[1];
    for (int i = 0; i < 4; i++) {
        elem_conn[i] = i;
    }
    HostVec<int32_t> geo_conn(N1, elem_conn);
    HostVec<int32_t> vars_conn(N1, elem_conn);

    T E = 1e7, nu = 0.3, thick = 0.1;
    HostVec<Data> physData(num_elements, Data(E, nu, thick));

    std::vector<int> my_bcs;
    my_bcs.push_back(0);  // not actually using bcs here, don't really care
    HostVec<int> bcs(my_bcs.size());
    // deep copy here
    for (int ibc = 0; ibc < my_bcs.size(); ibc++) {
        bcs[ibc] = my_bcs.at(ibc);
    }

    if (print) printf("num_nodes %d, num_elements %d\n", num_nodes, num_elements);
    // printf("xpts:");
    // printVec<T>(10, xa0.getPtr());

    int num_components = 0;
    HostVec<int> elem_components(num_components);

    Assembler assembler(num_nodes, num_nodes, num_elements, geo_conn, vars_conn, xa0, bcs, physData,
                        num_components, elem_components);

    return assembler;
}

// Helper function to convert string to lowercase (in-place)
void to_lowercase(char* str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}