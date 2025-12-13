#include <iostream>

#include "chrono"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/vtk_writer.h"

#include <sstream>
#include <string>

// shell imports
#include "assembler.h"
#include "shell/shell_elem_group.h"
#include "shell/physics/isotropic_shell.h"

/**
 solve on CPU with cusparse for debugging
 **/

int main(void) {
    using T = double;

    std::ios::sync_with_stdio(false);  // always flush print immediately

    TACSMeshLoader<T> mesh_loader{};
    mesh_loader.scanBDFFile("Beam.bdf");

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = true;
    // constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    // material & thick properties
    double E = 1.2e6, nu = 0.0, thick = 0.1; 
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

    // check bcs
    // DeviceVec<int> d_bcs = assembler.getBCs();
    // auto h_bcs = d_bcs.createHostVec();
    // printf("bcs:");
    // printVec<int>(h_bcs.getSize(), h_bcs.getPtr());

    // perform a factorization on the rowPtr, colPtr (before creating matrix)
    double fillin = 10.0;  // 10.0
    assembler.symbolic_factorization(fillin, true);

    // compute load magnitude of tip force
    double length = 10.0; 
    double width = 1.0;
    double Izz = width * thick * thick * thick / 12.0;
    double beam_tip_force = 4.0 * E * Izz / length / length;

    // find nodes within tolerance of x=10.0
    int num_nodes = assembler.get_num_nodes();
    int num_vars = assembler.get_num_vars();
    HostVec<T> h_loads(num_vars);
    DeviceVec<T> d_xpts = assembler.getXpts();
    auto h_xpts = d_xpts.createHostVec();
    int num_tip_nodes = 0;
    for (int inode = 0; inode < num_nodes; inode++) {
        if (abs(h_xpts[3*inode] - length) < 1e-6) {
            num_tip_nodes++;
        }
    }
    for (int inode = 0; inode < num_nodes; inode++) {
        if (abs(h_xpts[3*inode] - length) < 1e-6) {
            h_loads[6*inode+2] = beam_tip_force / num_tip_nodes;
        }
    }
    auto d_loads = h_loads.createDeviceVec();
    assembler.apply_bcs(d_loads);

    double loads_norm = CUBLAS::get_vec_norm(d_loads);
    printf("loads_norm = %.4e\n", loads_norm);


    // setup kmat, res, variables
    auto res = assembler.createVarsVec();
    auto rhs = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);


    // nonlinear static solve settings
    // -------------------------------
    // int num_load_steps = 1;
    int num_load_steps = 20;
    int num_newton = 30; //30;
    // double max_load_factor = 0.05;
    double max_load_factor = 1.0;

    // continuation solver
    // -------------------
    for (int load_step = 0; load_step < num_load_steps; load_step++) {
        double load_factor = max_load_factor * (load_step + 1) / num_load_steps;
        printf("load step %d, load_Factor = %.4e\n", load_step, load_factor);

        // Newton iteration nonlinear solve
        // --------------------------------
        for (int inewton = 0; inewton < num_newton; inewton++) {
            
            // compute internal residual and stiffness matrix
            assembler.set_variables(vars);
            assembler.add_jacobian(res, kmat);
            assembler.apply_bcs(res);
            assembler.apply_bcs(kmat);

            // compute the RHS
            rhs.zeroValues();
            CUBLAS::axpy(load_factor, d_loads, rhs);
            CUBLAS::axpy(-1.0, res, rhs);
            assembler.apply_bcs(rhs);
            double rhs_norm = CUBLAS::get_vec_norm(rhs);

            // printf("rhs[104] = %.4e\n", rhs[104]);

            // solve for the change in variables (soln = u - u0) and update variables
            soln.zeroValues();
            CUSPARSE::direct_LU_solve<T>(kmat, rhs, soln);
            double soln_norm = CUBLAS::get_vec_norm(soln);
            printf("\tnewton step %d, rhs = %.4e, soln = %.4e\n", inewton, rhs_norm, soln_norm);
            CUBLAS::axpy(1.0, soln, vars);


            // compute the residual (much cheaper computation on GPU)
            assembler.set_variables(vars);
            assembler.add_residual(res);
            rhs.zeroValues();
            CUBLAS::axpy(load_factor, d_loads, rhs);
            CUBLAS::axpy(-1.0, res, rhs);
            assembler.apply_bcs(rhs);
            double full_resid_norm = CUBLAS::get_vec_norm(rhs);
            printf("\t\tfull res = %.4e\n", full_resid_norm);
            // TODO : need residual check

            if (abs(full_resid_norm) < 1e-7) {
                break;
            }
        }

        std::stringstream filename;
        filename << "out/beam_" << load_step << ".vtk";

        auto h_vars = vars.createHostVec();
        printToVTK<Assembler, HostVec<T>>(assembler, h_vars, filename.str());
        
    }
    return 0;
};