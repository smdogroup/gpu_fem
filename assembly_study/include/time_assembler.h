#pragma once

template <typename T, class Data, class Assembler, bool is_jac = true>
void time_assembler() {
    bool print = true;
    bool mesh_print = false;
    TACSMeshLoader<T> mesh_loader{mesh_print};
    mesh_loader.scanBDFFile("../examples/performance/uCRM-135_wingbox_fine.bdf");

    double E = 70e9, nu = 0.3, thick = 0.02;  // material & thick properties

    // make the assembler from the uCRM mesh
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
    assembler.moveBsrDataToDevice();

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();

    // debug
    int nvars = assembler.get_num_vars();
    auto h_soln1 = soln.createHostVec();
    for (int i = 0; i < nvars; i++) {
        h_soln1[i] = 1.1343 + 2.3142 * i + 4.132 * i * i;
        h_soln1[i] *= 1e-6;
    }
    auto soln2 = h_soln1.createDeviceVec();
    assembler.set_variables(soln2);

    assembler.apply_bcs(res);  // warmup call
    printf("\n");

    if constexpr (is_jac) {
        assembler.add_jacobian(res, kmat, print);
    } else {
        assembler.add_residual(res, print);  // prints runtime in here
        // check residual not zero
        printf("\tcheck resid: ");
        auto h_res = res.createHostVec();
        printVec<T>(10, h_res.getPtr());
    }
}