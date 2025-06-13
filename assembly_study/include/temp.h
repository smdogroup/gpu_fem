
template <typename T, class Data, class Basis, class ElemGroup1, class ElemGroup2>
void test_drill_strain_resid(const int iquad, const T xpts[], const T vars[],
                             const Data &physData) {
    T resid1[24], resid2[24];
    for (int i = 0; i < 24; i++) {
        resid1[i] = 0.0;
        resid2[i] = 0.0;
    }

    // original resid computation with debug_mode = 1 for drill strains
    constexpr int debug_mode = 1;
    ElemGroup1::template add_element_quadpt_residual<Data, debug_mode>(true, iquad, xpts, vars,
                                                                       physData, resid1);

    // new resid computation
    // printf("\n----------\n");
    T Tmat[36], XdinvT[36];
    for (int inode = 0; inode < 4; inode++) {
        ShellComputeNodalTransforms<T, Data, Basis>(inode, xpts, physData, &Tmat[9 * inode],
                                                    &XdinvT[9 * inode]);
    }
    ElemGroup2::template _add_drill_strain_quadpt_residual_fast<Data>(
        true, iquad, xpts, vars, physData, Tmat, XdinvT, resid2);

    double err = rel_err(24, resid1, resid2);
    bool passed = err < 1e-10;
    printTestReport("drill strain resid", passed, err);
    // printf("\tresid1:");
    // printVec<T>(24, resid1);
    // printf("\tresid2:");
    // printVec<T>(24, resid2);
}

template <typename T, class Data, class Basis, class Quadrature, class ElemGroup1, class ElemGroup2>
void test_tying_strain_resid(const int iquad, const T xpts[], const T vars[],
                             const Data &physData) {
    T resid1[24], resid2[24];
    for (int i = 0; i < 24; i++) {
        resid1[i] = 0.0;
        resid2[i] = 0.0;
    }

    // original resid computation with debug_mode = 2 for tying strains
    constexpr int debug_mode = 2;
    ElemGroup1::template add_element_quadpt_residual<Data, debug_mode>(true, iquad, xpts, vars,
                                                                       physData, resid1);

    // new resid computation
    // printf("\n----------\n");
    T Tmat[9], XdinvT[9], XdinvzT[9];
    ShellComputeQuadptTransforms<T, Data, Basis, Quadrature>(iquad, xpts, physData, Tmat, XdinvT,
                                                             XdinvzT);
    ElemGroup2::template _add_tying_strain_quadpt_residual_fast<Data>(true, iquad, xpts, vars,
                                                                      physData, XdinvT, resid2);

    double err = rel_err(24, resid1, resid2);
    bool passed = err < 1e-10;
    printTestReport("tying strain resid", passed, err);
    // printf("\tresid1:");
    // printVec<T>(24, resid1);
    // printf("\tresid2:");
    // printVec<T>(24, resid2);
}

template <typename T, class Data, class Basis, class Quadrature, class ElemGroup1, class ElemGroup2>
void test_bending_strain_resid(const int iquad, const T xpts[], const T vars[],
                               const Data &physData) {
    T resid1[24], resid2[24];
    for (int i = 0; i < 24; i++) {
        resid1[i] = 0.0;
        resid2[i] = 0.0;
    }

    // original resid computation with debug_mode = 3 for bending strains
    constexpr int debug_mode = 3;
    ElemGroup1::template add_element_quadpt_residual<Data, debug_mode>(true, iquad, xpts, vars,
                                                                       physData, resid1);

    // new resid computation
    // printf("\n----------\n");
    T Tmat[9], XdinvT[9], XdinvzT[9];
    ShellComputeQuadptTransforms<T, Data, Basis, Quadrature>(iquad, xpts, physData, Tmat, XdinvT,
                                                             XdinvzT);
    ElemGroup2::template _add_bending_strain_quadpt_residual_fast<Data>(
        true, iquad, xpts, vars, physData, Tmat, XdinvT, XdinvzT, resid2);

    double err = rel_err(24, resid1, resid2);
    bool passed = err < 1e-10;
    printTestReport("bending strain resid", passed, err);
    // printf("\tresid1:");
    // printVec<T>(24, resid1);
    // printf("\tresid2:");
    // printVec<T>(24, resid2);
}