// dependencies to make element, constitutive objects
#include "TACSIsoShellConstitutive.h"
#include "TACSMaterialProperties.h"
#include "TACSShellElementDefs.h"
#include "TACSShellElementTransform.h"

template <typename T>
void printVec(const int N, const T *vec);

template <>
void printVec<double>(const int N, const double *vec) {
    for (int i = 0; i < N; i++) {
        printf("%.5e,", vec[i]);
    }
    printf("\n");
}

// this example is based off of examples/crm/crm.cpp in TACS

int main(int argc, char *argv[]) {
    // make the MPI communicator
    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    MPI_Comm_rank(comm, &rank);

    std::string nonlinear_str;
    // bool nonlinear = false;
    bool nonlinear = true;
    if (argc == 2) {
        nonlinear_str = argv[1];
        nonlinear = nonlinear_str == "nonlinear";
    }

    using T = TacsScalar;

    TACSShellTransform *transform = new TACSShellNaturalTransform();

    // set material properties for aluminum (no thermal props input this time)
    TacsScalar rho = 2718;
    TacsScalar specific_heat = 0.0;
    TacsScalar E = 70.0e9;
    TacsScalar nu = 0.33;
    TacsScalar ys = 1e11;
    TacsScalar cte = 10e-6;  // double check this value
    TacsScalar kappa = 0.0;
    TACSMaterialProperties *mat =
        new TACSMaterialProperties(rho, specific_heat, E, nu, ys, cte, kappa);

    // create the one element data
    int num_elements = 1;
    int num_geo_nodes = 4;
    int num_vars_nodes = 4;

    // set the xpts randomly for this example
    int32_t num_xpts = 12;
    T *xpts = new T[num_xpts];
    for (int ixpt = 0; ixpt < num_xpts; ixpt++) {
        xpts[ixpt] = 1.0345452 + 2.23123432 * ixpt + 0.323 * ixpt * ixpt;
    }

    // set vars randomly for this example
    int32_t num_vars = 24;
    T *vars = new T[num_vars];
    memset(vars, 0.0, num_vars * sizeof(T));
    T *dvars = new T[num_vars];
    memset(dvars, 0.0, num_vars * sizeof(T));
    T *ddvars = new T[num_vars];
    memset(ddvars, 0.0, num_vars * sizeof(T));

    T *res = new T[num_vars];
    memset(res, 0.0, num_vars * sizeof(T));
    T *Kmat = new T[num_vars * num_vars];
    memset(Kmat, 0.0, num_vars * num_vars * sizeof(T));

    bool nz_vars = true;
    if (nz_vars) {
        for (int ivar = 0; ivar < num_vars; ivar++) {
            vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
            vars[ivar] *= 1e6;
        }
    }

    T *p_vars = new T[num_vars];
    memset(p_vars, 0.0, num_vars * sizeof(T));
    T *p_vars2 = new T[num_vars];
    memset(p_vars2, 0.0, num_vars * sizeof(T));
    for (int ivar = 0; ivar < 24; ivar++) {
        p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
        p_vars2[ivar] = (-1.4543 * 1.024343 + 2.812 * -9.4323 * ivar);
    }

    TacsScalar thick = 0.005;  // shell thickness
    TACSIsoShellConstitutive *con = new TACSIsoShellConstitutive(mat, thick);

    // now create the shell element object
    TACSElement *elem;
    if (nonlinear) {
        printf("nonlinear CPU shell\n");
        elem = new TACSQuad4NonlinearShell(transform, con);
    } else {
        printf("linear CPU shell\n");
        elem = new TACSQuad4Shell(transform, con);
    }

    // call compute energies on the one element
    int elemIndex = 0;
    double time = 0.0;
    // printf("TACS Ue 1 = %.8e\n", *Ue);
    elem->addJacobian(elemIndex, time, 1.0, 0.0, 0.0, xpts, vars, dvars, ddvars, res, Kmat);

    // take a vec-mat-vec product on Kmat
    double jac_TD = 0.0;
    double res_TD = 0.0;
    for (int i = 0; i < 24; i++) {
        res_TD += p_vars[i] * res[i];
        for (int j = 0; j < 24; j++) {
            jac_TD += p_vars[i] * p_vars2[j] * Kmat[24 * i + j];
        }
    }

    printf("Analytic jacobian CPU\n");
    printf("res TD = %.8e\n", res_TD);
    printf("jac TD = %.8e\n", jac_TD);

    printf("res: ");
    printVec<double>(24, res);

    printf("kmat:\n");
    for (int i = 0; i < 24; i++) {  // i < 2
        // printf("kmat row %d: ", i);
        for (int j = 0; j < 24; j++) {
            printf("%.14e,", Kmat[24 * i + j]);
        }
        // printf("\n");
    }

    MPI_Finalize();

    return 0;
}