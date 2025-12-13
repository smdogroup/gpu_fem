#include "assembler.h"
#include "plane_stress/plane_stress.h"

int main(void) {
    using T = double;

    const A2D::GreenStrainType strain = A2D::GreenStrainType::LINEAR;

    using Quad = TriangleQuadrature<T>;
    using Geo = LinearTriangleGeo<T,Quad>;
    using Basis = QuadraticTriangleBasis<T,Quad>;
    using Physics = PlaneStressPhysics<T,Quad,strain>;
    using Group = PlaneStressElementGroup<T, Geo, Basis, Physics>;
    using Data = typename Physics::IsotropicData;
    using Assembler = ElementAssembler<T, Group>;

    int num_geo_nodes = 500;
    int num_vars_nodes = 800;
    int num_elements = 1000;

    // make fake element connectivity for testing
    int N = Geo::num_nodes * num_elements;
    int32_t *geo_conn = new int32_t[N];
    for (int i = 0; i < N; i++) {
      geo_conn[i] = rand() % num_geo_nodes;
    }

    // randomly generate the connectivity for the variables / basis
    int N2 = Basis::num_nodes * num_elements;
    int32_t *vars_conn = new int32_t[N2];
    for (int i = 0; i < N2; i++) {
      vars_conn[i] = rand() % num_vars_nodes;
    }

    // set the xpts randomly for this example
    int32_t num_xpts = Geo::spatial_dim * num_geo_nodes;
    T *xpts = new T[num_xpts];
    for (int ixpt = 0; ixpt < num_xpts; ixpt++) {
      xpts[ixpt] = static_cast<double>(rand()) / RAND_MAX;
    }

    // initialize ElemData
    double E = 70e9, nu = 0.3, t = 0.005; // aluminum plate
    Data elemData[num_elements];
    for (int ielem = 0; ielem < num_elements; ielem++) {
        elemData[ielem] = Data(E, nu, t);
    }

    // make the assembler
    Assembler assembler(num_geo_nodes, num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, elemData);

    // define variables here for testing different vars inputs
    // set some host data to zero
    int32_t num_vars = assembler.get_num_vars();
    T *h_vars = new T[num_vars];
    memset(h_vars, 0.0, num_vars * sizeof(T));

    bool nz_vars = true;
    if (nz_vars) {
      for (int ivar = 0; ivar < num_vars; ivar++) {
        h_vars[ivar] = static_cast<double>(rand()) / RAND_MAX;
      }
    }

    #ifdef USE_GPU
    T *d_vars;
    cudaMalloc((void**)&d_vars, num_vars * sizeof(T));
    cudaMemcpy(d_vars, h_vars, num_vars * sizeof(T), cudaMemcpyHostToDevice);   
    assembler.set_variables(d_vars);
    #else // USE_GPU
    assembler.set_variables(h_vars);
    #endif
    

    // define the residual vector (host or device)
    T *h_residual = new T[num_vars];
    memset(h_residual, 0.0, num_vars * sizeof(T));
    #ifdef USE_GPU
    T *d_residual;
    cudaMalloc((void**)&d_residual, num_vars * sizeof(T));
    cudaMemset(d_residual, 0.0, num_vars * sizeof(T));     
    #endif

    #ifdef USE_GPU
    assembler.add_residual(d_residual);
    cudaMemcpy(h_residual, d_residual, num_vars * sizeof(T), cudaMemcpyDeviceToHost);
    #else
    assembler.add_residual(h_residual);
    #endif

    // print data of host residual
    int M = 10;
    for (int i = 0; i < M; i++) {
      printf("res[%d] = %.8e\n", i, h_residual[i]);
    }

    return 0;
}