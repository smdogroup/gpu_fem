#include "assembler.h"
#include "shell/shell.h"
#include "chrono"

int main(void) {
    using T = double;

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T,has_ref_axis>;
    using Physics = IsotropicShell<T, Data>;
    
    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>; 
    using Assembler = ElementAssembler<T, ElemGroup>;

    printf("running!\n");

    // int num_elements = 1;
    // int num_geo_nodes = 4;
    // int num_vars_nodes = 4;

    int num_geo_nodes = 1e2;
    int num_vars_nodes = 1e2;
    int num_elements = 1e3;

    // int num_geo_nodes = 3e3;
    // int num_vars_nodes = num_geo_nodes;
    // int num_elements = 1e5;

    // make fake element connectivity for testing
    int N = Geo::num_nodes * num_elements;
    int32_t *geo_conn = new int32_t[N];
    for (int i = 0; i < N; i++) {
      geo_conn[i] = i % num_geo_nodes; 
    }

    // randomly generate the connectivity for the variables / basis
    int N2 = Basis::num_nodes * num_elements;
    int32_t *vars_conn = new int32_t[N2];
    for (int i = 0; i < N2; i++) {
      vars_conn[i] = i % num_vars_nodes;
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

    int num_vars2 = num_vars*num_vars;
    T *h_mat = new T[num_vars2];
    memset(h_mat, 0.0, num_vars2 * sizeof(T));
    #ifdef USE_GPU
    T *d_mat;
    cudaMalloc((void**)&d_mat, num_vars2 * sizeof(T));
    cudaMemset(d_mat, 0.0, num_vars2 * sizeof(T));     
    #endif

    // time add residual method
    auto start = std::chrono::high_resolution_clock::now();

    // call add jacobian
    #ifdef USE_GPU
    assembler.add_jacobian(d_residual, d_mat);
    cudaMemcpy(h_residual, d_residual, num_vars * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mat, d_mat, num_vars * sizeof(T), cudaMemcpyDeviceToHost);
    #else
    assembler.add_jacobian(h_residual, h_mat);
    #endif

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // print data of host mat
    for (int i = 0; i < 24*24; i++) {
      printf("K[%d] = %.8e\n", i, h_mat[i]);
    }

    printf("took %d microseconds to run add jacobian\n", (int)duration.count());

    return 0;
};