#pragma once
#include "assembler.h"
#include "linalg/linalg.h"

template <class Assembler>
Assembler createFakeAssembler(int num_bcs, int num_elements, int num_geo_nodes,
                              int num_vars_nodes) {

    using T = typename Assembler::T;
    using Basis = typename Assembler::Basis;
    using Geo = typename Assembler::Geo;
    using Data = typename Assembler::Data;

    HostVec<int> bcs(num_bcs);
    bcs.randomize(num_vars_nodes);

    // need pointer here otherwise goes out of scope
    int N = Geo::num_nodes * num_elements;
    HostVec<int32_t> geo_conn(N);
    geo_conn.randomize(num_geo_nodes);
    make_unique_conn(num_elements, Geo::num_nodes, num_geo_nodes,
                     geo_conn.getPtr());

    int N2 = Basis::num_nodes * num_elements;
    HostVec<int32_t> vars_conn(N2);
    vars_conn.randomize(num_vars_nodes);
    make_unique_conn(num_elements, Basis::num_nodes, num_vars_nodes,
                     vars_conn.getPtr());

    // set the xpts randomly for this example
    int32_t num_xpts = Geo::spatial_dim * num_geo_nodes;
    HostVec<T> xpts(num_xpts);
    xpts.randomize();

    double E = 70e9, nu = 0.3, t = 0.005; // aluminum plate
    HostVec<Data> physData(num_elements, Data(E, nu, t));

    // make the assembler
    Assembler assembler(num_geo_nodes, num_vars_nodes, num_elements, geo_conn,
                        vars_conn, xpts, bcs, physData);

    return assembler;
}