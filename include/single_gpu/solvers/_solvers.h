#pragma once

#ifdef USE_GPU
#ifdef USE_CUSPARSE

// utils
#include "linear_static/_utils.h"

// bsr matrix solvers
#include "linear_static/bsr_bicg_stab.h"
#include "linear_static/bsr_direct_LU.h"
#include "linear_static/bsr_gmres.h"
#include "linear_static/bsr_gmres_dr.h"
#include "linear_static/bsr_hgmres.h"
#include "linear_static/bsr_pcg.h"

// csr matrix solvers
#include "linear_static/csr_direct_chol.h"
#include "linear_static/csr_direct_chol2.h"
#include "linear_static/csr_gmres.h"

#endif  // CUSPARSE
#endif  // USE_GPU

// nonlinear static
#include "nonlinear_static/newton.h"

// dynamic
#include "dynamic/lin_gen_alpha.h"
#include "dynamic/nl_gen_alpha.h"