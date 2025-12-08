// binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::vector
#include "lin_cfi_optim.h"         // your solver class header
#include "lin_mitc_optim.h"
#include "lin_stiff_optim.h"
// #include "nl_optim.h"        

namespace py = pybind11;

PYBIND11_MODULE(wingmultigrid, m) {
    py::class_<LinearCFIWingSolver>(m, "LinearCFIWingSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, double, int, int, int, bool, int>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("force") = 684e3,
             py::arg("omega") = 0.8,
             py::arg("level") = 2,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("print") = false,
             py::arg("n_krylov") = 50)
        .def("set_design_variables", &LinearCFIWingSolver::set_design_variables)
        .def("get_num_vars",         &LinearCFIWingSolver::get_num_vars)
        .def("get_num_dvs",          &LinearCFIWingSolver::get_num_dvs)
        .def("get_num_lin_solves",          &LinearCFIWingSolver::get_num_lin_solves)
        .def("solve",                &LinearCFIWingSolver::solve)
        .def("writeSolution",         &LinearCFIWingSolver::writeSolution)
        .def("writeLoadsToVTK",         &LinearCFIWingSolver::writeLoadsToVTK)
        .def("writeExplodedVTKs",         &LinearCFIWingSolver::writeExplodedVTKs)
        .def("evalFunction",         &LinearCFIWingSolver::evalFunction)
        .def("evalFunctionSens",
             [](LinearCFIWingSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &LinearCFIWingSolver::free);

    py::class_<LinearMITCWingSolver>(m, "LinearMITCWingSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, double, int, int, int, bool, int>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("force") = 684e3,
             py::arg("omega") = 0.8,
             py::arg("level") = 2,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("print") = false,
             py::arg("n_krylov") = 50)
        .def("set_design_variables", &LinearMITCWingSolver::set_design_variables)
        .def("get_num_vars",         &LinearMITCWingSolver::get_num_vars)
        .def("get_num_dvs",          &LinearMITCWingSolver::get_num_dvs)
        .def("get_num_lin_solves",          &LinearMITCWingSolver::get_num_lin_solves)
        .def("solve",                &LinearMITCWingSolver::solve)
        .def("writeSolution",         &LinearMITCWingSolver::writeSolution)
        .def("writeLoadsToVTK",         &LinearMITCWingSolver::writeLoadsToVTK)
        .def("writeExplodedVTKs",         &LinearMITCWingSolver::writeExplodedVTKs)
        .def("evalFunction",         &LinearMITCWingSolver::evalFunction)
        .def("evalFunctionSens",
             [](LinearMITCWingSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &LinearMITCWingSolver::free);

    py::class_<LinearStiffenedWingSolver>(m, "LinearStiffenedWingSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, double, int, int, int, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("force") = 684e3,
             py::arg("omega") = 0.8,
             py::arg("level") = 2,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("print") = false)
        .def("set_design_variables", &LinearStiffenedWingSolver::set_design_variables)
        .def("get_num_vars",         &LinearStiffenedWingSolver::get_num_vars)
        .def("get_num_dvs",          &LinearStiffenedWingSolver::get_num_dvs)
        .def("get_num_lin_solves",          &LinearStiffenedWingSolver::get_num_lin_solves)
        .def("solve",                &LinearStiffenedWingSolver::solve)
        .def("writeSolution",         &LinearStiffenedWingSolver::writeSolution)
        .def("evalFunction",         &LinearStiffenedWingSolver::evalFunction)
        .def("evalFunctionSens",
             [](LinearStiffenedWingSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &LinearStiffenedWingSolver::free);

    // py::class_<NonlinearPlateSolver>(m, "NonlinearPlateSolver")
    //     // only expose (rhoKS, load_mag)
    //     .def(py::init<double, double, double, double, int, int, int, double, double, double, int, int, double, bool>(),
    //          py::arg("rhoKS")    = 100.0,
    //          py::arg("safety_factor") = 1.5,
    //          py::arg("load_mag") = 100.0,
    //          py::arg("omega") = 1.0,
    //          py::arg("nxe") = 100,
    //          py::arg("nx_comp") = 5,
    //          py::arg("ny_comp") = 5,
    //          py::arg("SR") = 50.0,
    //          py::arg("ORDER") = 8,
    //          py::arg("Lx") = 1.0,
    //          py::arg("nsmooth") = 1,
    //          py::arg("ninnercyc") = 1,
    //          py::arg("in_plane_frac") = 0.1,
    //          py::arg("print") = false)
    //     .def("set_design_variables", &NonlinearPlateSolver::set_design_variables)
    //     .def("get_num_vars",         &NonlinearPlateSolver::get_num_vars)
    //     .def("get_num_dvs",          &NonlinearPlateSolver::get_num_dvs)
    //     .def("get_num_lin_solves",          &NonlinearPlateSolver::get_num_lin_solves)
    //     .def("solve",                &NonlinearPlateSolver::solve)
    //     .def("writeSolution",         &NonlinearPlateSolver::writeSolution)
    //     .def("evalFunction",         &NonlinearPlateSolver::evalFunction)
    //     .def("evalFunctionSens",
    //          [](NonlinearPlateSolver &s, const std::string &name) {
    //              std::vector<double> sens(s.get_num_dvs());
    //              s.evalFunctionSens(name, sens.data());
    //              return sens;
    //          })
    //     .def("free", &NonlinearPlateSolver::free);
}