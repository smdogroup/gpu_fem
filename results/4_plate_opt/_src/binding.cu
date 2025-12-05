// binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::vector
#include "lin_optim.h"         // your solver class header
#include "nl_optim.h"        

namespace py = pybind11;

PYBIND11_MODULE(platemultigrid, m) {
    py::class_<LinearPlateSolver>(m, "LinearPlateSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double, double, int, double, int, int, double, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("load_mag") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("Lx") = 1.0,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("in_plane_frac") = 0.1,
             py::arg("print") = false)
        .def("set_design_variables", &LinearPlateSolver::set_design_variables)
        .def("get_num_vars",         &LinearPlateSolver::get_num_vars)
        .def("get_num_dvs",          &LinearPlateSolver::get_num_dvs)
        .def("get_num_lin_solves",          &LinearPlateSolver::get_num_lin_solves)
        .def("solve",                &LinearPlateSolver::solve)
        .def("writeSolution",         &LinearPlateSolver::writeSolution)
        .def("evalFunction",         &LinearPlateSolver::evalFunction)
        .def("evalFunctionSens",
             [](LinearPlateSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &LinearPlateSolver::free);

    py::class_<NonlinearPlateSolver>(m, "NonlinearPlateSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double, double, double, int, int, double, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("load_mag") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0,
             py::arg("ORDER") = 8,
             py::arg("Lx") = 1.0,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("in_plane_frac") = 0.1,
             py::arg("print") = false)
        .def("set_design_variables", &NonlinearPlateSolver::set_design_variables)
        .def("get_num_vars",         &NonlinearPlateSolver::get_num_vars)
        .def("get_num_dvs",          &NonlinearPlateSolver::get_num_dvs)
        .def("get_num_lin_solves",          &NonlinearPlateSolver::get_num_lin_solves)
        .def("solve",                &NonlinearPlateSolver::solve)
        .def("writeSolution",         &NonlinearPlateSolver::writeSolution)
        .def("evalFunction",         &NonlinearPlateSolver::evalFunction)
        .def("evalFunctionSens",
             [](NonlinearPlateSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &NonlinearPlateSolver::free);
}