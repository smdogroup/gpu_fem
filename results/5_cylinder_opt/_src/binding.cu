// binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::vector
#include "lin_optim.h"         // your solver class header
#include "nl_optim.h"        

namespace py = pybind11;

PYBIND11_MODULE(cylindermultigrid, m) {
    py::class_<LinearCylinderSolver>(m, "LinearCylinderSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double, double, double, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("pressure") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0,
             py::arg("ORDER") = 8,
             py::arg("in_plane_frac") = 0.2,
             py::arg("print") = false)
        .def("set_design_variables", &LinearCylinderSolver::set_design_variables)
        .def("get_num_vars",         &LinearCylinderSolver::get_num_vars)
        .def("get_num_dvs",          &LinearCylinderSolver::get_num_dvs)
        .def("get_num_lin_solves",          &LinearCylinderSolver::get_num_lin_solves)
        .def("solve",                &LinearCylinderSolver::solve)
        .def("writeSolution",         &LinearCylinderSolver::writeSolution)
        .def("evalFunction",         &LinearCylinderSolver::evalFunction)
        .def("evalFunctionSens",
             [](LinearCylinderSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &LinearCylinderSolver::free);

    py::class_<NonlinearCylinderSolver>(m, "NonlinearCylinderSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double, double, int, int, double, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("pressure") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0,
             py::arg("ORDER") = 8,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("in_plane_frac") = 0.2,
             py::arg("print") = false)
        .def("set_design_variables", &NonlinearCylinderSolver::set_design_variables)
        .def("get_num_vars",         &NonlinearCylinderSolver::get_num_vars)
        .def("get_num_dvs",          &NonlinearCylinderSolver::get_num_dvs)
        .def("get_num_lin_solves",          &NonlinearCylinderSolver::get_num_lin_solves)
        .def("solve",                &NonlinearCylinderSolver::solve)
        .def("writeSolution",         &NonlinearCylinderSolver::writeSolution)
        .def("evalFunction",         &NonlinearCylinderSolver::evalFunction)
        .def("evalFunctionSens",
             [](NonlinearCylinderSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &NonlinearCylinderSolver::free);
}