// binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::vector
#include "optim.h"         // your solver class header

namespace py = pybind11;

PYBIND11_MODULE(gpusolver, m) {
    py::class_<TacsGpuMultigridSolver>(m, "TacsGpuMultigridSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("load_mag") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0)
        .def("set_design_variables", &TacsGpuMultigridSolver::set_design_variables)
        .def("get_num_vars",         &TacsGpuMultigridSolver::get_num_vars)
        .def("get_num_dvs",          &TacsGpuMultigridSolver::get_num_dvs)
        .def("solve",                &TacsGpuMultigridSolver::solve)
        .def("writeSolution",         &TacsGpuMultigridSolver::writeSolution)
        .def("evalFunction",         &TacsGpuMultigridSolver::evalFunction)
        .def("evalFunctionSens",
             [](TacsGpuMultigridSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &TacsGpuMultigridSolver::free);
}