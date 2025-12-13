// binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::vector
#include "optim.h"         // your solver class header

namespace py = pybind11;

PYBIND11_MODULE(gpusolver, m) {
    py::class_<TACSGPUSolver>(m, "TACSGPUSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("load_mag") = 100.0)
        .def("set_design_variables", &TACSGPUSolver::set_design_variables)
        .def("get_num_vars",         &TACSGPUSolver::get_num_vars)
        .def("get_num_dvs",          &TACSGPUSolver::get_num_dvs)
        .def("solve",                &TACSGPUSolver::solve)
        .def("writeSolution",         &TACSGPUSolver::writeSolution)
        .def("evalFunction",         &TACSGPUSolver::evalFunction)
        .def("evalFunctionSens",
             [](TACSGPUSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &TACSGPUSolver::free);
}