// binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // for py::array_t
#include <pybind11/stl.h>  // for std::vector
#include "nl_wing.h"         // your solver class header

namespace py = pybind11;

PYBIND11_MODULE(nlwinggpu, m) {
    py::class_<NonlinearWingGPUSolver>(m, "NonlinearWingGPUSolver")
        .def(py::init<int, double, double, double, bool, bool, bool, bool, int, int, int, double, double>(),
             py::arg("level") = 2,
             py::arg("force") = 4.0e7,
             py::arg("omegaMC") = 0.7,
             py::arg("SR") = 10.0,
             py::arg("use_predictor") = true,
             py::arg("kmg_print") = false,
             py::arg("nl_debug") = false,
             py::arg("debug_gmg") = false,
             py::arg("nsmooth") = 4,
             py::arg("ninnercyc") = 2,
             py::arg("n_krylov") = 50,
             py::arg("omegaLS_min") = 0.5,
             py::arg("omegaLS_max") = 2.0)

        // ---- core nonlinear solvers ----
        .def("continuationSolve",
             [](NonlinearWingGPUSolver &s,
                py::array_t<double> u0,
                double lambda0 = 0.2,
                double lambdaf = 1.0,
                double inner_atol = 1e-8) {
                 py::array_t<double> uf(u0.size());
                 s.continuationSolve(u0.data(), uf.mutable_data(), lambda0, lambdaf, inner_atol);
                 return uf;
             },
             py::arg("u0"), py::arg("lambda0") = 0.2,
             py::arg("lambdaf") = 1.0, py::arg("inner_atol") = 1e-8)

        .def("vcycleSolve",
             [](NonlinearWingGPUSolver &s,
                py::array_t<double> u,
                double lambda,
                int n_cycles = 40) {
                 py::array_t<double> du(u.size());
                 s.vcycleSolve(u.data(), lambda, du.mutable_data(), n_cycles);
                 return du;
             },
             py::arg("u"), py::arg("lambda"),
             py::arg("n_cycles") = 40)

        .def("kcycleSolve",
             [](NonlinearWingGPUSolver &s,
                py::array_t<double> u,
                double lambda) {
                 py::array_t<double> du(u.size());
                 s.kcycleSolve(u.data(), lambda, du.mutable_data());
                 return du;
             },
             py::arg("u"), py::arg("lambda"))

        .def("getResidual",
             [](NonlinearWingGPUSolver &s,
                py::array_t<double> u,
                double lambda) {
                 py::array_t<double> res(u.size());
                 double val = s.getResidual(u.data(), lambda, res.mutable_data());
                 return py::make_tuple(val, res);
             },
             py::arg("u"), py::arg("lambda"))

        .def("setGridDefect",
             [](NonlinearWingGPUSolver &s,
                py::array_t<double> u,
                double lambda, bool set_fine_LU = false) {
                 s.setGridDefect(u.data(), lambda, set_fine_LU);
             },
             py::arg("u"), py::arg("lambda"), py::arg("set_fine_LU") = false)

        .def("getCoarseFineStep",
             [](NonlinearWingGPUSolver &s,
                py::array_t<double> i_defect,
                py::array_t<double> ism_defect,
                py::array_t<double> cf_soln,
                py::array_t<double> ch_defect,
                py::array_t<double> fsm_defect,
                py::array_t<double> lu_soln,
                bool smooth = true) {
                 s.getCoarseFineStep(i_defect.mutable_data(),
                                     ism_defect.mutable_data(),
                                     cf_soln.mutable_data(),
                                     ch_defect.mutable_data(),
                                     fsm_defect.mutable_data(),
                                     lu_soln.mutable_data(),
                                     smooth);
             },
             py::arg("i_defect"), py::arg("ism_defect"),
             py::arg("cf_soln"), py::arg("ch_defect"),
             py::arg("fsm_defect"), py::arg("lu_soln"),
             py::arg("smooth") = true)

        .def("writeSolution",
             [](NonlinearWingGPUSolver &s,
                const std::string &filename,
                py::array_t<double> u) {
                 s.writeSolution(filename, u.mutable_data());
             },
             py::arg("filename"), py::arg("u"))

        // ---- getters ----
        .def("get_num_vars", &NonlinearWingGPUSolver::get_num_vars)
        .def("free", &NonlinearWingGPUSolver::free);
}