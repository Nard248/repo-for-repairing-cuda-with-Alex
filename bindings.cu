#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <complex>
#include "parameters.h"

namespace py = pybind11;

void initialize_parameters_py(float dt, float dx, float dy, int Nt, int Nx, int Ny, std::tuple<int, int, int> myu_size, std::tuple<float, float> myu_mstd, Params& params) {
    std::tuple<float, float, float> d = { dt, dx, dy };
    std::tuple<int, int, int> N = { Nt, Nx, Ny };
    params = initialize_parameters(d, N, myu_size, myu_mstd);
}

void compute_state_py(py::array_t<std::complex<double>> d_myu_np, Params& params) {
    auto buf = d_myu_np.request();
    std::complex<double>* ptr = static_cast<std::complex<double>*>(buf.ptr);

    // Convert std::complex<double> to thrust::complex<double>
    thrust::host_vector<thrust::complex<double>> h_myu(buf.size);
    for (size_t i = 0; i < buf.size; ++i) {
        h_myu[i] = thrust::complex<double>(ptr[i].real(), ptr[i].imag());
    }

    // Allocate device memory and copy the data
    thrust::device_vector<thrust::complex<double>> d_myu = h_myu;

    // Call the compute_state function
    compute_state(thrust::raw_pointer_cast(d_myu.data()), params);
}

PYBIND11_MODULE(simulation, m) {
    py::class_<Params>(m, "Params")
        .def(py::init<>())
        .def_readwrite("dt", &Params::dt)
        .def_readwrite("dx", &Params::dx)
        .def_readwrite("dy", &Params::dy)
        .def_readwrite("Nt", &Params::Nt)
        .def_readwrite("Nx", &Params::Nx)
        .def_readwrite("Ny", &Params::Ny)
        .def_readwrite("d_kx", &Params::d_kx)
        .def_readwrite("d_ky", &Params::d_ky)
        .def_readwrite("d_Kx", &Params::d_Kx)
        .def_readwrite("d_Ky", &Params::d_Ky)
        .def_readwrite("d_q", &Params::d_q)
        .def_readwrite("d_exponent", &Params::d_exponent)
        .def_readwrite("d_expm1", &Params::d_expm1)
        .def_readwrite("d_step_1", &Params::d_step_1)
        .def_readwrite("d_step_2", &Params::d_step_2)
        .def_readwrite("d_N_hat_prev", &Params::d_N_hat_prev)
        .def_readwrite("d_myu", &Params::d_myu);

    m.def("initialize_parameters", &initialize_parameters_py, "Initialize parameters");
    m.def("compute_state", &compute_state_py, "Compute state");
    m.def("compute_myu", &compute_myu, "Compute myu");
    m.def("print_parameters", &print_parameters, "Print parameters");
    m.def("free_parameters", &free_parameters, "Free parameters");
}
