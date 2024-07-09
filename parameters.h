#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <iostream>
#include <vector>
#include <tuple>
#include <complex>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <cufft.h>

struct Params {
    float dt, dx, dy;
    int Nt, Nx, Ny;
    std::tuple<int, int, int> myu_size;
    std::tuple<float, float> myu_mstd;
    double* d_kx;
    double* d_ky;
    double* d_Kx;
    double* d_Ky;
    double* d_q;
    double* d_exponent;
    double* d_expm1;
    double* d_step_1;
    double* d_step_2;
    thrust::complex<double>* d_N_hat_prev;
    thrust::complex<double>* d_myu;
};

Params initialize_parameters(std::tuple<float, float, float> d,
    std::tuple<int, int, int> N,
    std::tuple<int, int, int> myu_size,
    std::tuple<float, float> myu_mstd);

void compute_myu(Params& params);
void print_parameters(const Params& params);
void free_parameters(Params& params);
void compute_state(thrust::complex<double>* d_myu, Params& params);

#endif // PARAMETERS_H
