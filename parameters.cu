#include <cufft.h>
#include "parameters.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <vector>
#include <thrust/complex.h>
#include <cufftXt.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ thrust::complex<double> non_linear_function(thrust::complex<double> xx, thrust::complex<double> yy) {
    return xx * (yy - thrust::abs(xx) * thrust::abs(xx));
}

__global__ void compute_N_hat(thrust::complex<double>* A, thrust::complex<double>* myu, thrust::complex<double>* N_hat, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx && j < Ny) {
        int idx = i * Ny + j;
        N_hat[idx] = non_linear_function(A[idx], myu[idx]);
    }
}

__global__ void next_state(thrust::complex<double>* result, thrust::complex<double>* A_hat, thrust::complex<double>* N_hat, Params params, int order) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < params.Nx * params.Ny) {
        int i = idx / params.Ny;
        int j = idx % params.Ny;
        if (order == 1 || params.d_N_hat_prev == nullptr) {
            params.d_N_hat_prev[idx] = N_hat[idx];
            result[idx] = A_hat[idx] * params.d_exponent[idx] + N_hat[idx] * params.d_step_1[idx];
        }
        else {
            result[idx] = A_hat[idx] * params.d_exponent[idx] + N_hat[idx] * params.d_step_1[idx] - (N_hat[idx] - params.d_N_hat_prev[idx]) * params.d_step_2[idx] * 0.01;
            params.d_N_hat_prev[idx] = N_hat[idx];
        }
    }
}

__global__ void apply_nonlinear_function(thrust::complex<double>* A, thrust::complex<double>* myu, thrust::complex<double>* result, int Nx, int Ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nx * Ny) {
        int i = idx / Ny;
        int j = idx % Ny;
        thrust::complex<double> xx = A[idx];
        thrust::complex<double> yy = myu[idx];
        result[idx] = xx * (yy - thrust::abs(xx) * thrust::abs(xx));
    }
}

void compute_state(thrust::complex<double>* d_myu, Params& params) {
    thrust::complex<double>* d_A;
    thrust::complex<double>* d_N_hat;
    cudaError_t err;

    std::cout << "Allocating memory for d_A..." << std::endl;
    err = cudaMalloc(&d_A, params.Nt * params.Nx * params.Ny * sizeof(thrust::complex<double>));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_A: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "Allocating memory for d_N_hat..." << std::endl;
    err = cudaMalloc(&d_N_hat, params.Nx * params.Ny * sizeof(thrust::complex<double>));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_N_hat: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        return;
    }

    thrust::host_vector<thrust::complex<double>> h_A_0(params.Nx * params.Ny);
    for (int i = 0; i < params.Nx * params.Ny; ++i) {
        double real_part = std::rand() / (double)RAND_MAX * 0.01;
        double imag_part = std::rand() / (double)RAND_MAX * 0.01;
        h_A_0[i] = thrust::complex<double>(real_part, imag_part);
    }
    std::cout << "Copying data to d_A..." << std::endl;
    err = cudaMemcpy(d_A, h_A_0.data(), params.Nx * params.Ny * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data to d_A: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_N_hat);
        return;
    }

    cufftHandle plan;
    std::cout << "Creating FFT plan..." << std::endl;
    if (cufftPlan2d(&plan, params.Nx, params.Ny, CUFFT_Z2Z) != CUFFT_SUCCESS) {
        std::cerr << "Error creating FFT plan" << std::endl;
        cudaFree(d_A);
        cudaFree(d_N_hat);
        return;
    }

    for (int t = 1; t < params.Nt; ++t) {
        std::cout << "Time step " << t << "..." << std::endl;

        int gridSize = (params.Nx * params.Ny + 255) / 256;
        apply_nonlinear_function << <gridSize, 256 >> > (d_A + (t - 1) * params.Nx * params.Ny, d_myu, d_N_hat, params.Nx, params.Ny);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Error in apply_nonlinear_function kernel: " << cudaGetErrorString(err) << std::endl;
            break;
        }

        std::cout << "Executing FFT on A..." << std::endl;
        if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(d_A + (t - 1) * params.Nx * params.Ny), reinterpret_cast<cufftDoubleComplex*>(d_A + t * params.Nx * params.Ny), CUFFT_FORWARD) != CUFFT_SUCCESS) {
            std::cerr << "Error executing FFT on A" << std::endl;
            break;
        }

        std::cout << "Executing FFT on N_hat..." << std::endl;
        if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(d_N_hat), reinterpret_cast<cufftDoubleComplex*>(d_N_hat), CUFFT_FORWARD) != CUFFT_SUCCESS) {
            std::cerr << "Error executing FFT on N_hat" << std::endl;
            break;
        }

        std::cout << "Starting time-stepping operation..." << std::endl;
        next_state << <gridSize, 256 >> > (d_A + t * params.Nx * params.Ny, d_A + (t - 1) * params.Nx * params.Ny, d_N_hat, params, t);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Error in next_state kernel: " << cudaGetErrorString(err) << std::endl;
            break;
        }
    }

    std::cout << "Destroying FFT plan..." << std::endl;
    if (cufftDestroy(plan) != CUFFT_SUCCESS) {
        std::cerr << "Error destroying FFT plan" << std::endl;
    }

    std::cout << "Freeing memory..." << std::endl;
    cudaFree(d_A);
    cudaFree(d_N_hat);
}

__device__ double custom_expm1(double x) {
    if (fabs(x) < 1e-5) {
        return x + 0.5 * x * x; // Use Taylor series approximation for better precision with small x
    }
    else {
        return exp(x) - 1.0;
    }
}

__global__ void initialize_kx_ky(double* kx, double* ky, int Nx, int Ny, double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nx) {
        kx[i] = (i < Nx / 2) ? i / (dx * Nx) : (i - Nx) / (dx * Nx);
    }
    if (i < Ny) {
        ky[i] = (i < Ny / 2) ? i / (dy * Ny) : (i - Ny) / (dy * Ny);
    }
}

__global__ void initialize_meshgrid(double* Kx, double* Ky, double* kx, double* ky, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx && j < Ny) {
        Kx[i * Ny + j] = kx[i];
        Ky[i * Ny + j] = ky[j];
    }
}

__global__ void initialize_params(double* q, double* exponent, double* expm1, double* step_1, double* step_2, double* Kx, double* Ky, float dt, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx && j < Ny) {
        int idx = i * Ny + j;
        double q_val = 1e-6 - 4.0 * M_PI * M_PI * (Kx[idx] * Kx[idx] + Ky[idx] * Ky[idx]);
        q[idx] = q_val;
        exponent[idx] = exp(q_val * dt);
        expm1[idx] = custom_expm1(q_val * dt); // Use custom_expm1
        step_1[idx] = expm1[idx] / q_val;
        step_2[idx] = (expm1[idx] - dt * q_val) / (dt * q_val * q_val);
    }
}

Params initialize_parameters(std::tuple<float, float, float> d,
    std::tuple<int, int, int> N,
    std::tuple<int, int, int> myu_size,
    std::tuple<float, float> myu_mstd) {
    Params params;
    params.dt = std::get<0>(d);
    params.dx = std::get<1>(d);
    params.dy = std::get<2>(d);
    params.Nt = std::get<0>(N);
    params.Nx = std::get<1>(N);
    params.Ny = std::get<2>(N);
    params.myu_size = myu_size;
    params.myu_mstd = myu_mstd;

    cudaMalloc(&params.d_kx, params.Nx * sizeof(double));
    cudaMalloc(&params.d_ky, params.Ny * sizeof(double));

    int blockSize = 256;
    int numBlocksX = (params.Nx + blockSize - 1) / blockSize;
    int numBlocksY = (params.Ny + blockSize - 1) / blockSize;

    initialize_kx_ky << <numBlocksX, blockSize >> > (params.d_kx, params.d_ky, params.Nx, params.Ny, params.dx, params.dy);

    cudaMalloc(&params.d_Kx, params.Nx * params.Ny * sizeof(double));
    cudaMalloc(&params.d_Ky, params.Nx * params.Ny * sizeof(double));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((params.Nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (params.Ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initialize_meshgrid << <numBlocks, threadsPerBlock >> > (params.d_Kx, params.d_Ky, params.d_kx, params.d_ky, params.Nx, params.Ny);

    cudaMalloc(&params.d_q, params.Nx * params.Ny * sizeof(double));
    cudaMalloc(&params.d_exponent, params.Nx * params.Ny * sizeof(double));
    cudaMalloc(&params.d_expm1, params.Nx * params.Ny * sizeof(double));
    cudaMalloc(&params.d_step_1, params.Nx * params.Ny * sizeof(double));
    cudaMalloc(&params.d_step_2, params.Nx * params.Ny * sizeof(double));

    initialize_params << <numBlocks, threadsPerBlock >> > (params.d_q, params.d_exponent, params.d_expm1, params.d_step_1, params.d_step_2, params.d_Kx, params.d_Ky, params.dt, params.Nx, params.Ny);

    cudaMalloc(&params.d_N_hat_prev, params.Nx * params.Ny * sizeof(thrust::complex<double>));
    cudaMemset(params.d_N_hat_prev, 0, params.Nx * params.Ny * sizeof(thrust::complex<double>));

    return params;
}

std::vector<double> kron(const std::vector<double>& A, const std::vector<double>& B, int Ax, int Ay, int Bx, int By) {
    std::vector<double> result(Ax * Bx * Ay * By);
    for (int i = 0; i < Ax; ++i) {
        for (int j = 0; j < Ay; ++j) {
            for (int k = 0; k < Bx; ++k) {
                for (int l = 0; l < By; ++l) {
                    result[(i * Bx + k) * (Ay * By) + (j * By + l)] = A[i * Ay + j] * B[k * By + l];
                }
            }
        }
    }
    return result;
}

void initialize_random_seed(unsigned int seed) {
    std::srand(seed);
}

void compute_myu(Params& params) {
    initialize_random_seed(20); // Initialize the seed to 20

    int myu_size_x = std::get<0>(params.myu_size);
    int myu_size_y = std::get<1>(params.myu_size);
    int myu_size_z = std::get<2>(params.myu_size);

    std::vector<double> myu_small(myu_size_x * myu_size_y * myu_size_z);
    for (double& x : myu_small) {
        x = std::abs(std::rand() / (double)RAND_MAX * std::get<1>(params.myu_mstd) + std::get<0>(params.myu_mstd));
    }

    // Debug print: Verify contents of myu_small
    std::cout << "myu_small: ";
    for (const auto& val : myu_small) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    int scale_x = params.Nt / myu_size_x;
    int scale_y = params.Nx / myu_size_y;
    int scale_z = params.Ny / myu_size_z;

    // Allocate memory for the final myu array
    int myu_size_total = params.Nt * params.Nx * params.Ny;
    std::vector<double> myu(myu_size_total);

    // Correctly fill myu with scaled values
    for (int t = 0; t < params.Nt; ++t) {
        for (int x = 0; x < params.Nx; ++x) {
            for (int y = 0; y < params.Ny; ++y) {
                int idx_small = (t / scale_x) * (myu_size_y * myu_size_z) + (x / scale_y) * myu_size_z + (y / scale_z);
                int idx = t * (params.Nx * params.Ny) + x * params.Ny + y;
                myu[idx] = myu_small[idx_small];
            }
        }
    }

    // Debug print: Verify contents of myu
    std::cout << "myu (first 10 values): ";
    for (int i = 0; i < std::min(10, myu_size_total); ++i) {
        std::cout << myu[i] << " ";
    }
    std::cout << "\n";

    // Allocate GPU memory and copy the data
    cudaError_t err = cudaMalloc(&params.d_myu, myu_size_total * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_myu: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMemcpy(params.d_myu, myu.data(), myu_size_total * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data to d_myu: " << cudaGetErrorString(err) << std::endl;
    }
}

void print_parameters(const Params& params) {
    std::cout << "dt: " << params.dt << "\n";
    std::cout << "dx: " << params.dx << "\n";
    std::cout << "dy: " << params.dy << "\n";
    std::cout << "Nt: " << params.Nt << "\n";
    std::cout << "Nx: " << params.Nx << "\n";
    std::cout << "Ny: " << params.Ny << "\n";
    std::cout << "myu_size: [" << std::get<0>(params.myu_size) << ", " << std::get<1>(params.myu_size) << ", " << std::get<2>(params.myu_size) << "]\n";
    std::cout << "myu_mstd: [" << std::get<0>(params.myu_mstd) << ", " << std::get<1>(params.myu_mstd) << "]\n";

    // Print first 10 values of GPU arrays
    std::vector<double> h_kx(params.Nx);
    cudaMemcpy(h_kx.data(), params.d_kx, params.Nx * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "kx: ";
    for (int i = 0; i < std::min(10, params.Nx); ++i) {
        std::cout << h_kx[i] << " ";
    }
    std::cout << "\n";

    std::vector<double> h_ky(params.Ny);
    cudaMemcpy(h_ky.data(), params.d_ky, params.Ny * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "ky: ";
    for (int i = 0; i < std::min(10, params.Ny); ++i) {
        std::cout << h_ky[i] << " ";
    }
    std::cout << "\n";

    std::vector<double> h_q(params.Nx * params.Ny);
    cudaMemcpy(h_q.data(), params.d_q, params.Nx * params.Ny * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "q: ";
    for (int i = 0; i < std::min(10, params.Nx * params.Ny); ++i) {
        std::cout << h_q[i] << " ";
    }
    std::cout << "\n";

    std::vector<double> h_exponent(params.Nx * params.Ny);
    cudaMemcpy(h_exponent.data(), params.d_exponent, params.Nx * params.Ny * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "exponent: ";
    for (int i = 0; i < std::min(10, params.Nx * params.Ny); ++i) {
        std::cout << h_exponent[i] << " ";
    }
    std::cout << "\n";

    std::vector<double> h_expm1(params.Nx * params.Ny);
    cudaMemcpy(h_expm1.data(), params.d_expm1, params.Nx * params.Ny * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "expm1: ";
    for (int i = 0; i < std::min(10, params.Nx * params.Ny); ++i) {
        std::cout << h_expm1[i] << " ";
    }
    std::cout << "\n";

    std::vector<double> h_step_1(params.Nx * params.Ny);
    cudaMemcpy(h_step_1.data(), params.d_step_1, params.Nx * params.Ny * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "step_1: ";
    for (int i = 0; i < std::min(10, params.Nx * params.Ny); ++i) {
        std::cout << h_step_1[i] << " ";
    }
    std::cout << "\n";

    std::vector<double> h_step_2(params.Nx * params.Ny);
    cudaMemcpy(h_step_2.data(), params.d_step_2, params.Nx * params.Ny * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "step_2: ";
    for (int i = 0; i < std::min(10, params.Nx * params.Ny); ++i) {
        std::cout << h_step_2[i] << " ";
    }
    std::cout << "\n";

    std::cout << "GPU memory allocated and initialized.\n";
}

void free_parameters(Params& params) {
    cudaFree(params.d_kx);
    cudaFree(params.d_ky);
    cudaFree(params.d_Kx);
    cudaFree(params.d_Ky);
    cudaFree(params.d_q);
    cudaFree(params.d_exponent);
    cudaFree(params.d_expm1);
    cudaFree(params.d_step_1);
    cudaFree(params.d_step_2);
    cudaFree(params.d_N_hat_prev);
}