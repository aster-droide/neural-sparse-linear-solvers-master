#include <chrono>

#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse_v2.h>

#include <solvers/CuSparseSolver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

Eigen::VectorXd CuSparseSolver::solve(const SparseSystem& system,
                                      double& duration) const
{
    auto [crow_index, col_index, values, b] = system.toStdCSR();
    const auto n = static_cast<int>(b.size());
    const auto nnz = static_cast<int>(values.size());

    cusparseMatDescr_t A_description = nullptr;
    cusparseCreateMatDescr(&A_description);

    int* cuda_crow_index;
    int* cuda_col_index;
    double* cuda_values;
    double* cuda_b;
    double* cuda_sol;

    cusolverSpHandle_t cusolver_handler = nullptr;
    cusolverSpCreate(&cusolver_handler);
    cudaMalloc(reinterpret_cast<void**>(&cuda_values), sizeof(double) * nnz);
    cudaMalloc(reinterpret_cast<void**>(&cuda_crow_index),
               sizeof(int) * (n + 1));
    cudaMalloc(reinterpret_cast<void**>(&cuda_col_index), sizeof(int) * nnz);
    cudaMalloc(reinterpret_cast<void**>(&cuda_b), sizeof(double) * n);
    cudaMalloc(reinterpret_cast<void**>(&cuda_sol), sizeof(double) * n);

    cudaMemcpy(cuda_values, values.data(), sizeof(double) * nnz,
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_crow_index, crow_index.data(), sizeof(int) * (n + 1),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_col_index, col_index.data(), sizeof(int) * nnz,
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    const double tolerance = 1e-12;
    int singularity = 0;

    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    cusolverSpDcsrlsvchol(cusolver_handler, n, nnz, A_description, cuda_values,
                          cuda_crow_index, cuda_col_index, cuda_b, tolerance,
                          static_cast<int>(_reorder), cuda_sol, &singularity);
    cudaDeviceSynchronize();
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd result(n);
    cudaMemcpy(result.data(), cuda_sol, sizeof(double) * n,
               cudaMemcpyDeviceToHost);

    cudaFree(cuda_crow_index);
    cudaFree(cuda_col_index);
    cudaFree(cuda_values);
    cudaFree(cuda_b);
    cudaFree(cuda_sol);
    cusolverSpDestroy(cusolver_handler);
    cudaDeviceReset();
    std::chrono::duration<double> time_difference = end - start;
    duration = time_difference.count();
    return result;
}

}    // namespace solvers
