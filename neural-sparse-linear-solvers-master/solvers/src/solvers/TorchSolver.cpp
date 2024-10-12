#include <chrono>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <torch/torch.h>

#include <solvers/SparseSystem.hpp>
#include <solvers/TorchSolver.hpp>

namespace solvers {

Eigen::VectorXd TorchSolver::_cg_solve(const SparseSystem& system,
                                       double& duration) const
{
    auto [A_indices, A_values, b] = system.toTorchCOO(_gpu);
    const int64_t n = b.size(0);
    torch::TensorOptions options = torch::TensorOptions()
                                       .dtype(A_values.dtype())
                                       .device(A_values.device());
    torch::Tensor A =
        torch::sparse_coo_tensor(A_indices, A_values, {n, n}, options);
    torch::Tensor x = torch::zeros_like(b);
    torch::Tensor r = b.clone();
    torch::Tensor p = r.clone();
    torch::Tensor r_squared_norm = r.square().sum();
    torch::Tensor Ap;
    torch::Tensor r1_squared_norm;
    torch::Tensor alpha;
    torch::Tensor beta;
    torch::Tensor stop_threshold = torch::clamp_min(b.norm(), 1.) * _tolerance;
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    for (size_t k = 0; k < _iterations; ++k) {
        Ap = torch::mv(A, p);
        alpha = r_squared_norm / (p * Ap).sum();
        x += alpha * p;
        r -= alpha * Ap;
        r1_squared_norm = r.square().sum();
        if ((r1_squared_norm.sqrt() < stop_threshold).item<bool>()) {
            break;
        }
        beta = r1_squared_norm / r_squared_norm;
        p = r + beta * p;
        r_squared_norm = r1_squared_norm;
    }
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd result = Eigen::Map<Eigen::VectorXd>(
        x.to(torch::kCPU).data_ptr<double>(), x.size(0));
    std::chrono::duration<double> time_difference = end - start;
    duration = time_difference.count();
    return result;
}

Eigen::VectorXd TorchSolver::solve(const SparseSystem& system,
                                   double& duration) const
{
    Eigen::VectorXd result;
    switch (_method) {
        case TorchMethod::CG:
            result = _cg_solve(system, duration);
            break;
        default:
            throw std::logic_error("Invalid solving method");
            break;
    }
    return result;
}

}    // namespace solvers
