#include <chrono>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

#include <torch/script.h>
#include <torch/torch.h>

#include <solvers/NeuralSolver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

NeuralSolver::NeuralSolver(std::filesystem::path model_path, const bool gpu)
    : _model_path(std::move(model_path)), _gpu(gpu)
{
    c10::Device device((_gpu) ? torch::kCUDA : torch::kCPU);
    try {
        module = torch::jit::load(_model_path, device);
    }
    catch (const c10::Error& e) {
        throw std::runtime_error("Error loading graph neural solver model");
    }
    module.eval();
}

Eigen::VectorXd NeuralSolver::solve(const SparseSystem& system,
                                    double& duration) const
{
    auto [A_indices, A_values, b] = system.toTorchCOO(_gpu);

    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    torch::Tensor output = module.forward({b, A_indices, A_values}).toTensor();
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd result = Eigen::Map<Eigen::VectorXd>(
        output.to(torch::kCPU).data_ptr<double>(), output.size(0));
    std::chrono::duration<double> time_difference = end - start;
    duration = time_difference.count();
    return result;
}

}    // namespace solvers
