#ifndef SOLVERS_NEURALSOLVER_HPP
#define SOLVERS_NEURALSOLVER_HPP

#include <filesystem>

#include <Eigen/Dense>

#include <torch/torch.h>

#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

class NeuralSolver : public Solver {
private:
    /**
     * \brief _model_path Path to Torch traced module.
     */
    std::filesystem::path _model_path;

    /**
     * \brief _gpu Whether to use GPU.
     */
    bool _gpu;

public:
    /**
     * \brief module Torch traced module.
     */
    mutable torch::jit::script::Module module;

    explicit NeuralSolver(std::filesystem::path model_path, bool gpu = false);

    Eigen::VectorXd solve(const SparseSystem& system,
                          double& duration) const override;
};

}    // namespace solvers

#endif    // SOLVERS_NEURALSOLVER_HPP
