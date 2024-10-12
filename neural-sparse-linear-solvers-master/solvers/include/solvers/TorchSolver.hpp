#ifndef SOLVERS_TORCHSOLVER_HPP
#define SOLVERS_TORCHSOLVER_HPP

#include <Eigen/Dense>

#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

enum TorchMethod { CG };

/**
 * \brief TorchSolver Iterative solver based on Torch library.
 */
class TorchSolver : public Solver {
private:
    /**
     * \brief _method Resolution algorithm.
     */
    TorchMethod _method;

    /**
     * \brief _tolerance Minimum absolute or relative error needed to stop the iterations.
     */
    double _tolerance;

    /**
     * \brief _iterations Maximum number of iterations.
     */
    int _iterations;

    /**
     * \brief _gpu Whether to use GPU.
     */
    bool _gpu;

    /**
     * \brief _cg_solve Solve a sparse linear system using Conjugate Gradient method.
     * 
     * \param[in] system Sparse linear system.
     * \param[out] duration Running time of the resolution in seconds.
     * \return Solution vector of the sparse linear system in Eigen format.
     */
    Eigen::VectorXd _cg_solve(const SparseSystem& system,
                              double& duration) const;

public:
    explicit TorchSolver(const TorchMethod method = TorchMethod::CG,
                         const double tolerance = 1e-8,
                         const int iterations = 300,
                         const bool gpu = false)
        : _method(method),
          _tolerance(tolerance),
          _iterations(iterations),
          _gpu(gpu){};

    Eigen::VectorXd solve(const SparseSystem& system,
                          double& duration) const override;
};

}    // namespace solvers

#endif    // SOLVERS_TORCHSOLVER_HPP
