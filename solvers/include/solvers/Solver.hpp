#ifndef SOLVERS_SOLVER_HPP
#define SOLVERS_SOLVER_HPP

#include <Eigen/Dense>

#include <solvers/SparseSystem.hpp>

namespace solvers {

/**
 * \brief Solver Base class for linear system solvers.
 */
class Solver {
public:
    /**
     * \brief solve Solve a sparse linear system.
     * 
     * \param[in] system Sparse linear system.
     * \param[out] duration Running time of the resolution in seconds.
     * \return Solution vector of the sparse linear system in Eigen format.
     */
    virtual Eigen::VectorXd solve(const SparseSystem& system,
                                  double& duration) const = 0;
};

}    // namespace solvers

#endif    // SOLVERS_SOLVER_HPP
