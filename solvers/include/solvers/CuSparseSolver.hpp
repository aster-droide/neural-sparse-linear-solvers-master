#ifndef SOLVERS_CUSPARSESOLVER_HPP
#define SOLVERS_CUSPARSESOLVER_HPP

#include <Eigen/Dense>

#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

enum class CuSparseMethod { Cholesky };
enum class CuSparseReorder { None, SymRCM, SymAMD, METIS };

/**
 * \brief CuSparseSolver Direct solver based on cuSPARSE library.
 */
class CuSparseSolver : public Solver {
private:
    /**
     * \brief _method Resolution algorithm.
     */
    CuSparseMethod _method;

    /**
     * \brief _reorder Algorithm of node reordering used to reduce fill-in.
     */
    CuSparseReorder _reorder;

public:
    explicit CuSparseSolver(
        const CuSparseMethod method = CuSparseMethod::Cholesky,
        const CuSparseReorder reorder = CuSparseReorder::METIS)
        : _method(method), _reorder(reorder){};

    Eigen::VectorXd solve(const SparseSystem& system,
                          double& duration) const override;
};

}    // namespace solvers

#endif    // SOLVERS_CUSPARSESOLVER_HPP
