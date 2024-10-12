#ifndef SOLVERS_VIENNACLSOLVER_HPP
#define SOLVERS_VIENNACLSOLVER_HPP

#include <Eigen/Dense>

#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/ichol.hpp>
#include <viennacl/linalg/ilu.hpp>
#include <viennacl/linalg/jacobi_precond.hpp>

#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

enum class ViennaCLMethod { CG };
enum class ViennaCLPreconditioner { None, ILU0, IChol0 };

/**
 * \brief ViennaCLSolver Iterative solver based on ViennaCL library.
 */
class ViennaCLSolver : public Solver {
public:
    /**
     * \brief _method Resolution algorithm.
     */
    ViennaCLMethod _method;

    /**
     * \brief _preconditioner Preconditioning algorithm.
     */
    ViennaCLPreconditioner _preconditioner;

    /**
     * \brief _cg_config Parameters for conjugate gradient algorithm.
     */
    viennacl::linalg::cg_tag _cg_config;

    /**
     * \brief _ilu0_config Parameters for ILU(0) preconditioner.
     */
    viennacl::linalg::ilu0_tag _ilu0_config;

    /**
     * \brief _ichol0_config Parameters for ICC preconditioner.
     */
    viennacl::linalg::ichol0_tag _ichol0_config;


    explicit ViennaCLSolver(
        ViennaCLMethod method = ViennaCLMethod::CG,
        ViennaCLPreconditioner preconditioner = ViennaCLPreconditioner::None,
        double tolerance = 1e-8,
        int iterations = 1000);

    Eigen::VectorXd solve(const SparseSystem& system,
                          double& duration) const override;
};

}    // namespace solvers

#endif    // SOLVERS_VIENNACLSOLVER_HPP
