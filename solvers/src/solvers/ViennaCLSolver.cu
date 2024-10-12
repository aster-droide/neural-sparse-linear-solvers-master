#ifndef VIENNACL_WITH_CUDA
#define VIENNACL_WITH_CUDA
#endif

#ifndef VIENNACL_HAVE_EIGEN
#define VIENNACL_HAVE_EIGEN
#endif

#include <chrono>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <viennacl/compressed_matrix.hpp>
#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/ichol.hpp>
#include <viennacl/linalg/ilu.hpp>
#include <viennacl/linalg/jacobi_precond.hpp>
#include <viennacl/vector.hpp>

#include <solvers/SparseSystem.hpp>
#include <solvers/ViennaCLSolver.hpp>

namespace solvers {

ViennaCLSolver::ViennaCLSolver(const ViennaCLMethod method,
                               const ViennaCLPreconditioner preconditioner,
                               const double tolerance,
                               const int iterations)
    : _method(method), _preconditioner(preconditioner)
{
    _cg_config = viennacl::linalg::cg_tag(tolerance, iterations);
}

Eigen::VectorXd ViennaCLSolver::solve(const SparseSystem& system,
                                      double& duration) const
{
    auto [A, b] = system.toEigenCSR();
    Eigen::SparseMatrix<float, Eigen::RowMajor> A_float = A.cast<float>();
    Eigen::VectorXf b_float = b.cast<float>();
    const size_t dim = b.size();
    viennacl::compressed_matrix<float> vcl_A(dim, dim);
    viennacl::vector<float> vcl_b(dim);
    viennacl::copy(A_float, vcl_A);
    viennacl::copy(b_float, vcl_b);
    viennacl::vector<float> viennacl_result;

    switch (_method) {
        case ViennaCLMethod::CG:
            break;
        default:
            throw std::logic_error("Invalid solving method");
            break;
    }

    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    switch (_preconditioner) {
        case ViennaCLPreconditioner::None:
            if (_method == ViennaCLMethod::CG) {
                viennacl_result =
                    viennacl::linalg::solve(vcl_A, vcl_b, _cg_config);
            }
            break;
        case ViennaCLPreconditioner::ILU0: {
            viennacl::linalg::ilu0_precond<viennacl::compressed_matrix<float>>
                ilu0(vcl_A, _ilu0_config);
            if (_method == ViennaCLMethod::CG) {
                viennacl_result =
                    viennacl::linalg::solve(vcl_A, vcl_b, _cg_config, ilu0);
            }
            break;
        }
        case ViennaCLPreconditioner::IChol0: {
            viennacl::linalg::ichol0_precond<viennacl::compressed_matrix<float>>
                ichol0(vcl_A, _ichol0_config);
            if (_method == ViennaCLMethod::CG) {
                viennacl_result =
                    viennacl::linalg::solve(vcl_A, vcl_b, _cg_config, ichol0);
            }
            break;
        }
        default:
            throw std::logic_error("Invalid preconditioner method");
            break;
    }
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_difference = end - start;
    duration = time_difference.count();
    Eigen::VectorXf result_float(static_cast<int>(dim));
    viennacl::copy(viennacl_result, result_float);
    Eigen::VectorXd result = result_float.cast<double>();
    return result;
}

}    // namespace solvers
