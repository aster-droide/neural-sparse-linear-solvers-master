#ifndef SOLVERS_SPARSESYSTEM_HPP
#define SOLVERS_SPARSESYSTEM_HPP

#include <filesystem>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <torch/torch.h>

namespace solvers {

/**
 * \brief SparseSystem Represent a solved sparse symmetric linear system.
 */
class SparseSystem {
private:
    /**
     * \brief _upper_A Triangular upper part of the sparse symmetric matrix of the system.
     */
    Eigen::SparseMatrix<double, Eigen::RowMajor> _upper_A;

    /**
     * \brief _b Constant term of the system.
     */
    Eigen::VectorXd _b;

    /**
     * \brief _x Solution of the system.
     */
    Eigen::VectorXd _x;

public:
    SparseSystem(const Eigen::SparseMatrix<double, Eigen::RowMajor>& upper_A,
                 Eigen::VectorXd b,
                 Eigen::VectorXd x);

    /**
     * \brief loadNpz Load a sparse system from a .npz file in StAnD format.
     * 
     * \param[in] npz_path Path to the input .npz file.
     * \return Solved sparse symmetric linear system.
     */
    [[nodiscard]] static SparseSystem loadNpz(
        const std::filesystem::path& npz_path);

    /**
     * \brief saveNps Save a sparse system in a .npz file in StAnD format.
     * 
     * \param[in] npz_path Path to the output .npz file.
     */
    void saveNpz(const std::filesystem::path& npz_path) const;

    /**
     * \brief dim Get the number of rows and columns of the sparse matrix.
     * 
     * \return Number of rows and columns of the sparse matrix.
     */
    [[nodiscard]] size_t dim() const;

    /**
     * \brief nnz Get the number of non-null coefficients of the sparse matrix.
     * 
     * \return Number of non-null coefficients of the sparse matrix.
     */
    [[nodiscard]] size_t nnz() const;

    /**
     * \brief getEigenSolution Get the solution vector in Eigen format.
     * 
     * \return Solution vector.
     */
    [[nodiscard]] Eigen::VectorXd getEigenSolution() const;

    /**
     * \brief getTorchSolution Get the solution vector in Torch format.
     * 
     * \param[in] gpu Whether to put output tensors on GPU.
     * \return Solution vector.
     */
    [[nodiscard]] torch::Tensor getTorchSolution(bool gpu = false) const;

    /**
     * \brief toEigenCSR Get the coefficient matrix and the constant term in Eigen format.
     *        The sparse coefficient matrix is stored in CSR format.
     * 
     * \return Pair of coefficient matrix and constant term in Eigen format.
     */
    [[nodiscard]] std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>,
                             Eigen::VectorXd>
    toEigenCSR() const;

    /**
     * \brief toStdCSR Get the coefficient matrix and the constant term in std::vector format.
     *        The sparse coefficient matrix is stored in CSR format and it is
     *        represented by three vectors: compressed row indices, column indices and values.
     * 
     * \return Tuple of vectors representing:
     *         - compressed row indices of the coefficient matrix;
     *         - column indices of the coefficient matrix;
     *         - values of the coefficients;
     *         - constant term.
     */
    [[nodiscard]] std::tuple<std::vector<int>,
                             std::vector<int>,
                             std::vector<double>,
                             std::vector<double>>
    toStdCSR() const;

    /**
     * \brief toStdCSR Get the coefficient matrix and the constant term in std::vector format.
     *        The sparse coefficient matrix is stored in CSR format and it is
     *        represented by three vectors: compressed row indices, column indices and values.
     * 
     * \param[in] gpu Whether to put output tensors on GPU.
     * \return Tuple of tensors representing:
     *         - compressed row indices of the coefficient matrix;
     *         - column indices of the coefficient matrix;
     *         - values of the coefficients;
     *         - constant term.
     */
    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    toTorchCOO(bool gpu = false) const;
};

}    // namespace solvers

#endif    // SOLVERS_SPARSESYSTEM_HPP
