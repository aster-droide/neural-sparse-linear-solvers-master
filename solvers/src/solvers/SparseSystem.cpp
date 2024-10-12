#include <filesystem>
#include <tuple>
#include <utility>
#include <vector>

#include <cnpy/cnpy.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <torch/torch.h>

#include <solvers/SparseSystem.hpp>

namespace solvers {

SparseSystem::SparseSystem(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& upper_A,
    Eigen::VectorXd b,
    Eigen::VectorXd x)
    : _upper_A(upper_A), _b(std::move(b)), _x(std::move(x))
{
    _upper_A.makeCompressed();
}

SparseSystem SparseSystem::loadNpz(const std::filesystem::path& npz_path)
{
    cnpy::npz_t system = cnpy::npz_load(npz_path.string());
    cnpy::NpyArray A_indices_array = system["A_indices"];
    cnpy::NpyArray A_values_array = system["A_values"];
    cnpy::NpyArray b_array = system["b"];
    cnpy::NpyArray x_array = system["x"];

    const auto n = static_cast<int64_t>(b_array.shape[0]);
    const auto nnz = static_cast<int64_t>(A_values_array.shape[0]);

    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(b_array.data<double>(), n);
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(x_array.data<double>(), n);
    Eigen::MatrixX2<int> indices =
        Eigen::Map<Eigen::MatrixX2<int>>(A_indices_array.data<int>(), nnz, 2);
    Eigen::VectorXd values =
        Eigen::Map<Eigen::VectorXd>(A_values_array.data<double>(), nnz);

    std::vector<Eigen::Triplet<double>> upper_A_entries;
    upper_A_entries.reserve(nnz / 2 + n);
    for (int k = 0; k < nnz; ++k) {
        if (indices(k, 0) <= indices(k, 1)) {
            upper_A_entries.emplace_back(Eigen::Triplet<double>(
                indices(k, 0), indices(k, 1), values(k)));
        }
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor> upper_A(n, n);
    upper_A.setFromTriplets(upper_A_entries.begin(), upper_A_entries.end());
    upper_A.makeCompressed();
    return SparseSystem{upper_A, b, x};
}

void SparseSystem::saveNpz(const std::filesystem::path& npz_path) const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(
        _upper_A.selfadjointView<Eigen::Upper>());
    const size_t n = A.rows();
    const size_t nnz = A.nonZeros();

    using InnerIterator =
        Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator;
    std::vector<int> indices_vector;
    indices_vector.reserve(2 * nnz);
    for (Eigen::Index row = 0; row < static_cast<Eigen::Index>(n); ++row) {
        for (InnerIterator it(A, row); it; ++it) {
            indices_vector.emplace_back(it.row());
        }
    }
    for (size_t c = 0; c < nnz; ++c) {
        indices_vector.emplace_back(*(A.innerIndexPtr() + c));
    }

    cnpy::npz_save<int>(npz_path.string(), "A_indices", indices_vector.data(),
                        {2, nnz}, "a");
    cnpy::npz_save<double>(npz_path.string(), "A_values", A.valuePtr(), {nnz},
                           "a");
    cnpy::npz_save<double>(npz_path.string(), "b", _b.data(), {n}, "a");
    cnpy::npz_save<double>(npz_path.string(), "x", _x.data(), {n}, "a");
}

size_t SparseSystem::dim() const
{
    return static_cast<size_t>(_b.size());
}

size_t SparseSystem::nnz() const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(
        _upper_A.selfadjointView<Eigen::Upper>());
    return static_cast<size_t>(A.nonZeros());
}

Eigen::VectorXd SparseSystem::getEigenSolution() const
{
    return _x;
}

torch::Tensor SparseSystem::getTorchSolution(const bool gpu) const
{
    torch::TensorOptions options;
    const int64_t n = _x.size();
    std::vector<double> x_vector(_x.data(), _x.data() + n);
    torch::Tensor x = torch::from_blob(x_vector.data(), c10::IntArrayRef{n},
                                       options.dtype(torch::kFloat64));
    if (gpu) {
        x = x.to(torch::kCUDA);
    }
    return x;
}

std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::VectorXd>
SparseSystem::toEigenCSR() const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(
        _upper_A.selfadjointView<Eigen::Upper>());
    return std::make_tuple(A, _b);
}

std::tuple<std::vector<int>,
           std::vector<int>,
           std::vector<double>,
           std::vector<double>>
SparseSystem::toStdCSR() const
{
    const Eigen::SparseMatrix<double, Eigen::RowMajor> A(
        _upper_A.selfadjointView<Eigen::Upper>());
    const size_t n = A.rows();
    const size_t nnz = A.nonZeros();
    std::vector<int> crow_index(A.outerIndexPtr(), A.outerIndexPtr() + n + 1);
    std::vector<int> col_index(A.innerIndexPtr(), A.innerIndexPtr() + nnz);
    std::vector<double> values(A.valuePtr(), A.valuePtr() + nnz);
    std::vector<double> b(_b.data(), _b.data() + n);
    return std::make_tuple(crow_index, col_index, values, b);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SparseSystem::toTorchCOO(const bool gpu) const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(
        _upper_A.selfadjointView<Eigen::Upper>());
    const int64_t n = A.rows();
    const int64_t nnz = A.nonZeros();

    using InnerIterator =
        Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator;
    std::vector<int> row_index_vector;
    row_index_vector.reserve(nnz);
    for (int64_t row = 0; row < n; ++row) {
        for (InnerIterator it(A, row); it; ++it) {
            row_index_vector.emplace_back(it.row());
        }
    }

    torch::TensorOptions options;

    torch::Tensor row_index = torch::from_blob(row_index_vector.data(), {nnz},
                                               options.dtype(torch::kInt32));
    row_index = row_index.to(torch::kInt64);

    torch::Tensor col_index = torch::from_blob(A.innerIndexPtr(), {nnz},
                                               options.dtype(torch::kInt32));
    col_index = col_index.to(torch::kInt64);
    torch::Tensor index = torch::stack({row_index, col_index});

    torch::Tensor values =
        torch::from_blob(A.valuePtr(), {nnz}, options.dtype(torch::kFloat64));

    std::vector<double> b_vector(_b.data(), _b.data() + n);
    torch::Tensor b =
        torch::from_blob(b_vector.data(), {n}, options.dtype(torch::kFloat64));

    if (gpu) {
        index = index.to(torch::kCUDA);
        values = values.to(torch::kCUDA);
        b = b.to(torch::kCUDA);
    }
    return std::make_tuple(index, values, b);
}

}    // namespace solvers
