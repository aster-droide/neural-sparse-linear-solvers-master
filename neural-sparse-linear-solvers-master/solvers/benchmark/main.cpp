#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include <Eigen/Dense>

#include <torch/torch.h>

#include <solvers/CuSparseSolver.hpp>
#include <solvers/NeuralSolver.hpp>
#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>
#include <solvers/TorchSolver.hpp>
#include <solvers/ViennaCLSolver.hpp>

namespace fs = std::filesystem;

std::vector<fs::path> getDatasetPaths(const fs::path& dataset_dir)
{
    std::vector<fs::path> paths;
    for (const fs::path& entry : fs::directory_iterator(dataset_dir)) {
        if (entry.extension() == ".npz") {
            paths.emplace_back(entry);
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

bool endsWith(std::string const& value, std::string const& ending)
{
    if (ending.size() > value.size()) {
        return false;
    }
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void benchmarkSolver(benchmark::State& state,
                     const solvers::Solver& solver,
                     const std::vector<fs::path>& dataset,
                     const int warmup)
{
    double error_sum = 0.;
    for (int i = 0; i < warmup; ++i) {
        solvers::SparseSystem system =
            solvers::SparseSystem::loadNpz(dataset[i]);
        double duration;
        Eigen::VectorXd result = solver.solve(system, duration);
        std::cout << "Warmup duration " << duration << std::endl;
    }
    for (auto _ : state) {
        int64_t it = state.items_processed();
        solvers::SparseSystem system =
            solvers::SparseSystem::loadNpz(dataset[it]);
        double duration;
        Eigen::VectorXd result = solver.solve(system, duration);
        state.SetIterationTime(duration);
        Eigen::VectorXd solution = system.getEigenSolution();
        double error = (result - solution).norm() / solution.norm();
        error_sum += error;
        state.SetItemsProcessed(it + 1);
    }
    state.counters["Average Relative Error"] =
        benchmark::Counter(error_sum, benchmark::Counter::kAvgIterations);
}

std::vector<benchmark::internal::Benchmark*> registerCuSparseBenchmarks(
    const std::vector<fs::path>& dataset)
{
    std::map<std::string,
             std::pair<solvers::CuSparseMethod, solvers::CuSparseReorder>>
        configs = {{"CuSparse/Cholesky",
                    {solvers::CuSparseMethod::Cholesky,
                     solvers::CuSparseReorder::None}},
                   {"CuSparse/CholeskySymRCM",
                    {solvers::CuSparseMethod::Cholesky,
                     solvers::CuSparseReorder::SymRCM}},
                   {"CuSparse/CholeskySymAMD",
                    {solvers::CuSparseMethod::Cholesky,
                     solvers::CuSparseReorder::SymAMD}},
                   {"CuSparse/CholeskyMetis",
                    {solvers::CuSparseMethod::Cholesky,
                     solvers::CuSparseReorder::METIS}}};
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    benchmarks.reserve(configs.size());
    constexpr int warmup = 0;
    for (const auto& [name, config] : configs) {
        solvers::CuSparseSolver solver(config.first, config.second);
        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            name.c_str(), benchmarkSolver, solver, dataset, warmup));
    }
    return benchmarks;
}

std::vector<benchmark::internal::Benchmark*> registerViennaCLBenchmarks(
    const std::vector<fs::path>& dataset,
    const double tolerance = 1e-6,
    const int iterations = 1000)
{
    std::map<std::string, std::pair<solvers::ViennaCLMethod,
                                    solvers::ViennaCLPreconditioner>>
        configs = {
            {"ViennaCL/Cg",
             {solvers::ViennaCLMethod::CG,
              solvers::ViennaCLPreconditioner::None}},
            {"ViennaCL/CgIlu0",
             {solvers::ViennaCLMethod::CG,
              solvers::ViennaCLPreconditioner::ILU0}},
            {"ViennaCL/CgIchol0",
             {solvers::ViennaCLMethod::CG,
              solvers::ViennaCLPreconditioner::IChol0}},
        };
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    benchmarks.reserve(configs.size());
    constexpr int warmup = 0;
    for (const auto& [name, config] : configs) {
        solvers::ViennaCLSolver solver(config.first, config.second, tolerance,
                                       iterations);
        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            name.c_str(), benchmarkSolver, solver, dataset, warmup));
    }
    return benchmarks;
}

std::vector<benchmark::internal::Benchmark*> registerTorchBenchmarks(
    const std::vector<fs::path>& dataset,
    const double tolerance = 1e-6,
    const int iterations = 1000)
{
    std::map<std::string, solvers::TorchMethod> configs = {
        {"Torch/Cg", solvers::TorchMethod::CG},
    };
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    benchmarks.reserve(configs.size());
    constexpr bool gpu = true;
    constexpr int warmup = 0;
    for (const auto& [name, config] : configs) {
        solvers::TorchSolver solver(config, tolerance, iterations, gpu);
        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            name.c_str(), benchmarkSolver, solver, dataset, warmup));
    }
    return benchmarks;
}

std::vector<benchmark::internal::Benchmark*> registerNeuralBenchmarks(
    const std::vector<fs::path>& dataset,
    const fs::path& model,
    const int warmup = 25)
{
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    benchmarks.reserve(1);
    constexpr bool gpu = true;
    solvers::NeuralSolver solver(model, gpu);
    benchmarks.emplace_back(benchmark::RegisterBenchmark(
        "Torch/Nsls", benchmarkSolver, solver, dataset, warmup));
    return benchmarks;
}

int main(int argc, char** argv)
{
    fs::path dataset_dir;
    fs::path model_path;

    const char* dataset_option = "--dataset=";
    const char* model_option = "--model=";
    std::vector<char*> benchmark_arguments;
    benchmark_arguments.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        if (strncmp(dataset_option, argv[i], strlen(dataset_option)) == 0) {
            dataset_dir.assign(argv[i] + strlen(dataset_option));
        }
        else if (strncmp(model_option, argv[i], strlen(model_option)) == 0) {
            model_path.assign(argv[i] + strlen(model_option));
        }
        else {
            benchmark_arguments.emplace_back(argv[i]);
        }
    }
    if (dataset_dir.empty()) {
        std::cout << "The required argument --dataset is missing, use "
                     "--dataset=<dataset_directory>"
                  << std::endl;
        return 1;
    }
    if (!fs::exists(dataset_dir)) {
        std::cout << "No such dataset directory: " << dataset_dir << std::endl;
        return 1;
    }
    if (!fs::exists(model_path)) {
        std::cout << "No such model path: " << model_path << std::endl;
        return 1;
    }

    std::vector<fs::path> dataset = getDatasetPaths(dataset_dir);

    std::vector<std::vector<benchmark::internal::Benchmark*>> benchmarks = {
        registerNeuralBenchmarks(dataset, model_path),
        registerTorchBenchmarks(dataset),
        registerViennaCLBenchmarks(dataset),
        registerCuSparseBenchmarks(dataset),
    };
    for (const auto& lib_benchmarks : benchmarks) {
        for (const auto& benchmark : lib_benchmarks) {
            benchmark->UseManualTime()
                ->Unit(benchmark::kMillisecond)
                ->Iterations(dataset.size());
        }
    }

    argc = static_cast<int>(benchmark_arguments.size());
    argv = benchmark_arguments.data();
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
