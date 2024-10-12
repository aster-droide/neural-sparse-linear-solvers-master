cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(SOURCES
    solvers/CuSparseSolver.cpp
    solvers/NeuralSolver.cpp
    solvers/SparseSystem.cpp
    solvers/TorchSolver.cpp
    solvers/ViennaCLSolver.cu
)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Torch REQUIRED)

add_library(solvers "${SOURCES}")

target_include_directories(solvers PUBLIC "${PROJECT_SOURCE_DIR}/include" "${CUDA_INCLUDE_DIRS}")
# libs folder considered as system header
target_include_directories(solvers SYSTEM PUBLIC "${CMAKE_LIBRARY_PATH}")

target_link_libraries(
  solvers
  PUBLIC Eigen3::Eigen
         OpenMP::OpenMP_CXX
         cnpy
         "${CUDA_cusparse_LIBRARY}"
         "${CUDA_cusolver_LIBRARY}"
         "${TORCH_LIBRARIES}")

target_compile_features(solvers PRIVATE cxx_std_17)
set_target_properties(solvers PROPERTIES CXX_EXTENSIONS OFF)
