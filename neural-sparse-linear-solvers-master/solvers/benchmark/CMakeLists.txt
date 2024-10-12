cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(solvers_benchmark LANGUAGES C CXX CUDA)

find_package(benchmark REQUIRED)
find_package(Torch REQUIRED)

# Create benchmark target
add_executable(solvers_benchmark main.cpp)

target_include_directories(solvers_benchmark PRIVATE)

target_link_libraries(
  solvers_benchmark PRIVATE solvers benchmark::benchmark "${TORCH_LIBRARIES}")

# Define compile features for this target: c++17 for tests
target_compile_features(solvers_benchmark PRIVATE cxx_std_17)

# Disable CXX extensions (it's better to do not rely upon compiler extensions)
set_target_properties(solvers_benchmark PROPERTIES CXX_EXTENSIONS OFF)
