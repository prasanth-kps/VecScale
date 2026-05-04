cmake_minimum_required(VERSION 3.16)
project(VecScaleCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VECSCALE_ENABLE_OPENMP "Enable OpenMP acceleration" ON)
option(VECSCALE_ENABLE_MPI "Enable MPI runner target" ON)
option(VECSCALE_ENABLE_CUDA "Enable CUDA backend in compute path" OFF)

find_package(Threads REQUIRED)

add_library(vecscale STATIC
    src/vecscale/aggregator.cpp
    src/vecscale/baselines.cpp
    src/vecscale/benchmark.cpp
    src/vecscale/compute.cpp
    src/vecscale/data.cpp
    src/vecscale/router.cpp
    src/vecscale/worker.cpp
)

target_include_directories(vecscale PUBLIC src)
target_link_libraries(vecscale PUBLIC Threads::Threads)

if(VECSCALE_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(vecscale PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(vecscale PUBLIC VECSCALE_HAS_OPENMP=1)
    endif()
endif()

if(VECSCALE_ENABLE_CUDA)
    enable_language(CUDA)
    set_target_properties(vecscale PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_sources(vecscale PRIVATE src/vecscale/compute_cuda.cu)
    target_compile_definitions(vecscale PUBLIC VECSCALE_HAS_CUDA=1)
else()
    target_sources(vecscale PRIVATE src/vecscale/compute_cuda_stub.cpp)
endif()

add_executable(prepare_dataset scripts/prepare_dataset.cpp)
target_link_libraries(prepare_dataset PRIVATE vecscale)

add_executable(run_demo scripts/run_demo.cpp)
target_link_libraries(run_demo PRIVATE vecscale)

if(VECSCALE_ENABLE_MPI)
    find_package(MPI QUIET)
    if(MPI_CXX_FOUND)
        add_executable(run_demo_mpi scripts/run_demo_mpi.cpp)
        target_link_libraries(run_demo_mpi PRIVATE vecscale MPI::MPI_CXX)
    endif()
endif()
