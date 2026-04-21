cmake_minimum_required(VERSION 3.16)
project(VecScaleCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

add_executable(prepare_dataset scripts/prepare_dataset.cpp)
target_link_libraries(prepare_dataset PRIVATE vecscale)

add_executable(run_demo scripts/run_demo.cpp)
target_link_libraries(run_demo PRIVATE vecscale)
