cmake_minimum_required(VERSION 3.20)
project(VecScale CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# ---- Eigen ----
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found locally — fetching via FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
  )
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING   OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
endif()

# ---- nlohmann/json ----
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found locally — fetching via FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# ---- OpenMP ----
find_package(OpenMP)

# ---- Core library ----
add_library(vecscale_lib
  src/compute.cpp
  src/data.cpp
  src/aggregator.cpp
  src/router.cpp
  src/worker.cpp
  src/baselines.cpp
  src/benchmark.cpp
)

target_include_directories(vecscale_lib PUBLIC include)
target_link_libraries(vecscale_lib PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json)

if(OpenMP_CXX_FOUND)
  target_link_libraries(vecscale_lib PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(vecscale_lib PUBLIC VECSCALE_USE_OPENMP)
  message(STATUS "OpenMP enabled")
else()
  message(STATUS "OpenMP not found — single-threaded baseline only")
endif()

# ---- CLI tools ----
add_executable(prepare_dataset tools/prepare_dataset.cpp)
target_link_libraries(prepare_dataset PRIVATE vecscale_lib)

add_executable(run_demo tools/run_demo.cpp)
target_link_libraries(run_demo PRIVATE vecscale_lib)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
