# CMakeLists.txt — Standalone IPM solver Python module
# Builds ipm_solver.cpython-*.so for use outside simplinho

cmake_minimum_required(VERSION 3.16)
project(ipm_solver VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find dependencies
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

# Eigen (same as main project)
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4.0 QUIET)
    if(NOT Eigen3_FOUND)
        message(STATUS "Eigen3 not found, using FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            Eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.1
        )
        FetchContent_MakeAvailable(Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
endif()

# pybind11
find_package(pybind11 2.6.0 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, using FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.4
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Build IPM module
pybind11_add_module(ipm_solver
    ipm_module.cpp
    ../src/ipm/IPSolver.cpp
    ../src/ipm/ipm_linear_solver.cpp
)

target_include_directories(ipm_solver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(ipm_solver PRIVATE Eigen3::Eigen)

target_compile_features(ipm_solver PRIVATE cxx_std_23)

# Install
install(TARGETS ipm_solver LIBRARY DESTINATION .)
