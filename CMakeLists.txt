cmake_minimum_required(VERSION 3.20)

project(solvers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

if(NOT COMMAND CPMAddPackage)
    FetchContent_Declare(
        CPM
        GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
        GIT_TAG v0.42.1
    )
    FetchContent_MakeAvailable(CPM)
    include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)
endif()

CPMAddPackage(
    NAME Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    OPTIONS
        "EIGEN_BUILD_DOC OFF"
        "BUILD_TESTING OFF"
)

CPMAddPackage(
    NAME nanobind
    GITHUB_REPOSITORY wjakob/nanobind
    GIT_TAG v2.9.2
)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

add_library(solvers_headers INTERFACE)
target_include_directories(
    solvers_headers
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_link_libraries(solvers_headers INTERFACE Eigen3::Eigen)
target_compile_features(solvers_headers INTERFACE cxx_std_20)

nanobind_add_module(osqp_ext python/osqp_ext.cpp)
target_link_libraries(osqp_ext PRIVATE solvers_headers)
set_target_properties(
    osqp_ext
    PROPERTIES
        OUTPUT_NAME osqp
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/solvers
)

nanobind_add_module(piqp_ext python/piqp_ext.cpp)
target_link_libraries(piqp_ext PRIVATE solvers_headers)
set_target_properties(
    piqp_ext
    PROPERTIES
        OUTPUT_NAME piqp
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/solvers
)

nanobind_add_module(qdldl_ext python/qdldl_ext.cpp)
target_link_libraries(qdldl_ext PRIVATE solvers_headers)
set_target_properties(
    qdldl_ext
    PROPERTIES
        OUTPUT_NAME qdldl
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/solvers
)

configure_file(python/solvers/__init__.py ${CMAKE_BINARY_DIR}/solvers/__init__.py COPYONLY)

install(TARGETS osqp_ext piqp_ext qdldl_ext LIBRARY DESTINATION solvers)
install(FILES python/solvers/__init__.py DESTINATION solvers)
