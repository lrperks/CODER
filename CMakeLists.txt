cmake_minimum_required(VERSION 3.20)
project(vedit LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Benchmark flag set (recorded in results/config.json).
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Vendored: tree-sitter 0.24.0 runtime, tree-sitter-python 0.23.6 grammar.
add_library(tree_sitter STATIC third_party/tree-sitter/src/lib.c)
target_include_directories(tree_sitter PUBLIC third_party/tree-sitter/include)
target_include_directories(tree_sitter PRIVATE third_party/tree-sitter/src)
add_library(tree_sitter_python STATIC
  third_party/tree-sitter-python/src/parser.c
  third_party/tree-sitter-python/src/scanner.c)
target_include_directories(tree_sitter_python PRIVATE third_party/tree-sitter-python/src)

add_library(vedit_core STATIC
  cpp/edits/parse.cpp
  cpp/edits/program.cpp
  cpp/edits/edits.cpp
  cpp/edits/prompt.cpp
  cpp/runtime/nfa.cpp
  cpp/runtime/grammar.cpp)
target_include_directories(vedit_core PUBLIC cpp)
target_link_libraries(vedit_core PUBLIC tree_sitter tree_sitter_python)

# pybind11 2.13.6 (pip); configure with -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
find_package(pybind11 2.13.6 CONFIG REQUIRED)
pybind11_add_module(vedit cpp/bindings/module.cpp)
target_link_libraries(vedit PRIVATE vedit_core)
set_target_properties(vedit PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

option(VEDIT_BUILD_TESTS "Build GoogleTest unit tests (libgtest-dev 1.14.0)" ON)
if(VEDIT_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(vedit_tests cpp/tests/test_edits.cpp cpp/tests/test_grammar.cpp)
  target_link_libraries(vedit_tests PRIVATE vedit_core GTest::gtest_main)
  add_test(NAME vedit_tests COMMAND vedit_tests)
endif()
