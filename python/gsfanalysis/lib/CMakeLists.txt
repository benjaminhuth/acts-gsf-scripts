cmake_minimum_required(VERSION 3.26)
project(libmode)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
add_library(mode SHARED mode.cpp)

target_compile_features(mode PRIVATE cxx_std_20)
