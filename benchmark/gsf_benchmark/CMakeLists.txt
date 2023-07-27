cmake_minimum_required(VERSION 3.16)

project(gsf_benchmark)

find_package(Acts REQUIRED COMPONENTS Core)

set(ACTS_SRC_DIR "/home/benjamin/Documents/acts_project/acts")
set(ACTS_LIB64_DIR "${ACTS_SRC_DIR}/build/lib64")

add_library(ActsExamplesFramework SHARED IMPORTED)
set_target_properties(ActsExamplesFramework PROPERTIES IMPORTED_LOCATION "${ACTS_LIB64_DIR}/libActsExamplesFramework.so")
set_target_properties(ActsExamplesFramework PROPERTIES INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Framework/include")



add_executable(gsf_benchmark main.cpp)

target_link_libraries(gsf_benchmark ActsCore ActsExamplesFramework)
