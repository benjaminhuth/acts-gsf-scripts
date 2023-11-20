cmake_minimum_required(VERSION 3.16)

project(gsf_cpp_benchmark)

set(CMAKE_CXX_STANDARD 17)

find_package(Acts REQUIRED COMPONENTS Core)
find_package(TBB REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

if(NOT DEFINED CACHE{ACTS_SRC_DIR})
    message(FATAL_ERROR "ACTS_SRC_DIR not set")
endif()

if(NOT DEFINED CACHE{ACTS_INSTALL_DIR})
    message(FATAL_ERROR "ACTS_INSTALL_DIR not set")
endif()

set(ACTS_LIB_DIR "${ACTS_INSTALL_DIR}/lib64")

add_library(ActsExamplesFramework SHARED IMPORTED)
add_dependencies(ActsExamplesFramework ActsCore)
set_target_properties(ActsExamplesFramework PROPERTIES IMPORTED_LOCATION "${ACTS_LIB_DIR}/libActsExamplesFramework.so")
set_target_properties(ActsExamplesFramework PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Framework/include")

add_library(ActsExamplesDD4hep SHARED IMPORTED)
add_dependencies(ActsExamplesDD4hep ActsCore)
set_target_properties(ActsExamplesDD4hep PROPERTIES IMPORTED_LOCATION "${ACTS_LIB_DIR}/libActsExamplesDetectorDD4hep.so")
set_target_properties(ActsExamplesDD4hep PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Detectors/DD4hepDetector/include")

add_library(ActsExamplesIoCsv SHARED IMPORTED)
add_dependencies(ActsExamplesIoCsv ActsCore)
set_target_properties(ActsExamplesIoCsv PROPERTIES IMPORTED_LOCATION "${ACTS_LIB_DIR}/libActsExamplesIoCsv.so")
set_target_properties(ActsExamplesIoCsv PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Io/Csv/include")

add_library(ActsExamplesDigitization SHARED IMPORTED)
add_dependencies(ActsExamplesDigitization ActsCore)
set_target_properties(ActsExamplesDigitization PROPERTIES IMPORTED_LOCATION "${ACTS_LIB_DIR}/libActsExamplesDigitization.so")
set_target_properties(ActsExamplesDigitization PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Algorithms/Digitization/include")

add_library(ActsExamplesIoRoot SHARED IMPORTED)
add_dependencies(ActsExamplesIoRoot ActsCore)
set_target_properties(ActsExamplesIoRoot PROPERTIES IMPORTED_LOCATION "${ACTS_LIB_DIR}/libActsExamplesIoRoot.so")
set_target_properties(ActsExamplesIoRoot PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Io/Root/include")

add_library(ActsExamplesIoJson SHARED IMPORTED)
add_dependencies(ActsExamplesIoJson ActsCore)
set_target_properties(ActsExamplesIoJson PROPERTIES IMPORTED_LOCATION "${ACTS_LIB_DIR}/libActsExamplesIoJson.so")
set_target_properties(ActsExamplesIoJson PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Io/Json/include")

add_library(ActsExamplesFitting SHARED IMPORTED)
add_dependencies(ActsExamplesFitting ActsCore)
set_target_properties(ActsExamplesFitting PROPERTIES IMPORTED_LOCATION "${ACTS_LIB_DIR}/libActsExamplesTrackFitting.so")
set_target_properties(ActsExamplesFitting PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Algorithms/TrackFitting/include")

add_library(ActsExamplesTruthTracking SHARED IMPORTED)
add_dependencies(ActsExamplesTruthTracking ActsCore)
set_target_properties(ActsExamplesTruthTracking PROPERTIES IMPORTED_LOCATION "${ACTS_LIB_DIR}/libActsExamplesTruthTracking.so")
set_target_properties(ActsExamplesTruthTracking PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Algorithms/TruthTracking")

# add_library(OpenDataDetector SHARED IMPORTED)
# set_target_properties(OpenDataDetector PROPERTIES IMPORTED_LOCATION "${ACTS_LIB_DIR}/libOpenDataDetector.so")

add_executable(gsf_benchmark gsf.cpp)
add_executable(kf_benchmark kf.cpp)

target_link_options(gsf_benchmark PRIVATE -Wall -Wpedantic)
target_link_options(kf_benchmark PRIVATE -Wall -Wpedantic)
target_compile_options(gsf_benchmark BEFORE PRIVATE -fno-omit-frame-pointer)
target_compile_options(kf_benchmark BEFORE PRIVATE -fno-omit-frame-pointer)

target_link_libraries(gsf_benchmark
    ActsExamplesFramework
    ActsExamplesDD4hep
    ActsExamplesIoCsv
    ActsExamplesFitting
    ActsExamplesTruthTracking
    ActsExamplesIoRoot
    ActsExamplesIoJson
    ActsExamplesDigitization
    # OpenDataDetector
    ActsCore
    ActsPluginDD4hep
    ActsPluginFpeMonitoring
    ActsPluginJson
    TBB::tbb
    Boost::program_options
    nlohmann_json::nlohmann_json
)

target_link_libraries(kf_benchmark
    ActsExamplesFramework
    ActsExamplesDD4hep
    ActsExamplesIoCsv
    ActsExamplesFitting
    ActsExamplesTruthTracking
    ActsExamplesIoRoot
    ActsExamplesIoJson
    ActsExamplesDigitization
    # OpenDataDetector
    ActsCore
    ActsPluginDD4hep
    ActsPluginFpeMonitoring
    ActsPluginJson
    TBB::tbb
    Boost::program_options
    nlohmann_json::nlohmann_json
)
