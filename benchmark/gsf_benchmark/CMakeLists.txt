find_package(Acts REQUIRED COMPONENTS Core)
find_package(TBB REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

set(ACTS_SRC_DIR "/home/benjamin/Documents/acts_project/acts")
set(ACTS_LIB_DIR "/home/benjamin/Documents/acts_project/install/lib64")

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

add_executable(gsf_benchmark main.cpp)

target_link_options(gsf_benchmark PRIVATE -Wall -Wpedantic)
# target_compile_options(gsf_benchmark BEFORE PRIVATE -fPIC)

target_link_libraries(gsf_benchmark
    ActsExamplesFramework
    ActsExamplesDD4hep
    ActsExamplesIoCsv
    ActsExamplesFitting
    ActsExamplesTruthTracking
    # OpenDataDetector
    ActsCore
    ActsPluginDD4hep
    ActsPluginFpeMonitoring
    ActsPluginJson
    TBB::tbb
    Boost::program_options
    nlohmann_json::nlohmann_json
)
