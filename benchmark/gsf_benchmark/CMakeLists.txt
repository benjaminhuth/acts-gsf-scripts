find_package(Acts REQUIRED COMPONENTS Core)

set(ACTS_SRC_DIR "/home/benjamin/Documents/acts_project/acts")
set(ACTS_LIB64_DIR "${ACTS_SRC_DIR}/build/lib64")

add_library(ActsExamplesFramework SHARED IMPORTED)
set_target_properties(ActsExamplesFramework PROPERTIES IMPORTED_LOCATION "${ACTS_LIB64_DIR}/libActsExamplesFramework.so")
set_target_properties(ActsExamplesFramework PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Framework/include")

add_library(ActsExamplesDD4hep SHARED IMPORTED)
set_target_properties(ActsExamplesDD4hep PROPERTIES IMPORTED_LOCATION "${ACTS_LIB64_DIR}/libActsExamplesDD4hepDetectorWithOptions.so")
set_target_properties(ActsExamplesDD4hep PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Detectors/DD4hepDetector/include")

add_library(ActsExamplesIoCsv SHARED IMPORTED)
set_target_properties(ActsExamplesIoCsv PROPERTIES IMPORTED_LOCATION "${ACTS_LIB64_DIR}/libActsExamplesIoCsv.so")
set_target_properties(ActsExamplesIoCsv PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Examples/Io/Csv/include")

add_library(ActsExamplesFitting SHARED IMPORTED)
set_target_properties(ActsExamplesFitting PROPERTIES IMPORTED_LOCATION "${ACTS_LIB64_DIR}/libActsExamplesTrackFitting.so")
set_target_properties(ActsExamplesFitting PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Algorithms/TrackFitting/include")

add_library(ActsExamplesTruthTracking SHARED IMPORTED)
set_target_properties(ActsExamplesTruthTracking PROPERTIES IMPORTED_LOCATION "${ACTS_LIB64_DIR}/libActsExamplesTruthTracking.so")
set_target_properties(ActsExamplesTruthTracking PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACTS_SRC_DIR}/Algorithms/TruthTracking/include")

add_executable(gsf_benchmark main.cpp)

target_link_libraries(gsf_benchmark ActsCore ActsPluginDD4hep ActsExamplesFramework ActsExamplesDD4hep ActsExamplesIoCsv ActsExamplesFitting ActsExamplesTruthTracking)
