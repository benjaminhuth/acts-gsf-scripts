#include "Acts/ActsVersion.hpp"
#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Plugins/Json/JsonMaterialDecorator.hpp"
#include "Acts/TrackFitting/BetheHeitlerApprox.hpp"
#include "ActsExamples/DD4hepDetector/DD4hepDetector.hpp"
#include "ActsExamples/DD4hepDetector/DD4hepGeometryService.hpp"
#include "ActsExamples/EventData/MeasurementCalibration.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/RandomNumbers.hpp"
#include "ActsExamples/Framework/Sequencer.hpp"
#include "ActsExamples/Io/Csv/CsvMeasurementReader.hpp"
#include "ActsExamples/Io/Csv/CsvParticleReader.hpp"
#include "ActsExamples/Io/Csv/CsvSimHitReader.hpp"
#include "ActsExamples/TrackFitting/TrackFittingAlgorithm.hpp"
#include "ActsExamples/TruthTracking/ParticleSmearing.hpp"
#include "ActsExamples/TruthTracking/TruthSeedSelector.hpp"
#include "ActsExamples/TruthTracking/TruthTrackFinder.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

using namespace Acts::UnitLiterals;
namespace po = boost::program_options;

int main(int argc, char **argv) {
  std::string inputDir;
  std::vector<std::string> oddFiles(1);
  std::string jsonMaterialMap;

  po::options_description desc;
  desc.add_options()("input", po::value(&inputDir), "input directory")(
      "odd", po::value(&oddFiles.front()), "ODD XML file")(
      "matmap", po::value(&jsonMaterialMap), "json material map");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  // Check if there are enough args or if --help is given
  if (!vm.count("input") || !vm.count("odd") || !vm.count("matmap")) {
    std::cerr << desc << "\n";
    return 1;
  }

  auto bfield =
      std::make_shared<Acts::ConstantBField>(Acts::Vector3{0., 0., 2._T});

  ActsExamples::DD4hep::DD4hepGeometryService::Config oddCfg;
  oddCfg.xmlFileNames = oddFiles;

  Acts::MaterialMapJsonConverter::Config matCfg;
  auto matDec = std::make_shared<Acts::JsonMaterialDecorator>(
      matCfg, jsonMaterialMap, Acts::Logging::VERBOSE);

  ActsExamples::DD4hep::DD4hepDetector odd{};
  const auto [trkGeo, dec] = odd.finalize(oddCfg, matDec);

  auto rng = std::make_shared<ActsExamples::RandomNumbers>(
      ActsExamples::RandomNumbers::Config{42});

  ActsExamples::Sequencer::Config seqCfg;
  seqCfg.events = -1;
  seqCfg.numThreads = 1;
  seqCfg.trackFpes = false;

  ActsExamples::Sequencer s(seqCfg);

  for (const auto &d : dec) {
    s.addContextDecorator(d);
  }

  const std::string kParticles = "particles";
  const std::string kParticlesSelected = "particles-sel";
  const std::string kHits = "simhits";
  const std::string kMeasurements = "measurements";
  const std::string kMeasurmentSimhitMap = "meas-simhit-map";
  const std::string kMeasurmentParticleMap = "meas-particle-map";
  const std::string kSourceLinks = "sourcelinks";
  const std::string kParameters = "pars";
  const std::string kProtoTracks = "prototracks";

  {
    ActsExamples::CsvParticleReader::Config cfg;
    cfg.inputDir = inputDir;
    cfg.inputStem = "particles_initial";
    cfg.outputParticles = kParticles;
    s.addReader(std::make_shared<ActsExamples::CsvParticleReader>(
        cfg, Acts::Logging::INFO));
  }

  {
    ActsExamples::CsvSimHitReader::Config cfg;
    cfg.inputDir = inputDir;
    cfg.inputStem = "hits";
    cfg.outputSimHits = kHits;
    s.addReader(std::make_shared<ActsExamples::CsvSimHitReader>(
        cfg, Acts::Logging::INFO));
  }

  {
    ActsExamples::CsvMeasurementReader::Config cfg;
    cfg.inputDir = inputDir;
    cfg.inputSimHits = kHits;
    cfg.outputMeasurements = kMeasurements;
    cfg.outputSourceLinks = kSourceLinks;
    cfg.outputMeasurementSimHitsMap = kMeasurmentSimhitMap;
    cfg.outputMeasurementParticlesMap = kMeasurmentParticleMap;
    s.addReader(std::make_shared<ActsExamples::CsvMeasurementReader>(
        cfg, Acts::Logging::INFO));
  }

  {
    ActsExamples::TruthSeedSelector::Config cfg;
    cfg.inputParticles = kParticles;
    cfg.outputParticles = kParticlesSelected;
    cfg.inputMeasurementParticlesMap = kMeasurmentParticleMap;
    cfg.nHitsMin = 9;
    s.addAlgorithm(std::make_shared<ActsExamples::TruthSeedSelector>(
        cfg, Acts::Logging::INFO));
  }

  {
    ActsExamples::ParticleSmearing::Config cfg;
    cfg.inputParticles = kParticlesSelected;
    cfg.randomNumbers = rng;
    cfg.outputTrackParameters = kParameters;
    s.addAlgorithm(std::make_shared<ActsExamples::ParticleSmearing>(
        cfg, Acts::Logging::INFO));
  }

  {
    ActsExamples::TruthTrackFinder::Config cfg;
    cfg.inputMeasurementParticlesMap = kMeasurmentParticleMap;
    cfg.inputParticles = kParticlesSelected;
    cfg.outputProtoTracks = kProtoTracks;
    s.addAlgorithm(std::make_shared<ActsExamples::TruthTrackFinder>(
        cfg, Acts::Logging::INFO));
  }

  {
    std::size_t maxComponents = 12;
    double weightCutoff = 1.e-4;
    auto logger = Acts::getDefaultLogger("Gsf", Acts::Logging::VERBOSE);

    auto gsf = ActsExamples::makeGsfFitterFunction(
        trkGeo, bfield, Acts::Experimental::makeDefaultBetheHeitlerApprox(),
        maxComponents, weightCutoff, Acts::MixtureReductionMethod::eMaxWeight,
        false, false, *logger);

    ActsExamples::TrackFittingAlgorithm::Config cfg;
    cfg.inputProtoTracks = kProtoTracks;
    cfg.inputInitialTrackParameters = kParameters;
    cfg.inputMeasurements = kMeasurements;
    cfg.pickTrack = 3;
    cfg.inputSourceLinks = kSourceLinks;
    cfg.calibrator = std::make_shared<ActsExamples::PassThroughCalibrator>();
    cfg.outputTracks = "tracks";
    cfg.fit = gsf;

    s.addAlgorithm(std::make_shared<ActsExamples::TrackFittingAlgorithm>(
        cfg, Acts::Logging::INFO));
  }

  s.run();
}
