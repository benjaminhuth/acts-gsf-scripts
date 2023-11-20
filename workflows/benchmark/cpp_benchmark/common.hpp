#pragma once

#include "Acts/ActsVersion.hpp"
#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/TrackFitting/BetheHeitlerApprox.hpp"
#include "ActsExamples/DD4hepDetector/DD4hepDetector.hpp"
#include "ActsExamples/DD4hepDetector/DD4hepGeometryService.hpp"
#include "ActsExamples/Digitization/DigitizationAlgorithm.hpp"
#include "ActsExamples/Digitization/DigitizationConfig.hpp"
#include "ActsExamples/EventData/MeasurementCalibration.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/RandomNumbers.hpp"
#include "ActsExamples/Framework/Sequencer.hpp"
#include "ActsExamples/Io/Csv/CsvMeasurementReader.hpp"
#include "ActsExamples/Io/Csv/CsvParticleReader.hpp"
#include "ActsExamples/Io/Csv/CsvSimHitReader.hpp"
#include "ActsExamples/Io/Json/JsonDigitizationConfig.hpp"
#include "ActsExamples/Io/Root/RootMaterialDecorator.hpp"
#include "ActsExamples/TrackFitting/TrackFittingAlgorithm.hpp"
#include "ActsExamples/TruthTracking/ParticleSmearing.hpp"
#include "ActsExamples/TruthTracking/TruthSeedSelector.hpp"
#include "ActsExamples/TruthTracking/TruthTrackFinder.hpp"

#include <memory>
#include <string>
#include <vector>

struct Hook : public Acts::GeometryIdentifierHook {
  std::map<uint64_t, std::vector<double>> map = {
      {28, {850.0}},         // LStrip negative z
      {30, {850.0}},         // LStrip positive z
      {23, {400.0, 550.0}},  // SStrip negative z
      {25, {400.0, 550.0}},  // SStrip positive z
      {16, {100.0}},         // Pixels negative z
      {18, {100.0}}          // Pixels positive z
  };

  Acts::GeometryIdentifier decorateIdentifier(
      Acts::GeometryIdentifier geoid,
      const Acts::Surface &surface) const override {
    if (map.find(geoid.volume()) != map.end()) {
      auto r = std::hypot(surface.center(Acts::GeometryContext{})[0],
                          surface.center(Acts::GeometryContext{})[1]);

      geoid.setExtra(1);
      for (auto cut : map.at(geoid.volume())) {
        if (r > cut) {
          geoid.setExtra(geoid.extra() + 1);
        }
      }
    }
    return geoid;
  }
};

using namespace Acts::UnitLiterals;

template<typename make_fitter_t>
void run(const make_fitter_t &fitter, const std::string &inputDir, const std::vector<std::string> &oddFiles, 
         const std::string &rootMaterialMap, const std::string &digiFile) {
  auto bfield =
      std::make_shared<Acts::ConstantBField>(Acts::Vector3{0., 0., 2._T});

  ActsExamples::DD4hep::DD4hepGeometryService::Config oddCfg;
  oddCfg.xmlFileNames = oddFiles;
  oddCfg.geometryIdentifierHook = std::make_shared<Hook>();

  ActsExamples::RootMaterialDecorator::Config matCfg;
  matCfg.fileName = rootMaterialMap;
  auto matDec = std::make_shared<ActsExamples::RootMaterialDecorator>(
      matCfg, Acts::Logging::INFO);

  ActsExamples::DD4hep::DD4hepDetector odd{};
  const auto [trkGeo, dec] = odd.finalize(oddCfg, matDec);

  auto rng = std::make_shared<ActsExamples::RandomNumbers>(
      ActsExamples::RandomNumbers::Config{42});

  ActsExamples::Sequencer::Config seqCfg;
  seqCfg.events = -1;
  seqCfg.numThreads = 1;
  seqCfg.trackFpes = false;
  seqCfg.outputDir = ".";

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
    ActsExamples::DigitizationConfig cfg(
        ActsExamples::readDigiConfigFromJson(digiFile));
    cfg.trackingGeometry = trkGeo;
    cfg.randomNumbers = rng;
    cfg.inputSimHits = kHits;
    cfg.outputMeasurementParticlesMap = kMeasurmentParticleMap;
    cfg.outputMeasurementSimHitsMap = kMeasurmentSimhitMap;
    cfg.outputMeasurements = kMeasurements;
    cfg.outputSourceLinks = kSourceLinks;

    s.addAlgorithm(std::make_shared<ActsExamples::DigitizationAlgorithm>(
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
    ActsExamples::TrackFittingAlgorithm::Config cfg;
    cfg.inputProtoTracks = kProtoTracks;
    cfg.inputInitialTrackParameters = kParameters;
    cfg.inputMeasurements = kMeasurements;
    cfg.inputSourceLinks = kSourceLinks;
    cfg.calibrator = std::make_shared<ActsExamples::PassThroughCalibrator>();
    cfg.outputTracks = "tracks";
    cfg.fit = fitter(trkGeo, bfield);

    s.addAlgorithm(std::make_shared<ActsExamples::TrackFittingAlgorithm>(
        cfg, Acts::Logging::INFO));
  }
  
  s.run();
}
