#include <iostream>

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/ActsVersion.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/Sequencer.hpp"
#include "ActsExamples/DD4hepDetector/DD4hepDetector.hpp"
#include "ActsExamples/DD4hepDetector/DD4hepGeometryService.hpp"
#include "Acts/Plugins/Json/JsonMaterialDecorator.hpp"
#include "ActsExamples/Io/Csv/CsvMeasurementReader.hpp"
#include "ActsExamples/Io/Csv/CsvParticleReader.hpp"
#include "ActsExamples/Io/Csv/CsvSimHitReader.hpp"

#include <memory>
#include <vector>
#include <string>

using namespace Acts::UnitLiterals;

int main(int argc, char **argv) {
    auto bfield = std::make_shared<Acts::ConstantBField>(Acts::Vector3{0.,0.,2.});

    std::vector<std::string> oddFiles;
    ActsExamples::DD4hep::DD4hepGeometryService::Config oddCfg;
    oddCfg.xmlFileNames = oddFiles;

    std::string jsonMaterialMap;
    Acts::MaterialMapJsonConverter::Config matCfg;
    auto matDec = std::make_shared<Acts::JsonMaterialDecorator>(matCfg, jsonMaterialMap, Acts::Logging::INFO);

    ActsExamples::DD4hep::DD4hepDetector odd{};
    const auto [trkGeo, dec] = odd.finalize(oddCfg, matDec);

    ActsExamples::Sequencer::Config seqCfg;
    seqCfg.events=1;
    seqCfg.numThreads=1;
    seqCfg.trackFpes=false;

    ActsExamples::Sequencer s(seqCfg);

    std::string inputDir;

    const std::string kParticles = "particles";
    const std::string kHits = "simhits";
    const std::string kMeasurements = "measurements";
    const std::string kMeasurmentSimhitMap = "meas-simhit-map";
    const std::string kSourceLinks = "sourcelinks";

    {
        ActsExamples::CsvParticleReader::Config cfg;
        cfg.inputDir = inputDir;
        cfg.inputStem = "particles_initial";
        cfg.outputParticles = kParticles
        s.addReader(std::make_shared<ActsExamples::CsvParticleReader>(cfg, Acts::Logging::INFO));
    }

    {
        ActsExamples::CsvMeasurementReader::Config cfg;
        cfg.inputDir = inputDir;
        cfg.outputMeasurements = kMeasurements;
        cfg.outputSourceLinks = kSourceLinks;
        cfg.outputMeasurementSimHitsMap = kMeasurmentSimhitMap;
        s.addReader(std::make_shared<ActsExamples::CsvMeasurementReader>(cfg, Acts::Logging::INFO));
    }

    {
        ActsExamples::CsvSimHitReader::Config cfg;
        cfg.inputDir = inputDir;
        cfg.inputStem = "hits";
        cfg.outputSimHits = kHits
        s.addReader(std::make_shared<ActsExamples::CsvSimHitReader>(cfg, Acts::Logging::INFO));
    }



}
