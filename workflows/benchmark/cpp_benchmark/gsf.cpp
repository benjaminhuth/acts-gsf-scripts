#include <filesystem>
#include <iostream>

#include <boost/program_options.hpp>

#include "common.hpp"

namespace po = boost::program_options;

int main(int argc, char **argv) {
  std::string inputDir;
  std::vector<std::string> oddFiles(1);
  std::string rootMaterialMap;
  std::string digiFile;
  std::size_t maxComponents = 1;

  po::options_description desc;
  desc.add_options()("input", po::value(&inputDir), "input directory")(
      "odd", po::value(&oddFiles.front()), "ODD XML file")(
      "matmap", po::value(&rootMaterialMap), "ROOT material map")(
      "digicfg", po::value(&digiFile), "Digitization config file")(
      "components", po::value(&maxComponents), "Max components");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  // Check if there are enough args or if --help is given
  if (!vm.count("input") || !vm.count("odd") || !vm.count("matmap")) {
    std::cerr << "Error in command line options!\n";
    std::cerr << desc << "\n";
    return 1;
  }

  auto makeGsf = [&](const auto &trkGeo, const auto &bfield){
    double weightCutoff = 1.e-4;
    auto logger = Acts::getDefaultLogger("Gsf", Acts::Logging::ERROR);

    const std::filesystem::path base =
        "/home/benjamin/Documents/athena/Tracking/TrkFitter/"
        "TrkGaussianSumFilter/Data/";

    auto bhapprox = ActsExamples::BetheHeitlerApprox::loadFromFiles(
        base / "GeantSim_LT01_cdf_nC6_O5.par",
        base / "GeantSim_GT01_cdf_nC6_O5.par");

    return ActsExamples::makeGsfFitterFunction(
        trkGeo, bfield, bhapprox, maxComponents, weightCutoff,
        Acts::ComponentMergeMethod::eMaxWeight, ActsExamples::MixtureReductionAlgorithm::KLDistance, *logger);
  };

  run(makeGsf, inputDir, oddFiles, rootMaterialMap, digiFile);
}
