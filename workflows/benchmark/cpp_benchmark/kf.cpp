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

  po::options_description desc;
  desc.add_options()("input", po::value(&inputDir), "input directory")(
      "odd", po::value(&oddFiles.front()), "ODD XML file")(
      "matmap", po::value(&rootMaterialMap), "ROOT material map")(
      "digicfg", po::value(&digiFile), "Digitization config file");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  // Check if there are enough args or if --help is given
  if (!vm.count("input") || !vm.count("odd") || !vm.count("matmap")) {
    std::cerr << "Error in command line options!\n";
    std::cerr << desc << "\n";
    return 1;
  }

  auto makeKf = [&](const auto &trkGeo, const auto &bfield){
    return ActsExamples::makeKalmanFitterFunction(
        trkGeo, bfield);
  };

  run(makeKf, inputDir, oddFiles, rootMaterialMap, digiFile);
}

