#include "odometry/YamlConfig.h"

int main(int argc, char const* argv[]) {
  std::string verbosity = "DEBUG";
  Printer::setPrintLevel(verbosity);

  std::string config_path = "config/kitti.yaml";
  auto config = std::make_shared<YamlConfig>(config_path);

  PRINT_DEBUG("test config %s\n", config->image_left_path);
  
  return EXIT_SUCCESS;
}