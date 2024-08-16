#pragma once

#include <string>
#include <fstream>
#include "common/CVYamlParse.h"

class YamlConfig {
public:
  std::string lightglue_onnx  = "models/superpoint_lightglue_fused_fp16.onnx";  
  std::string superpoint_onnx = "models/superpoint.onnx";                 
  std::string superpoint_lightglue_path = "models/superpoint_lightglue_end2end_fused.onnx";

  std::string image_left_path  = "data/left";
  std::string image_right_path = "data/right";

  std::string output_path     = "output";
  bool gray_flag              = true;

  int image_size              = 512;
  float threshold             = 0.05f;

  std::string device{"cuda"};

public:
  YamlConfig(const std::string config_path) {
    auto parser = std::make_shared<YamlParser>(config_path);
    if (parser != nullptr) {
      print_and_load(parser);
    }
  }
  ~YamlConfig() = default;

private:
  void print_and_load(const std::shared_ptr<YamlParser> &parser = nullptr) {
    parser->parse_config("lightglue_onnx", lightglue_onnx);
    parser->parse_config("superpoint_onnx", superpoint_onnx);
    parser->parse_config("superpoint_lightglue_path", superpoint_lightglue_path);

    parser->parse_config("image_left_path",  image_left_path);
    parser->parse_config("image_right_path", image_right_path);

    parser->parse_config("output_path", output_path);
    parser->parse_config("gray_flag", gray_flag);
 
    parser->parse_config("image_size", image_size);
    parser->parse_config("threshold", threshold);

    parser->parse_config("device", device);

    PRINT_DEBUG("PARAMETERS:\n");
    PRINT_DEBUG("  - lightglue_onnx: %s\n", lightglue_onnx.c_str());
    PRINT_DEBUG("  - superpoint_onnx: %s\n", superpoint_onnx.c_str());
    PRINT_DEBUG("  - superpoint_lightglue_path: %s\n", superpoint_lightglue_path.c_str());
    PRINT_DEBUG("  - image_left_path: %s\n",  image_left_path.c_str());    
    PRINT_DEBUG("  - image_right_path: %s\n", image_right_path.c_str());    
    PRINT_DEBUG("  - output_path: %s\n", output_path.c_str());    
    PRINT_DEBUG("  - gray_flag: %d\n", gray_flag);
    PRINT_DEBUG("  - image_size: %d\n", image_size);
    PRINT_DEBUG("  - image_size: %.3f\n", threshold);
    PRINT_DEBUG("  - device: %s\n", device.c_str());    
  }
};