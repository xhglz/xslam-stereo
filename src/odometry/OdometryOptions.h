#pragma once

#include <string>
#include <fstream>

#include "odometry/Camera.h"
#include "common/CVYamlParse.h"

namespace myslam {

struct OdometryOptions {

    std::string dataset_path = "data/kitti/sequences/00";
    std::string dl_yaml_path = "config/kitti.yaml";

    std::string track_method = "GFTT";
    int num_features = 150;
    int num_features_init = 50;
    int num_features_tracking = 50;

    void print_and_load(const std::shared_ptr<YamlParser> &parser = nullptr) {
        PRINT_DEBUG(YELLOW "SLAM PARAMETERS:\n" RESET);
        parser->parse_config("dl_yaml_path", dl_yaml_path);
        parser->parse_config("dataset_path", dataset_path);
        parser->parse_config("track_method", track_method);
        parser->parse_config("num_features", num_features);
        parser->parse_config("num_features_init", num_features_init);
        parser->parse_config("num_features_tracking", num_features_tracking);

        PRINT_DEBUG("  - dl_yaml_path: %s\n", dl_yaml_path.c_str());
        PRINT_DEBUG("  - dataset_path: %s\n", dataset_path.c_str());
        PRINT_DEBUG("  - track_method: %s\n", track_method.c_str());
        PRINT_DEBUG("  - num_features: %d\n", num_features);
        PRINT_DEBUG("  - num_features_init: %d\n", num_features_init);
        PRINT_DEBUG("  - num_features_tracking: %d\n", num_features_tracking);
    }
}; 

}

