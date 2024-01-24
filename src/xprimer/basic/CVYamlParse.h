#pragma once

#include <memory>
#include <Eigen/Eigen>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "logger/Print.h"
#include "quat_ops.h"

class YamlParser {
public:
  explicit YamlParser(const std::string &config_path, bool fail_if_not_found = true) : config_path_(config_path) {
    if (!fail_if_not_found && !std::filesystem::exists(config_path)) {
      config = nullptr;
      return;
    }
    if (!std::filesystem::exists(config_path)) {
      PRINT_ERROR(RED "unable to open the configuration file!\n%s\n" RESET, config_path.c_str());
      std::exit(EXIT_FAILURE);
    }

    config = std::make_shared<cv::FileStorage>(config_path, cv::FileStorage::READ);
    if (!fail_if_not_found && !config->isOpened()) {
      config = nullptr;
      return;
    }
    if (!config->isOpened()) {
      PRINT_ERROR(RED "unable to open the configuration file!\n%s\n" RESET, config_path.c_str());
      std::exit(EXIT_FAILURE);
    }
  }

  std::string get_config_folder() { return config_path_.substr(0, config_path_.find_last_of('/')) + "/"; }

  bool successful() const { return all_params_found_successfully; }

  template <class T> void parse_config(const std::string &node_name, T &node_result, bool required = true) {
    parse_config_yaml(node_name, node_result, required);
  }

  template <class T>
  void parse_external(const std::string &external_node_name, const std::string &sensor_name, const std::string &node_name, T &node_result,
                      bool required = true) {
    parse_external_yaml(external_node_name, sensor_name, node_name, node_result, required);
  }

  void parse_external(const std::string &external_node_name, const std::string &sensor_name, const std::string &node_name,
                      Eigen::Matrix3d &node_result, bool required = true) {
    parse_external_yaml(external_node_name, sensor_name, node_name, node_result, required);
  }

  void parse_external(const std::string &external_node_name, const std::string &sensor_name, const std::string &node_name,
                      Eigen::Matrix4d &node_result, bool required = true) {
    parse_external_yaml(external_node_name, sensor_name, node_name, node_result, required);
  }

private:
  std::string config_path_;

  std::shared_ptr<cv::FileStorage> config;

  bool all_params_found_successfully = true;

  static bool node_found(const cv::FileNode &file_node, const std::string &node_name) {
    bool found_node = false;
    for (const auto &item : file_node) {
      if (item.name() == node_name) {
        found_node = true;
      }
    }
    return found_node;
  }

  template <class T> void parse(const cv::FileNode &file_node, const std::string &node_name, T &node_result, bool required = true) {
    if (!node_found(file_node, node_name)) {
      if (required) {
        PRINT_WARNING(YELLOW "the node %s of type [%s] was not found...\n" RESET, node_name.c_str(), typeid(node_result).name());
        all_params_found_successfully = false;
      } else {
        PRINT_DEBUG("the node %s of type [%s] was not found (not required)...\n", node_name.c_str(), typeid(node_result).name());
      }
      return;
    }

    try {
      file_node[node_name] >> node_result;
    } catch (...) {
      if (required) {
        PRINT_WARNING(YELLOW "unable to parse %s node of type [%s] in the config file!\n" RESET, node_name.c_str(),
                      typeid(node_result).name());
        all_params_found_successfully = false;
      } else {
        PRINT_DEBUG("unable to parse %s node of type [%s] in the config file (not required)\n", node_name.c_str(),
                    typeid(node_result).name());
      }
    }
  }

  void parse(const cv::FileNode &file_node, const std::string &node_name, bool &node_result, bool required = true) {
    if (!node_found(file_node, node_name)) {
      if (required) {
        PRINT_WARNING(YELLOW "the node %s of type [%s] was not found...\n" RESET, node_name.c_str(), typeid(node_result).name());
        all_params_found_successfully = false;
      } else {
        PRINT_DEBUG("the node %s of type [%s] was not found (not required)...\n", node_name.c_str(), typeid(node_result).name());
      }
      return;
    }

    try {
      if (file_node[node_name].isInt() && (int)file_node[node_name] == 1) {
        node_result = true;
        return;
      }
      if (file_node[node_name].isInt() && (int)file_node[node_name] == 0) {
        node_result = false;
        return;
      }

      std::string value;
      file_node[node_name] >> value;
      value = value.substr(0, value.find_first_of('#'));
      value = value.substr(0, value.find_first_of(' '));
      if (value == "1" || value == "true" || value == "True" || value == "TRUE") {
        node_result = true;
      } else if (value == "0" || value == "false" || value == "False" || value == "FALSE") {
        node_result = false;
      } else {
        PRINT_WARNING(YELLOW "the node %s has an invalid boolean type of [%s]\n" RESET, node_name.c_str(), value.c_str());
        all_params_found_successfully = false;
      }
    } catch (...) {
      if (required) {
        PRINT_WARNING(YELLOW "unable to parse %s node of type [%s] in the config file!\n" RESET, node_name.c_str(),
                      typeid(node_result).name());
        all_params_found_successfully = false;
      } else {
        PRINT_DEBUG("unable to parse %s node of type [%s] in the config file (not required)\n", node_name.c_str(),
                    typeid(node_result).name());
      }
    }
  }

  void parse(const cv::FileNode &file_node, const std::string &node_name, Eigen::Matrix3d &node_result, bool required = true) {
    if (!node_found(file_node, node_name)) {
      if (required) {
        PRINT_WARNING(YELLOW "the node %s of type [%s] was not found...\n" RESET, node_name.c_str(), typeid(node_result).name());
        all_params_found_successfully = false;
      } else {
        PRINT_DEBUG("the node %s of type [%s] was not found (not required)...\n", node_name.c_str(), typeid(node_result).name());
      }
      return;
    }

    node_result = Eigen::Matrix3d::Identity();
    try {
      for (int r = 0; r < (int)file_node[node_name].size() && r < 3; r++) {
        for (int c = 0; c < (int)file_node[node_name][r].size() && c < 3; c++) {
          node_result(r, c) = (double)file_node[node_name][r][c];
        }
      }
    } catch (...) {
      if (required) {
        PRINT_WARNING(YELLOW "unable to parse %s node of type [%s] in the config file!\n" RESET, node_name.c_str(),
                      typeid(node_result).name());
        all_params_found_successfully = false;
      } else {
        PRINT_DEBUG("unable to parse %s node of type [%s] in the config file (not required)\n", node_name.c_str(),
                    typeid(node_result).name());
      }
    }
  }

  void parse(const cv::FileNode &file_node, const std::string &node_name, Eigen::Matrix4d &node_result, bool required = true) {
    std::string node_name_local = node_name;
    if (node_name == "T_cam_imu" && !node_found(file_node, node_name)) {
      PRINT_INFO("parameter T_cam_imu not found, trying T_imu_cam instead (will return T_cam_imu still)!\n");
      node_name_local = "T_imu_cam";
    } else if (node_name == "T_imu_cam" && !node_found(file_node, node_name)) {
      PRINT_INFO("parameter T_imu_cam not found, trying T_cam_imu instead (will return T_imu_cam still)!\n");
      node_name_local = "T_cam_imu";
    }

    if (!node_found(file_node, node_name_local)) {
      if (required) {
        PRINT_WARNING(YELLOW "the node %s of type [%s] was not found...\n" RESET, node_name_local.c_str(), typeid(node_result).name());
        all_params_found_successfully = false;
      } else {
        PRINT_DEBUG("the node %s of type [%s] was not found (not required)...\n", node_name_local.c_str(), typeid(node_result).name());
      }
      return;
    }

    node_result = Eigen::Matrix4d::Identity();
    try {
      for (int r = 0; r < (int)file_node[node_name_local].size() && r < 4; r++) {
        for (int c = 0; c < (int)file_node[node_name_local][r].size() && c < 4; c++) {
          node_result(r, c) = (double)file_node[node_name_local][r][c];
        }
      }
    } catch (...) {
      if (required) {
        PRINT_WARNING(YELLOW "unable to parse %s node of type [%s] in the config file!\n" RESET, node_name.c_str(),
                      typeid(node_result).name());
        all_params_found_successfully = false;
      } else {
        PRINT_DEBUG("unable to parse %s node of type [%s] in the config file (not required)\n", node_name.c_str(),
                    typeid(node_result).name());
      }
    }

    if (node_name_local != node_name) {
      Eigen::Matrix4d tmp(node_result);
      node_result = ov_core::Inv_se3(tmp);
    }
  }

  template <class T> void parse_config_yaml(const std::string &node_name, T &node_result, bool required = true) {
    if (config == nullptr) {
      return;
    }

    try {
      parse(config->root(), node_name, node_result, required);
    } catch (...) {
      PRINT_WARNING(YELLOW "unable to parse %s node of type [%s] in the config file!\n" RESET, node_name.c_str(),
                    typeid(node_result).name());
      all_params_found_successfully = false;
    }
  }

  template <class T>
  void parse_external_yaml(const std::string &external_node_name, const std::string &sensor_name, const std::string &node_name,
                           T &node_result, bool required = true) {

    if (config == nullptr) {
      return;
    }

    std::string path;
    if (!node_found(config->root(), external_node_name)) {
      PRINT_ERROR(RED "the external node %s could not be found!\n" RESET, external_node_name.c_str());
      std::exit(EXIT_FAILURE);
    }
    (*config)[external_node_name] >> path;
    std::string relative_folder = config_path_.substr(0, config_path_.find_last_of('/')) + "/";

    auto config_external = std::make_shared<cv::FileStorage>(relative_folder + path, cv::FileStorage::READ);
    if (!config_external->isOpened()) {
      PRINT_ERROR(RED "unable to open the configuration file!\n%s\n" RESET, (relative_folder + path).c_str());
      std::exit(EXIT_FAILURE);
    }

    if (!node_found(config_external->root(), sensor_name)) {
      PRINT_WARNING(YELLOW "the sensor %s of type [%s] was not found...\n" RESET, sensor_name.c_str(), typeid(node_result).name());
      all_params_found_successfully = false;
      return;
    }

    try {
      parse((*config_external)[sensor_name], node_name, node_result, required);
    } catch (...) {
      PRINT_WARNING(YELLOW "unable to parse %s node of type [%s] in [%s] in the external %s config file!\n" RESET, node_name.c_str(),
                    typeid(node_result).name(), sensor_name.c_str(), external_node_name.c_str());
      all_params_found_successfully = false;
    }
  }
};
