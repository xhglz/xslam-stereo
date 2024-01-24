#pragma once

#include <vector>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

struct ImuData {
  double timestamp;

  // angular velocity (rad/s)
  Eigen::Matrix<double, 3, 1> wm;

  // linear acceleration (m/s^2)
  Eigen::Matrix<double, 3, 1> am;

  bool operator<(const ImuData &other) const { return timestamp < other.timestamp; }
};

struct CameraData {
  double timestamp;

  std::vector<int> sensor_ids;
  std::vector<cv::Mat> images;
  std::vector<cv::Mat> masks;

  bool operator<(const CameraData &other) const {
    if (timestamp == other.timestamp) {
      int id = *std::min_element(sensor_ids.begin(), sensor_ids.end());
      int id_other = *std::min_element(other.sensor_ids.begin(), other.sensor_ids.end());
      return id < id_other;
    } else {
      return timestamp < other.timestamp;
    }
  }
};