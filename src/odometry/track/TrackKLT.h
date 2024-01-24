#pragma once

#include <map>
#include "TrackBase.h"

class TrackKLT : public TrackBase {

public:
  explicit TrackKLT(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                    HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist)
      : TrackBase(cameras, numfeats, numaruco, stereo, histmethod), threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
        min_px_dist(minpxdist) {}

  void feed_new_camera(const CameraData &message) override;

protected:
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);

  void perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);

  void perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                        std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out);

  int threshold;
  int grid_x;
  int grid_y;

  int min_px_dist;

  int pyr_levels = 5;
  cv::Size win_size = cv::Size(15, 15);
  

  std::map<size_t, cv::Mat> img_curr;
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;
};