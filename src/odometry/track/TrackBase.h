#pragma once

#include <mutex>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <opencv2/opencv.hpp>

class TrackBase {

public:

  enum HistogramMethod { NONE, HISTOGRAM, CLAHE };

  TrackBase();

  virtual ~TrackBase() {}

protected:
  HistogramMethod histogram_method;

  std::atomic<size_t> currid;
  std::mutex mtx_last_vars;
  std::vector<std::mutex> mtx_feeds;

  std::map<size_t, cv::Mat> img_last;
  std::map<size_t, cv::Mat> img_mask_last;

  std::unordered_map<size_t, std::vector<size_t>> ids_last;
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> pts_last;
};
