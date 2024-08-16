#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "onnx/BaseOnnxRunner.h"
#include "image/KeyPointsType.h"
#include "common/Timer.h"
#include "common/YamlConfig.h"
#include "common/AccumulateAverage.h"

class LightGlueOrt : public BaseOnnxRunner {
private:
  unsigned int threads_num_;
  float match_threshold_ = 0.0f;
  int   height_ = -1;
  int   width_  = -1;
  std::vector<float> vec_sacles_{1.0f, 1.0f};

  Timer             m_timer;

  Ort::Env                          env_;
  Ort::SessionOptions               session_options_;
  std::unique_ptr<Ort::Session>     session_uptr_;
  Ort::AllocatorWithDefaultOptions  allocator_;

  std::vector<char*>                input_node_names_;
  std::vector<std::vector<int64_t>> input_node_shapes_;

  std::vector<char*>                output_node_names_;
  std::vector<std::vector<int64_t>> output_node_shapes_;

  std::set<std::pair<int, int>> matched_indices_;

  std::vector<Ort::Value> output_tensors_;

  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> key_points_result_;
  std::vector<cv::Point2f>                                      key_points_src_;
  std::vector<cv::Point2f>                                      key_points_dst_;

public:
  explicit LightGlueOrt(unsigned int threads_num = 0);
  ~LightGlueOrt(); 

  int initOrtEnv(const std::shared_ptr<YamlConfig> &config = nullptr);

  std::vector<cv::Point2f> preProcess(std::vector<cv::Point2f> kpts, const int height, const int width);

  int inference(const std::shared_ptr<YamlConfig> &config, 
    const std::vector<cv::Point2f> key_points_src, const std::vector<cv::Point2f> key_points_dst, 
    const cv::Mat& descriptor_src, const cv::Mat& descriptor_dst);

  int postProcess(const std::shared_ptr<YamlConfig> &config);

  void setParams(const std::vector<float>& scales, const int& height, const int& width, const float& threshold);

  void  setScales(const std::vector<float>& vec_scales);
  void  setHeight(const int& height);
  void  setWidth(const int& width);
  void  setMatchThreshold(const float& threshold);
  float getMatchThreshold() const;

  std::set<std::pair<int, int>> inferenceDescriptorPair(const std::shared_ptr<YamlConfig> &config, 
              const std::vector<cv::Point2f> key_points_src, const std::vector<cv::Point2f> key_points_dst, 
              const cv::Mat& descriptor_src, const cv::Mat& descriptor_dst);
};