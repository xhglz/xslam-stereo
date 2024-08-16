#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "onnx/BaseOnnxRunner.h"
#include "common/YamlConfig.h"
#include "image/ImageProcess.h"

class SPGLEnd2EndORT : public BaseOnnxRunner {
private:
  unsigned int threads_num_;

  Ort::Env                          env_;
  Ort::SessionOptions               session_options_;
  std::unique_ptr<Ort::Session>     session_uptr_;
  Ort::AllocatorWithDefaultOptions  allocator_;

  std::vector<char*>                input_node_names_;
  std::vector<std::vector<int64_t>> input_node_shapes_;

  std::vector<char*>                output_node_names_;
  std::vector<std::vector<int64_t>> output_node_shapes_;

  float     match_threshold_{0.0f};
  long long timer_{0};

  std::vector<float> scales_{1.0f, 1.0f};

  std::vector<Ort::Value> output_tensors_;

  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> key_points_result_;
  std::vector<cv::Point2f>                                      key_points_src_;
  std::vector<cv::Point2f>                                      key_points_dst_;


private:
  cv::Mat preProcess(const std::shared_ptr<YamlConfig> &config, const cv::Mat& image, float& scale);

  int inference(const std::shared_ptr<YamlConfig> &config, const cv::Mat& image_src, const cv::Mat& image_dst);

  int postProcess(const std::shared_ptr<YamlConfig> &config);

public:
  explicit SPGLEnd2EndORT(unsigned int threads_num = 1);
  ~SPGLEnd2EndORT();

  
  int initOrtEnv(const std::shared_ptr<YamlConfig> &config = nullptr);

  float getMatchThreshold() const;

  void setMatchThreshold(float match_threshold);

  double getTimer(const std::string& name) const;

  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getKeyPointsResult() const;
  std::vector<cv::Point2f>                                      getKeyPointsSrc()    const;
  std::vector<cv::Point2f>                                      getKeyPointsDst()    const;

  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> inferenceImagePair(const std::shared_ptr<YamlConfig> &config, const cv::Mat& image_src, const cv::Mat& image_dst);
};
