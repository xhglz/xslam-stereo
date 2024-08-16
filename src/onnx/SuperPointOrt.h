#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "onnx/BaseOnnxRunner.h"
#include "image/ImageProcess.h"
#include "image/KeyPointsType.h"

#include "common/Timer.h"
#include "common/YamlConfig.h"
#include "common/AccumulateAverage.h"

class SuperPointOrt : public BaseOnnxRunner {
private:
  unsigned int m_threads_num;
   
  Ort::Env                          m_env;
  Ort::SessionOptions               m_session_options;
  std::unique_ptr<Ort::Session>     m_uptr_session;

  Ort::AllocatorWithDefaultOptions  m_allocator;

  std::vector<char*>                m_vec_input_names;
  std::vector<std::vector<int64_t>> m_vec_input_shapes;

  std::vector<char*>                m_vec_output_names;
  std::vector<std::vector<int64_t>> m_vec_output_shapes;

  KeyPoints m_key_points;

  Timer             m_timer;
  AccumulateAverage m_average_timer;

  //   std::vector<float> m_scale{ 1.0f, 1.0f };
  float        m_scale{ 1.0f };
  int          m_height_transformed{ 0 };
  int          m_width_transformed{ 0 };
  unsigned int m_point_num{ 0 };

private:
  cv::Mat   prePorcess(const std::shared_ptr<YamlConfig> &config, const cv::Mat& image, float& scale);
  int       inference(const std::shared_ptr<YamlConfig> &config, const cv::Mat& image);
  KeyPoints postProcess(const std::shared_ptr<YamlConfig> &config, std::vector<Ort::Value> tensor);

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
public:
  explicit SuperPointOrt(unsigned int threads_num = 0, unsigned int point_num = 0 );
  ~SuperPointOrt();

  int initOrtEnv(const std::shared_ptr<YamlConfig> &config) override;

  KeyPoints inferenceImage(const std::shared_ptr<YamlConfig> &config, const cv::Mat& image);
  KeyPoints getKeyPoints() const;
  float     getScale() const;
  int       getHeightTransformed() const;
  int       getWidthTransformed() const;

  std::pair<KeyPoints, std::vector<Region>> distributeKeyPoints(const KeyPoints& key_points, const cv::Mat& image);
};
