#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "common/PrintDebug.h"
#include "common/YamlConfig.h"

class BaseOnnxRunner {
public:
  // data
  // Ort::AllocatorWithDefaultOptions allocator_;
  enum IO : u_int8_t {
    INPUT  = 0,
    OUTPUT = 1
  };

  virtual int initOrtEnv(const std::shared_ptr<YamlConfig> &config) {
    return EXIT_SUCCESS;
  }

  virtual float getMatchThreshold() {
    return 0.0f;
  }

  virtual void setMatchThreshold(const float& threshold) {}

  virtual double getTimer(const std::string& name) {
    return 0.0f;
  }

  virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
  inferenceImage(const std::shared_ptr<YamlConfig> &config, const cv::Mat& image_src, const cv::Mat& image_dst) {
    return std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>{};
  }

  virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getKeyPointsResult() {
    return std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>{};
  }

  void extractNodesInfo(const IO& io, std::vector<char*>& node_names, 
                        std::vector<std::vector<int64_t>>& node_shapes, 
                        const std::unique_ptr<Ort::Session>& session_uptr, 
                        const Ort::AllocatorWithDefaultOptions& allocator) {
    if (io != IO::INPUT && io != IO::OUTPUT) {
      throw std::runtime_error("io should be INPUT or OUTPUT");
    }

    size_t num_nodes = (io == IO::INPUT) ? session_uptr->GetInputCount() : session_uptr->GetOutputCount();
    node_names.reserve(num_nodes);
    // PRINT_DEBUG("num_nodes: %d\n", num_nodes);

    auto processNode = [ & ]( size_t i) {
      char* node_name_temp = new char[ std::strlen((io == IO::INPUT ? session_uptr->GetInputNameAllocated(i, allocator) : session_uptr->GetOutputNameAllocated(i, allocator)).get()) + 1];
      std::strcpy(node_name_temp, (io == IO::INPUT ? session_uptr->GetInputNameAllocated(i, allocator) : session_uptr->GetOutputNameAllocated(i, allocator)).get());
      // PRINT_DEBUG("extractor node name: %s\n", node_name_temp);
      node_names.push_back( node_name_temp );
      node_shapes.emplace_back((io == IO::INPUT ? session_uptr->GetInputTypeInfo( i ) : session_uptr->GetOutputTypeInfo(i)).GetTensorTypeAndShapeInfo().GetShape());
    };

    for (size_t i = 0; i < num_nodes; i++) {
      processNode(i);
    }
  }
};
