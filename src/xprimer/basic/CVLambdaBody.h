#pragma once

#include <functional>
#include <opencv2/opencv.hpp>

class LambdaBody : public cv::ParallelLoopBody {
public:
  explicit LambdaBody(const std::function<void(const cv::Range &)> &body) { _body = body; }
  void operator()(const cv::Range &range) const override { _body(range); }

private:
  std::function<void(const cv::Range &)> _body;
};