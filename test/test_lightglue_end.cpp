#include <opencv2/opencv.hpp>

#include "onnx/BaseOnnxRunner.h"
#include "onnx/SPGLEnd2EndORT.h"
#include "image/ImageProcess.h"

#include "common/Timer.h"
#include "common/YamlConfig.h"
#include "common/Visualizer.h"
#include "common/AccumulateAverage.h"

std::vector<cv::Mat> readImage(std::vector<cv::String> image_file_vec, bool grayscale = false) {
  int mode = cv::IMREAD_COLOR;
  if ( grayscale ) {
      mode = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
  }

  std::vector<cv::Mat> image_matlist;
  for (const auto& file : image_file_vec) {
    cv::Mat image = cv::imread(file, mode);
    if (image.empty()) {
      throw std::runtime_error("[ERROR] Could not read image at " + file);
    }
    if (!grayscale) {
      cv::cvtColor( image, image, cv::COLOR_BGR2RGB );  // BGR -> RGB
    }
    image_matlist.emplace_back( image );
  }

  return image_matlist;
}

int main( int argc, char const* argv[] ) {
  std::string verbosity = "DEBUG";
  Printer::setPrintLevel(verbosity);
  
  Timer timer;
  AccumulateAverage accumulate_average_timer;

  std::string config_path = "config/kitti.yaml";
  auto cfg = std::make_shared<YamlConfig>(config_path);

  std::vector<cv::String> image_file_left_vec;
  std::vector<cv::String> image_file_right_vec;

  // Read image file path
  cv::glob(cfg->image_left_path, image_file_left_vec);
  cv::glob(cfg->image_right_path, image_file_right_vec);

  // Read image
  if (image_file_left_vec.size() != image_file_right_vec.size()) {
    PRINT_ERROR(RED "image src number: %d\n" RESET, image_file_left_vec.size());
    PRINT_ERROR(RED "image dst number: %d\n" RESET, image_file_right_vec.size());
    throw std::runtime_error( "[ERROR] The number of images in the left and right folders is not equal" );
    return EXIT_FAILURE;
  }

  std::vector<cv::Mat> image_left_mat_vec = readImage(image_file_left_vec, cfg->gray_flag);
  std::vector<cv::Mat> image_right_mat_vec = readImage(image_file_right_vec, cfg->gray_flag);

    
  // end2end
  SPGLEnd2EndORT* feature_matcher;
  feature_matcher = new SPGLEnd2EndORT{0};
  feature_matcher->initOrtEnv(cfg);
  feature_matcher->setMatchThreshold(cfg->threshold);

  // inference
  int    count = 0;
  double time_consumed;
  auto   iter_src = image_left_mat_vec.begin();
  auto   iter_dst = image_right_mat_vec.begin();
  for ( ; iter_src != image_left_mat_vec.end(); ++iter_src, ++iter_dst) {
    PRINT_INFO("processing image %d / %d\n", image_file_left_vec[count], image_file_right_vec[count]);
    count++;
    timer.tic();
    auto key_points_result = feature_matcher->inferenceImagePair(cfg, *iter_src, *iter_dst);
    time_consumed          = timer.tocGetDuration();
    accumulate_average_timer.addValue(time_consumed);
    PRINT_INFO("time consumed: %.3f / %.3f ms\n", time_consumed, accumulate_average_timer.getAverage());

    auto key_points_src = feature_matcher->getKeyPointsSrc();
    auto key_points_dst = feature_matcher->getKeyPointsDst();

    visualizeMatches(*iter_src, *iter_dst, key_points_result, key_points_src, key_points_dst);
  }
  return 0;
}
