#include <future>
#include <memory>
#include <thread>
#include <opencv2/opencv.hpp>

#include "onnx/BaseOnnxRunner.h"
#include "onnx/SuperPointOrt.h"
#include "onnx/LightGlueOrt.h"

#include "image/ImageProcess.h"

#include "common/Timer.h"
#include "common/YamlConfig.h"
#include "common/Visualizer.h"
#include "common/AccumulateAverage.h"

std::vector<cv::Mat> readImage( std::vector<cv::String> image_file_vec, bool grayscale = false ) {
  int mode = cv::IMREAD_COLOR;
  if ( grayscale ) {
    mode = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
  }

  std::vector<cv::Mat> image_matlist;
  for ( const auto& file : image_file_vec) {
    cv::Mat image = cv::imread(file, mode);
    if ( image.empty() ) {
      throw std::runtime_error("[ERROR] Could not read image at " + file);
    }
    if (!grayscale) {
      cv::cvtColor( image, image, cv::COLOR_BGR2RGB);  // BGR -> RGB
    }
    image_matlist.emplace_back(image);
  }

  return image_matlist;
}

int main( int argc, char const* argv[] ) {
  std::string verbosity = "INFO";
  Printer::setPrintLevel(verbosity);

  std::string config_path = "config/kitti.yaml";
  if ( argc == 2 ) {
    config_path = argv[1];
  }
  auto cfg = std::make_shared<YamlConfig>(config_path);

  Timer timer;
  AccumulateAverage accumulate_average_timer;

  std::vector<cv::String> image_file_src_vec;
  std::vector<cv::String> image_file_dst_vec;

  // Read image file path
  cv::glob(cfg->image_left_path, image_file_src_vec);
  cv::glob(cfg->image_right_path, image_file_dst_vec);

  // Read image
  if (image_file_src_vec.size() != image_file_dst_vec.size()) {
    PRINT_ERROR(RED "image src number: %d\n" RESET, image_file_src_vec.size());
    PRINT_ERROR(RED "image dst number: %d\n" RESET, image_file_dst_vec.size());
    throw std::runtime_error( "[ERROR] The number of images in the left and right folders is not equal" );
    return EXIT_FAILURE;
  }

  std::vector<cv::Mat> image_src_mat_vec = readImage(image_file_src_vec, cfg->gray_flag);
  std::vector<cv::Mat> image_dst_mat_vec = readImage(image_file_dst_vec, cfg->gray_flag);

  std::shared_ptr<SuperPointOrt> extractor_left_ptr = std::make_unique<SuperPointOrt>(6, 200);
  extractor_left_ptr->initOrtEnv( cfg );
  std::shared_ptr<SuperPointOrt> extractor_right_ptr = std::make_unique<SuperPointOrt>(6, 200);
  extractor_right_ptr->initOrtEnv( cfg );

  // matcher
  std::unique_ptr<LightGlueOrt> matcher_ptr = std::make_unique<LightGlueOrt>();
  matcher_ptr->initOrtEnv(cfg);

  // inference
  int    count = 0;
  double time_consumed;
  auto   iter_src = image_src_mat_vec.begin();
  auto   iter_dst = image_dst_mat_vec.begin();
  for ( ; iter_src != image_src_mat_vec.end(); ++iter_src, ++iter_dst) {
    PRINT_DEBUG("processing image %d / %d\n", image_file_src_vec[count], image_file_dst_vec[count]);
    count++;
    timer.tic();

    auto left_future = std::async( std::launch::async, [extractor_left_ptr, cfg, iter_src]() {
      return extractor_left_ptr->inferenceImage(cfg, *iter_src);
    });

    auto right_future = std::async( std::launch::async, [extractor_right_ptr, cfg, iter_dst]() {
      return extractor_right_ptr->inferenceImage(cfg, *iter_dst);
    });

    auto key_points_result_left  = left_future.get();
    auto key_points_result_right = right_future.get();
    // auto key_points_src_distribute = extractor_left_ptr->distributeKeyPoints(key_points_result_left, *iter_src);
    // auto key_points_dst_distribute = extractor_right_ptr->distributeKeyPoints(key_points_result_right, *iter_dst);
    // auto key_points_src = key_points_src_distribute.first.getKeyPoints();
    // auto key_points_dst = key_points_dst_distribute.first.getKeyPoints();

    auto key_points_src = key_points_result_left.getKeyPoints();
    auto key_points_dst = key_points_result_right.getKeyPoints();

    float scale_temp = extractor_left_ptr->getScale();
    matcher_ptr->setParams(std::vector<float>(scale_temp, scale_temp), extractor_left_ptr->getHeightTransformed(), extractor_left_ptr->getWidthTransformed(), 0.0f);
    auto matches_set = matcher_ptr->inferenceDescriptorPair(cfg, key_points_src, key_points_dst, key_points_result_left.getDescriptor(), key_points_result_right.getDescriptor());
    // auto matches_set = matcher_ptr->inferenceDescriptorPair(cfg, key_points_src, key_points_dst, key_points_src_distribute.first.getDescriptor(), key_points_dst_distribute.first.getDescriptor() );

    std::vector<cv::Point2f> key_points_transformed_src = getKeyPointsInOriginalImage(key_points_src, scale_temp);
    std::vector<cv::Point2f> key_points_transformed_dst = getKeyPointsInOriginalImage(key_points_dst, scale_temp);

    std::vector<cv::Point2f> matches_src;
    std::vector<cv::Point2f> matches_dst;
    for (const auto& match : matches_set) {
      matches_src.emplace_back(key_points_transformed_src[match.first]);
      matches_dst.emplace_back(key_points_transformed_dst[match.second]);
    }
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> matches_pair = std::make_pair(matches_src, matches_dst);

    time_consumed = timer.tocGetDuration();
    accumulate_average_timer.addValue(time_consumed);
    PRINT_INFO(BLUE "time consumed(ms): %.3f / %.3f\n" RESET, time_consumed, accumulate_average_timer.getAverage());
    PRINT_INFO(BLUE "key points number src:%d dst:%d, match: %d\n" RESET, key_points_src.size(), key_points_dst.size(), matches_src.size());

    visualizeMatches(*iter_src, *iter_dst, matches_pair, key_points_transformed_src, key_points_transformed_dst);
  }
  return 0;
}
