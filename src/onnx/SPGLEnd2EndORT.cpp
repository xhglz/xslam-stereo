#include <thread>
#include "onnx/SPGLEnd2EndORT.h"
#include "common/PrintDebug.h"

SPGLEnd2EndORT::SPGLEnd2EndORT(unsigned int threads_num) : threads_num_(threads_num) {
  PRINT_DEBUG("SPGLEnd2EndORT created\n");
}

SPGLEnd2EndORT::~SPGLEnd2EndORT() {
  PRINT_DEBUG("SPGLEnd2EndORT destroyed\n");

  for (char* input_node_name : input_node_names_) {
    delete[] input_node_name;
  }
  input_node_names_.clear();
}


int SPGLEnd2EndORT::initOrtEnv(const std::shared_ptr<YamlConfig> &config) {
    PRINT_DEBUG("InitOrtEnv start\n");

    try {
      env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SPGLEnd2EndORT");

      // create session options
      session_options_ = Ort::SessionOptions();
      if (threads_num_ == 0) {
        threads_num_ = std::thread::hardware_concurrency();
      }
      session_options_.SetIntraOpNumThreads( threads_num_ );
      session_options_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_ALL );
      PRINT_DEBUG("Using %d threads, with graph optimization level %d\n", threads_num_, GraphOptimizationLevel::ORT_ENABLE_ALL);

      if(config->device == "cuda") {
        PRINT_DEBUG("CUDA provider with default options\n");

        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id                 = 0;                              // 这行设置 CUDA 设备 ID 为 0，这意味着 ONNX Runtime 将在第一个 CUDA 设备（通常是第一个 GPU）上运行模型。
        cuda_options.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchDefault;  // 这行设置 cuDNN 卷积算法搜索策略为默认值。cuDNN 是 NVIDIA 的深度神经网络库，它包含了许多用于卷积的优化算法。
        cuda_options.gpu_mem_limit             = 0;                              // 这行设置 GPU 内存限制为 0，这意味着 ONNX Runtime 可以使用所有可用的 GPU 内存。
        cuda_options.arena_extend_strategy     = 1;                              // 这行设置内存分配策略为 1，这通常意味着 ONNX Runtime 将在需要更多内存时扩展内存池。
        cuda_options.do_copy_in_default_stream = 1;                              // 行设置在默认流中进行复制操作为 1，这意味着 ONNX Runtime 将在 CUDA 的默认流中进行数据复制操作。
        cuda_options.has_user_compute_stream   = 0;                              // 这行设置用户计算流为 0，这意味着 ONNX Runtime 将使用其自己的计算流，而不是用户提供的计算流。
        cuda_options.default_memory_arena_cfg  = nullptr;                        // 这行设置默认内存区配置为 nullptr，这意味着 ONNX Runtime 将使用默认的内存区配置。

        session_options_.AppendExecutionProvider_CUDA( cuda_options );
        session_options_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_EXTENDED );
      }

      // const char* model_path = config.superpoint_lightglue_path.c_str();
      PRINT_DEBUG("Loading model from %s\n", config->superpoint_lightglue_path.c_str());
      session_uptr_ = std::make_unique<Ort::Session>(env_, config->superpoint_lightglue_path.c_str(), session_options_);

      // get input node names
      PRINT_DEBUG("input node names and shapes\n");
      extractNodesInfo( IO{ INPUT }, input_node_names_, input_node_shapes_, session_uptr_, allocator_);
      PRINT_DEBUG("output node names and shapes\n");
      extractNodesInfo( IO{ OUTPUT }, output_node_names_, output_node_shapes_, session_uptr_, allocator_);

      PRINT_DEBUG("ONNX Runtime environment initialized successfully!\n");
    } catch (const std::exception& e) {
      PRINT_ERROR("ONNX Runtime environment initialized failed! Error message: %s\n", e.what());
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

cv::Mat SPGLEnd2EndORT::preProcess(const std::shared_ptr<YamlConfig> &config, const cv::Mat& image_src, float& scale) {
  float scale_temp = scale;

  cv::Mat image_temp = image_src.clone();
  PRINT_DEBUG("image_temp size, width: %d, height: %d\n", image_temp.cols, image_temp.rows);

  std::string fn{"max"};
  std::string interp{"area"};
  cv::Mat     image_result = normalizeImage(resizeImage( image_temp, config->image_size, scale, fn, interp ) );

  PRINT_DEBUG("image_result size, width: %d, height: %d, scale from %.3f to %.3f\n", 
              image_result.cols, image_result.rows, scale_temp, scale);

  return image_result;
}

int SPGLEnd2EndORT::inference(const std::shared_ptr<YamlConfig> &config, const cv::Mat& image_src, const cv::Mat& image_dst) {
  PRINT_DEBUG("**** inference start ****\n");

  try {
    PRINT_DEBUG("image_src shape: [%d, %d], channel: %d\n", image_src.cols, image_src.rows, image_src.channels());
    PRINT_DEBUG("image_dst shape: [%d, %d], channel: %d\n", image_dst.cols, image_dst.rows, image_dst.channels());

    PRINT_DEBUG("input node names size:%d\n", input_node_names_.size());
    for (const auto& name : input_node_names_) {
      PRINT_DEBUG("****input node name: %s\n", name);
    }

    PRINT_DEBUG("output node names size:%d\n", output_node_names_.size())
    for (const auto& name : output_node_names_) {
      PRINT_DEBUG("****output node name: %s\n", name);
    }

    // build source input node shape and destination input node shape
    // only support super point
    PRINT_DEBUG("creating input tensors\n");
    input_node_shapes_[0] = {1, 1, image_src.size().height, image_src.size().width};
    input_node_shapes_[1] = {1, 1, image_dst.size().height, image_dst.size().width};
    int input_tensor_size_src = input_node_shapes_[0][0] * input_node_shapes_[0][1] * input_node_shapes_[0][2] * input_node_shapes_[0][3];
    int input_tensor_size_dst = input_node_shapes_[1][0] * input_node_shapes_[1][1] * input_node_shapes_[1][2] * input_node_shapes_[1][3];

    std::vector<float> input_tensor_values_src( input_tensor_size_src );
    std::vector<float> input_tensor_values_dst( input_tensor_size_dst );

    input_tensor_values_src.assign( image_src.begin<float>(), image_src.end<float>() );
    input_tensor_values_dst.assign( image_dst.begin<float>(), image_dst.end<float>() );

    // create input tensor object from data values
    PRINT_DEBUG("creating memory info handler\n");
    auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault );
    // Ort::MemoryInfo         memory_info_handler( "Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault );
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, input_tensor_values_src.data(), input_tensor_values_src.size(), input_node_shapes_[ 0 ].data(), input_node_shapes_[ 0 ].size() ) );
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, input_tensor_values_dst.data(), input_tensor_values_dst.size(), input_node_shapes_[ 1 ].data(), input_node_shapes_[ 1 ].size() ) );

    auto timer_tic          = std::chrono::high_resolution_clock::now();
    auto output_tensor_temp = session_uptr_->Run( Ort::RunOptions{ nullptr }, input_node_names_.data(), input_tensors.data(), input_tensors.size(), output_node_names_.data(), output_node_names_.size() );
    auto timer_toc          = std::chrono::high_resolution_clock::now();
    auto diff               = std::chrono::duration_cast<std::chrono::milliseconds>( timer_toc - timer_tic ).count();
    PRINT_DEBUG("inference time: %f ms\n", diff);

    int count = 0;
    for (const auto& tensor : output_tensor_temp) {
      if (!tensor.IsTensor()) {
        PRINT_ERROR("Output %d is not a tensor\n", count);
      }

      if (!tensor.HasValue()) {
        PRINT_ERROR("Output %d is empty\n", count);
      }

      count++;
    }

    output_tensors_ = std::move( output_tensor_temp );
  } catch (const std::exception& e) {
    PRINT_ERROR("inference failed with error message: %s\n", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


int SPGLEnd2EndORT::postProcess(const std::shared_ptr<YamlConfig> &config) {
  try {
    PRINT_DEBUG("**** postProcess start ****\n");
    PRINT_DEBUG("output tensors size: %d\n", output_tensors_.size() );

    std::vector<int64_t> key_points_0_shape = output_tensors_[ 0 ].GetTensorTypeAndShapeInfo().GetShape();  // shape: [1, num_key_points, 2]
    int64_t*             key_points_0_ptr   = (int64_t*)output_tensors_[ 0 ].GetTensorMutableData<void>();
    PRINT_DEBUG("key points 0 shape: %d, %d, %d, size: %d\n", key_points_0_shape[0], key_points_0_shape[1], key_points_0_shape[2], key_points_0_shape.size());
  
    std::vector<int64_t> key_points_1_shape = output_tensors_[ 1 ].GetTensorTypeAndShapeInfo().GetShape();  // shape: [1, num_key_points, 2]
    int64_t*             key_points_1_ptr   = (int64_t*)output_tensors_[ 1 ].GetTensorMutableData<void>();
    PRINT_DEBUG("key points 1 shape: [%d, %d, %d], size: %d\n", key_points_1_shape[0], key_points_1_shape[1], key_points_1_shape[2], key_points_1_shape.size());


    std::vector<int64_t> matches_0_shape = output_tensors_[2].GetTensorTypeAndShapeInfo().GetShape();  // shape: [num_matches, 2]
    int64_t*             matches_0_ptr   = (int64_t*)output_tensors_[2].GetTensorMutableData<void>();
    PRINT_DEBUG("matches 0 shape: [%d, %d], size: %d\n", matches_0_shape[0], matches_0_shape[1], matches_0_shape.size());

    std::vector<int64_t> match_scores_0_shape = output_tensors_[ 3 ].GetTensorTypeAndShapeInfo().GetShape();  // shape: [num_matches]
    float*               match_scores_0_ptr   = (float*)output_tensors_[ 3 ].GetTensorMutableData<void>();
    PRINT_DEBUG("matches score shape: [%d, %d], size: %d\n", match_scores_0_shape[0], match_scores_0_shape[1], match_scores_0_shape.size());

    // process key point
    std::vector<cv::Point2f> key_points_0_tmp, key_points_1_tmp;
    for (int i = 0; i < key_points_0_shape[1] * 2; i += 2 ) {
      key_points_0_tmp.emplace_back(cv::Point2f((key_points_0_ptr[i] + 0.5f ) / scales_[0] - 0.5f, (key_points_0_ptr[i+1] + 0.5f) / scales_[0] - 0.5f));
    }
    for (int i = 0; i < key_points_1_shape[1] * 2; i += 2 ) {
      key_points_1_tmp.emplace_back(cv::Point2f((key_points_1_ptr[i] + 0.5f ) / scales_[1] - 0.5f, (key_points_1_ptr[i+1] + 0.5f) / scales_[1] - 0.5f));
    }

    // create matches indices
    std::set<std::pair<int, int>> matches;
    int                           count = 0;
    for (int i = 0; i < matches_0_shape[ 0 ] * 2; i += 2 ) {
      if (matches_0_ptr[ i ] > -1 && match_scores_0_ptr[ count ] > match_threshold_) {
        // PRINT_DEBUG("matche pair index: [%d, %d] score: %d\n", matches_0_ptr[i], matches_0_ptr[i + 1], match_scores_0_ptr[count]);
        matches.insert(std::make_pair(matches_0_ptr[i], matches_0_ptr[i + 1]));
      }
      count++;
    }
    PRINT_DEBUG("matches size: %d\n", matches.size());

    std::vector<cv::Point2f> key_points_0, key_points_1;

    for ( const auto& match : matches ) {
      key_points_0.emplace_back( key_points_0_tmp[ match.first ] );
      key_points_1.emplace_back( key_points_1_tmp[ match.second ] );
    }

    key_points_result_.first  = key_points_0;
    key_points_result_.second = key_points_1;
    key_points_src_           = key_points_0_tmp;
    key_points_dst_           = key_points_1_tmp;

    PRINT_DEBUG("**** postProcess success ****\n" );
  } catch ( const std::exception& e ) {
    PRINT_DEBUG("**** postProcess failed with error message: %s ****\n", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
SPGLEnd2EndORT::inferenceImagePair(const std::shared_ptr<YamlConfig> &config, const cv::Mat& image_src, const cv::Mat& image_dst) {
  if (image_src.empty() || image_dst.empty()) {
    PRINT_ERROR(RED "inference failed, image is empty.\n" RESET);
    throw std::runtime_error("image is empty");
  }

  cv::Mat image_src_copy = image_src.clone();
  cv::Mat image_dst_copy = image_dst.clone();
  cv::Mat image_src_temp = preProcess(config, image_src_copy, scales_[0]);
  cv::Mat image_dst_temp = preProcess(config, image_dst_copy, scales_[1]);

  int inference_result = inference(config, image_src_temp, image_dst_temp);
  if (inference_result != EXIT_SUCCESS) {
    PRINT_ERROR(RED "inference failed\n" RESET);
    return std::make_pair( std::vector<cv::Point2f>(), std::vector<cv::Point2f>() );
  }

  int post_process_result = postProcess(config);
  if ( post_process_result != EXIT_SUCCESS ) {
    PRINT_ERROR(RED "inference failed\n" RESET);
    return std::make_pair( std::vector<cv::Point2f>(), std::vector<cv::Point2f>() );
  }

  output_tensors_.clear();
  return getKeyPointsResult();
}

float SPGLEnd2EndORT::getMatchThreshold() const {
  return match_threshold_;
}

void SPGLEnd2EndORT::setMatchThreshold(float match_threshold) {
  match_threshold_ = match_threshold;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> SPGLEnd2EndORT::getKeyPointsResult() const {
  return key_points_result_;
}

std::vector<cv::Point2f> SPGLEnd2EndORT::getKeyPointsSrc() const {
  return key_points_src_;
}

std::vector<cv::Point2f> SPGLEnd2EndORT::getKeyPointsDst() const {
  return key_points_dst_;
}