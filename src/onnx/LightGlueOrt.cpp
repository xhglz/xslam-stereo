#include <thread>
#include "common/PrintDebug.h"
#include "onnx/LightGlueOrt.h"
#include "image/ImageProcess.h"

LightGlueOrt::LightGlueOrt(unsigned int threads_num) : threads_num_(threads_num) {
  PRINT_DEBUG("LightGlueOrt is being created\n");
}

LightGlueOrt::~LightGlueOrt() {
  PRINT_DEBUG("LightGlueOrt is being destroyed\n");

  for (auto& name : input_node_names_) {
    delete[] name;
  }
  input_node_names_.clear();

  for (auto& name :  output_node_names_) {
    delete[] name;
  }
  output_node_names_.clear();
}

int LightGlueOrt::initOrtEnv(const std::shared_ptr<YamlConfig> &config) {
  try {
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SPGLEnd2EndORT");

    if (threads_num_ == 0) {
      threads_num_ = std::thread::hardware_concurrency();
    }

    // create session options
    session_options_ = Ort::SessionOptions();
    session_options_.SetIntraOpNumThreads(threads_num_);
    session_options_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_ALL );
    PRINT_DEBUG("Using %d threads, with graph optimization level %d\n", threads_num_, GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (config->device == "cuda") {
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
    PRINT_DEBUG("Loading model from %s\n", config->lightglue_onnx.c_str());
    session_uptr_ = std::make_unique<Ort::Session>(env_, config->lightglue_onnx.c_str(), session_options_);

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

std::vector<cv::Point2f> LightGlueOrt::preProcess(std::vector<cv::Point2f> kpts, const int height, const int width) {
  return normalizeKeyPoints(kpts, height, width);
}

int LightGlueOrt::inference(const std::shared_ptr<YamlConfig> &config, 
  const std::vector<cv::Point2f> key_points_src, const std::vector<cv::Point2f> key_points_dst, 
  const cv::Mat& descriptor_src, const cv::Mat& descriptor_dst) {
 
  try {
    input_node_shapes_[0] = {1, static_cast<int>(key_points_src.size()), 2};
    input_node_shapes_[1] = {1, static_cast<int>(key_points_dst.size()), 2};
    input_node_shapes_[2] = {1, static_cast<int>(key_points_src.size()), 256};
    input_node_shapes_[3] = {1, static_cast<int>(key_points_dst.size()), 256};

    auto   memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    float* key_points_src_data = new float[key_points_src.size() * 2];
    float* key_points_dst_data = new float[key_points_dst.size() * 2];

    for (size_t i = 0; i < key_points_src.size(); ++i) {
      key_points_src_data[i * 2]     = key_points_src[i].x;
      key_points_src_data[i * 2 + 1] = key_points_src[i].y;
    }

    for (size_t i = 0; i < key_points_dst.size(); ++i) {
      key_points_dst_data[i * 2]     = key_points_dst[i].x;
      key_points_dst_data[i * 2 + 1] = key_points_dst[i].y;
    }

    float* descriptor_src_data;
    if ( descriptor_src.isContinuous()) {
      descriptor_src_data = const_cast<float*>( descriptor_src.ptr<float>( 0 ) );
    } else {
      cv::Mat temp_descriptor = descriptor_src.clone();
      descriptor_src_data     = const_cast<float*>( descriptor_src.ptr<float>( 0 ) );
    }

    float* descriptor_dst_data;
    if ( descriptor_dst.isContinuous()) {
      descriptor_dst_data = const_cast<float*>( descriptor_dst.ptr<float>( 0 ) );
    } else {
      cv::Mat temp_descriptor = descriptor_dst.clone();
      descriptor_dst_data     = const_cast<float*>( descriptor_dst.ptr<float>( 0 ) );
    }

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_handler, key_points_src_data, key_points_src.size() * 2, input_node_shapes_[0].data(), input_node_shapes_[0].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_handler, key_points_dst_data, key_points_dst.size() * 2, input_node_shapes_[1].data(), input_node_shapes_[1] .size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_handler, descriptor_src_data, key_points_src.size() * 256, input_node_shapes_[2].data(), input_node_shapes_[2].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_handler, descriptor_dst_data, key_points_dst.size() * 256, input_node_shapes_[3].data(), input_node_shapes_[3].size()));

    auto output_tensor_temp = session_uptr_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), input_tensors.data(), input_tensors.size(), output_node_names_.data(), output_node_names_.size());
    output_tensors_ = std::move(output_tensor_temp);
  } catch (const std::exception& e) {
    PRINT_ERROR(RED "matcher inference failed with error message: {0}: %s\n" RESET, e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;   
}

int LightGlueOrt::postProcess(const std::shared_ptr<YamlConfig> &config) {
  try {
    std::vector<int64_t> matches_shape = output_tensors_[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t*             matches_ptr   = (int64_t*)output_tensors_[0].GetTensorMutableData<void>();

    std::vector<int64_t> scores_shape = output_tensors_[1].GetTensorTypeAndShapeInfo().GetShape();
    float*               scores_ptr   = (float*)output_tensors_[1].GetTensorMutableData<void>();

    std::set<std::pair<int, int>> matches;
    int                           count = 0;
    for ( int i = 0; i < matches_shape[0] * 2; i += 2 ) {
      if (matches_ptr[i] > -1 && matches_ptr[i + 1] > -1 && scores_ptr[count] > match_threshold_) {
        matches.insert(std::make_pair(matches_ptr[i], matches_ptr[i + 1]));
      }
      count++;
    }
    matched_indices_ = matches;
  } catch ( const std::exception& e ) {
    PRINT_ERROR(RED "LightGlueOrt post process failed with error message: %s\n" RESET, e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

std::set<std::pair<int, int>> LightGlueOrt::inferenceDescriptorPair(
                    const std::shared_ptr<YamlConfig> &config, 
                    const std::vector<cv::Point2f> key_points_src, 
                    const std::vector<cv::Point2f> key_points_dst, 
                    const cv::Mat& descriptor_src, const cv::Mat& descriptor_dst) {
  auto key_points_src_norm = preProcess(key_points_src, height_, width_);
  auto key_points_dst_norm = preProcess(key_points_dst, height_, width_);

  m_timer.tic();
  inference(config, key_points_src_norm, key_points_dst_norm, descriptor_src, descriptor_dst);
  postProcess(config);
  PRINT_DEBUG(BLUE "LightGlue cost time(ms): %f\n" RESET, m_timer.tocGetDuration() );

  return matched_indices_;
}

void LightGlueOrt::setScales(const std::vector<float>& scales) {
  vec_sacles_ = scales;
}

void LightGlueOrt::setHeight(const int& height) {
  height_ = height;
}

void LightGlueOrt::setWidth(const int& width) {
  width_ = width;
}

void LightGlueOrt::setMatchThreshold(const float& threshold) {
  match_threshold_ = threshold;
}

float LightGlueOrt::getMatchThreshold() const {
  return match_threshold_;
}

void LightGlueOrt::setParams(const std::vector<float>& scales, const int& height, const int& width, const float& threshold) {
  vec_sacles_      = scales;
  height_          = height;
  width_           = width;
  match_threshold_ = threshold;
}
