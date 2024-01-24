#include <thread>
#include "core/Manager.h"
#include "logger/Print.h"

Manager::Manager(ManagerOptions &params_) {
  this->params = params_;
  thread_update_running = false;
  trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(state->_cam_intrinsics_cameras, init_max_features,
                                                         state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
                                                         params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist));
}

Manager::~Manager() {

}

void Manager::feed_measurement_camera(const CameraData &cam0, const CameraData &cam1) {
  double dStampSec = cam0.timestamp;
  if (!init_feature) {
    PRINT_DEBUG("skip the first frame\n");
    init_feature = 1;
    return;
  }

  if (first_image_flag) {
    first_image_flag = false;
    first_image_time = dStampSec;
    last_image_time  = dStampSec;
    return;
  }

  if (dStampSec - last_image_time > 1.0 || dStampSec < last_image_time) {
    PRINT_ERROR(RED "camera discontinue\n" RESET);
    first_image_flag = true;
    last_image_time = 0;
    pub_count = 1;
    return;
  }
  last_image_time = dStampSec;

  int inputImageCnt;
  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;

  featureFrame = featureTracker.trackImage(t, _img, _img1);
}


void Manager::feed_measurement_imu(const ImuData &message) {
  double dStampSec = message.timestamp;
  if (dStampSec <= last_imu_time) {
    PRINT_ERROR(RED "Wrong IMU message\n" RESET);
    return;
  }
  queue_buf.lock();
  imu_queue.push(message);
  queue_buf.unlock();
  con.notify_one();
}