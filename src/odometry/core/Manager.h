#pragma once

#include <map>
#include <mutex>
#include <string>
#include <condition_variable>

#include "odometry/core/SensorData.h"
#include "odometry/core/ManagerOptions.h"
#include "odometry/track/TrackBase.h"

class Manager {
public:
  Manager(ManagerOptions &params_);
  ~Manager();

  ManagerOptions get_params() { return params; }

  void feed_measurement_imu(const ImuData &message);
  void feed_measurement_camera(const CameraData &cam0, const CameraData &cam1);

private:
  ManagerOptions params;
  std::map<int, double> camera_last_timestamp;

  std::queue<ImuData> imu_queue;
  std::deque<CameraData> camera_queue;

  std::condition_variable con;
  std::mutex queue_buf;

  std::atomic<bool> thread_update_running;
  
  int pub_count           = 1;
  bool init_feature       = 0;
  double last_imu_time    = 0;
  bool first_image_flag   = true;
  double first_image_time = 0;
  double last_image_time  = 0;

  // Feature
  std::shared_ptr<TrackBase> trackFEATS;
};