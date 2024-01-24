#include <unistd.h>
#include <thread>
#include <vector>
#include <string>
#include <csignal>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "logger/Print.h"
#include "core/Manager.h"
#include "core/ManagerOptions.h"
#include "test/EurocDataset.h"

std::shared_ptr<Manager> sys;

void signal_callback_handler(int signum) { std::exit(signum); }

void publish_stereo(const std::string &data_path, const int &nDelayTimes) {
  CameraCsv cam0_csv, cam1_csv;
  cam0_csv.load(data_path + "/cam0/data.csv");
  cam1_csv.load(data_path + "/cam1/data.csv");
  unsigned int i = 0; 
  unsigned int j = 0;
  unsigned int cnt = 0; 
  const unsigned int FrameCnt = 3; 
  while (i < cam0_csv.items.size()) {
    double cam0_t = cam0_csv.items[i].t;
    double cam1_t = cam1_csv.items[j].t;
    // 双目相机左右图像时差不得超过0.003s
    if (cam0_t < cam1_t - 0.003) {
      i++;
      continue;
    } else if (cam0_t > cam1_t + 0.003) {
      j++;
      continue;
    } else {
      std::string cam0_file = data_path + "/cam0/data/" + cam0_csv.items[i].filename;
      std::string cam1_file = data_path + "/cam1/data/" + cam1_csv.items[j].filename;
      i++;
      j++;
      if (cnt < FrameCnt) {
        cnt++;
      } else {

        // image0_data.emplace_back(cam0_t, cam0_file);
        // image1_data.emplace_back(cam1_t, cam1_file);
        #ifdef XSLAM_DEBUG
          // cv::Mat img = cv::imread(cam0_file, cv::IMREAD_GRAYSCALE);
          // cv::imshow("image", img);
          // cv::waitKey(10);
        #endif
      }
    }
    usleep(50000 * nDelayTimes); // 100 ms
  }
}

void publish_imu(const std::string &data_path, const int &nDelayTimes) {
  ImuCsv imu_csv;
  imu_csv.load(data_path + "/imu0/data.csv");
  for (auto &item : imu_csv.items) {
    ImuData message;
    message.timestamp = item.t;
    message.wm << item.w.x, item.w.y, item.w.z;
    message.am << item.a.x, item.a.y, item.a.z;
    sys->feed_measurement_imu(message);

    #ifdef XSLAM_DEBUG
      PRINT_DEBUG("imu: %f, %f, %f, %f, %f, %f, %f\n", item.t, \
        item.w.x, item.w.y, item.w.z, item.a.x, item.a.y, item.a.z);
    #endif
    usleep(5000 * nDelayTimes); // 10ms 
  }
}


int main(int argc, char **argv) {
  const int nDelayTimes   = 2;
  std::string data_path   = "../dataset/EuRoC/MH_02_easy/mav0";
  std::string config_path = "../config";

	if(argc == 3) {
    data_path   = argv[1];
    config_path = argv[2];
	}

  auto parser = std::make_shared<YamlParser>(config_path);
  std::string verbosity = "DEBUG";
  parser->parse_config("verbosity", verbosity);
  Printer::setPrintLevel(verbosity);

  ManagerOptions params;
  params.print_and_load(parser);
  sys = std::make_shared<Manager>(params);

	std::thread thread_pub_imu(publish_imu, std::ref(data_path), std::ref(nDelayTimes));
	std::thread thread_pub_stereo(publish_stereo, std::ref(data_path), std::ref(nDelayTimes));
	
  thread_pub_imu.join();
	thread_pub_stereo.join();

  signal(SIGINT, signal_callback_handler);
  return EXIT_SUCCESS;
}