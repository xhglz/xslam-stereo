#pragma once

#include <queue>
#include <iostream>

#include <csignal>
#include <execinfo.h>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "core/FeatureTrackerOptions.h"

bool inBorder(const cv::Point2f &pt);

// void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
// void reduceVector(vector<int> &v, vector<uchar> status);
                                        
class FeatureTracker {
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img, double _cur_time);

    static int n_id;
};
