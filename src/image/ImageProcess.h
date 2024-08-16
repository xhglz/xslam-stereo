#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat normalizeImage(const cv::Mat& image);

std::vector<cv::Point2f> normalizeKeyPoints(std::vector<cv::Point2f> key_points, const int& height, const int& width);

cv::Mat resizeImage(const cv::Mat& image, const int& size, float& scale, const std::string& fn = "max", const std::string& interpolation = "linear");

std::vector<cv::Point2f> getKeyPointsInOriginalImage(const std::vector<cv::Point2f>& key_points, const float scale);