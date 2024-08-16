#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include "common/PrintDebug.h"
#include "odometry/MathTypes.h"
#include "odometry/Feature.h"
#include "odometry/Mappoint.h"
#include "odometry/Frame.h"

namespace myslam {

class FeatsBase {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<FeatsBase> Ptr;

    FeatsBase(){}

    FeatsBase(int num_feats) {}

    virtual int DetectFeatures(Frame::Ptr &frame) = 0;

    virtual int FindFeaturesInRight(Frame::Ptr &frame, Camera::Ptr &camera_right) = 0;

    virtual int TrackLastFrame(Frame::Ptr &last_frame, Frame::Ptr &current_frame, Camera::Ptr &camera_left) = 0;

    virtual ~FeatsBase() {}
};


class FeatsGFTT : public FeatsBase {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<FeatsGFTT> Ptr;

    explicit FeatsGFTT(int num_feats) : FeatsBase(num_feats) {
        gftt = cv::GFTTDetector::create(num_feats, 0.01, 20);
    }

    int DetectFeatures(Frame::Ptr &frame) {
        cv::Mat mask(frame->left_img_.size(), CV_8UC1, 255);
        for (auto &feat : frame->features_left_) {
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                        feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
        }
        std::vector<cv::KeyPoint> keypoints;
        gftt->detect(frame->left_img_, keypoints, mask);
        int cnt_detected = 0;
        for (auto &kp : keypoints) { 
            frame->features_left_.push_back(Feature::Ptr(new Feature(frame, kp)));
            cnt_detected++;
        }

        PRINT_DEBUG("Detect %d new features\n", cnt_detected)
        return cnt_detected;
    }

    int FindFeaturesInRight(Frame::Ptr &frame, Camera::Ptr &camera_right) {
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_left, kps_right;
        for (auto &kp : frame->features_left_) {
            kps_left.push_back(kp->position_.pt);
            // 如果对象还活着，返回一个shared_ptr
            auto mp = kp->map_point_.lock();
            // std::shared_ptr<MapPoint> mp = kp->map_point_.lock();
            if (mp) {
                // use projected points as initial guess
                auto px = camera_right->world2pixel(mp->Pos(), frame->Pose());
                kps_right.push_back(cv::Point2f(px[0], px[1]));
            } else {
                // use same pixel in left iamge
                kps_right.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status;
        cv::Mat error;
        cv::calcOpticalFlowPyrLK(
            frame->left_img_, frame->right_img_, kps_left,
            kps_right, status, error, cv::Size(11, 11), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                cv::KeyPoint kp(kps_right[i], 7);
                Feature::Ptr feat(new Feature(frame, kp));
                feat->is_on_left_image_ = false;
                frame->features_right_.push_back(feat);
                num_good_pts++;
            } else {
                frame->features_right_.push_back(nullptr);
            }
        }
        PRINT_DEBUG("Find %d good pts in the right image.\n", num_good_pts);
        return num_good_pts;
    }

    int TrackLastFrame(Frame::Ptr &last_frame, Frame::Ptr &current_frame, Camera::Ptr &camera_left) {
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_last, kps_current;
        for (auto &kp : last_frame->features_left_) {
            if (kp->map_point_.lock()) {
                // use project point
                auto mp = kp->map_point_.lock();
                auto px = camera_left->world2pixel(mp->Pos(), current_frame->Pose());
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(cv::Point2f(px[0], px[1]));
            } else {
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(kp->position_.pt);
            }
        }
        std::vector<uchar> status;
        cv::Mat error;
        cv::calcOpticalFlowPyrLK(
            last_frame->left_img_, current_frame->left_img_, kps_last,
            kps_current, status, error, cv::Size(11, 11), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;

        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                cv::KeyPoint kp(kps_current[i], 7);
                Feature::Ptr feature(new Feature(current_frame, kp));
                feature->map_point_ = last_frame->features_left_[i]->map_point_;
                current_frame->features_left_.push_back(feature);
                num_good_pts++;
            }
        }
        PRINT_DEBUG("Find %d good pts in the last image.\n", num_good_pts);
        return num_good_pts;
    }
private:
    cv::Ptr<cv::GFTTDetector> gftt;
};


}