#pragma once

#include <future>
#include <memory>
#include <thread>
#include <opencv2/opencv.hpp>

#include "odometry/FeatsBase.h"
#include "onnx/BaseOnnxRunner.h"
#include "onnx/SuperPointOrt.h"
#include "onnx/LightGlueOrt.h"

#include "common/Timer.h"
#include "image/ImageProcess.h"
#include "common/YamlConfig.h"

namespace myslam {


class FeatsSuperPoint : public FeatsBase {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<FeatsSuperPoint> Ptr;

    explicit FeatsSuperPoint(std::shared_ptr<YamlConfig> cfg) : FeatsBase() {
        config = std::move(cfg);

        extractor = std::make_unique<SuperPointOrt>(6, 200);
        extractor->initOrtEnv(config);

        matcher = std::make_unique<LightGlueOrt>();
        matcher->initOrtEnv(config);
    }

    int DetectFeatures(Frame::Ptr &frame) {
        // 1.  需要描述子和特征点对应上，省略了根据 mask 提取特征
        cv::Mat mask(frame->left_img_.size(), CV_8UC1, 255);
        for (auto &feat : frame->features_left_) {
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                        feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
        }
        cv::Mat image;
        frame->left_img_.copyTo(image, mask);
        auto features = extractor->inferenceImage(config, image);

        // 2. 归一化后的特征点坐标 
        auto kpts_normal = features.getKeyPoints();
        frame->lFeatsNormal.insert(frame->lFeatsNormal.end(), kpts_normal.begin(), kpts_normal.end());

        // 3. 描述子用于匹配
        cv::Mat desc = features.getDescriptor().clone();
        frame->lDesc.push_back(desc);

        // 4. 图像中的特征点坐标
        std::vector<cv::Point2f> key_points = getKeyPointsInOriginalImage(kpts_normal, extractor->getScale());
        int cnt_detected = 0;
        // cv::point2f -> cv::keypoint 
        for (auto &vkp : key_points) { 
            cv::KeyPoint kp;
            kp.pt.x = vkp.x;
            kp.pt.y = vkp.y;
            kp.size = 7; // 原程序
            frame->features_left_.push_back(Feature::Ptr(new Feature(frame, kp)));
            cnt_detected++;
        }

        // DrawKeyPoints(image, key_points);
        // PRINT_DEBUG("Detect %d new features\n", cnt_detected)

        return cnt_detected;
    }

    int FindFeaturesInRight(Frame::Ptr &frame, Camera::Ptr &camera_right) {
        // 1. 提取右图特征
        auto features = extractor->inferenceImage(config, frame->right_img_);

        // 2. 归一化后的特征点坐标
        auto rFeatsNormal = features.getKeyPoints();
        
        // 3. 图像中的特征点坐标
        std::vector<cv::Point2f> key_points = getKeyPointsInOriginalImage(rFeatsNormal, extractor->getScale());

        // std::cout << "left feats num: " << frame->lFeatsNormal.size() << " right feats num:" << key_points.size() << std::endl;
        // std::cout << "left desc num: " << frame->lDesc.rows << " right desc num:" << features.getDescriptor().rows << std::endl;
        // DrawKeyPoints(frame->right_img_, key_points);

        // 4. 左右目的匹配
        float scale_temp = extractor->getScale();
        matcher->setParams(std::vector<float>(scale_temp, scale_temp), 
                extractor->getHeightTransformed(), extractor->getWidthTransformed(), 0.0f);
        auto matches_set = matcher->inferenceDescriptorPair(
                config, frame->lFeatsNormal, rFeatsNormal, frame->lDesc, features.getDescriptor());

        // 5. 对齐 left 的特征点，兼容原程序三角化 
        // std::cout << "matches num: " << matches_set.size() << std::endl;
        std::vector<int> match_index(frame->lFeatsNormal.size(), -1);
        for(const auto& match : matches_set) {
            match_index[match.first] = match.second;
        }

        int num_good_pts = 0;
        for (size_t i = 0; i < match_index.size(); i++) {
            int idx = match_index[i];
            if (idx > 0) {
                cv::KeyPoint kp(key_points[idx], 7);
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
        // 1. 提取当前帧特征
        auto features = extractor->inferenceImage(config, current_frame->left_img_);

        // 2. 归一化后的特征点坐标
        auto cFeatsNormal = features.getKeyPoints();
        
        // 3. 当前帧的描述子
        cv::Mat cDesc = features.getDescriptor().clone();

        // 4. 图像中的特征点坐标
        std::vector<cv::Point2f> currentKpts = getKeyPointsInOriginalImage(cFeatsNormal, extractor->getScale());

        // 5. 上一帧和当前帧的匹配
        float scale_temp = extractor->getScale();
        matcher->setParams(std::vector<float>(scale_temp, scale_temp), 
                extractor->getHeightTransformed(), extractor->getWidthTransformed(), 0.0f);
        auto matches_set = matcher->inferenceDescriptorPair(
                config, last_frame->lFeatsNormal, cFeatsNormal, last_frame->lDesc, cDesc);
        
        // 6. 对齐归一化特征点 特征点 描述子
        int num_good_pts = 0;   
        // std::vector<cv::Point2f> matches_src;
        // std::vector<cv::Point2f> matches_dst;
        for (const auto& match : matches_set) {
            // matches_src.emplace_back(last_frame->features_left_[match.first]->position_.pt);
            // matches_dst.emplace_back(currentKpts[match.second]);

            current_frame->lFeatsNormal.push_back(cFeatsNormal.at(match.second));
            current_frame->lDesc.push_back(cDesc.row(match.second));
            cv::KeyPoint kp(currentKpts[match.second], 7);
            Feature::Ptr feature(new Feature(current_frame, kp));
            feature->map_point_ = last_frame->features_left_[match.first]->map_point_;
            current_frame->features_left_.push_back(feature);
            num_good_pts++;
        }

        // std::cout << "\n current feats num: " << current_frame->features_left_.size() << " current desc num:" << current_frame->lDesc.rows << std::endl;
        // std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> matches_pair = std::make_pair(matches_src, matches_dst);
        // DrawKeyPoints(current_frame->left_img_, currentKpts);
        // DrawMatches(last_frame->left_img_, current_frame->left_img_, matches_pair);
        
        PRINT_DEBUG("Find %d good pts in the current image.\n", num_good_pts);

        return num_good_pts;
    }

private:
    std::shared_ptr<YamlConfig> config;
    std::shared_ptr<SuperPointOrt> extractor;
    std::unique_ptr<LightGlueOrt> matcher;

    void DrawKeyPoints(const cv::Mat& image_src, const std::vector<cv::Point2f> key_points_src) {
        cv::Mat img_src_color;
        cv::cvtColor(image_src, img_src_color, cv::COLOR_GRAY2BGR);

        for (const auto& point : key_points_src ) {
            cv::circle(img_src_color, point, 2, cv::Scalar( 0, 0, 255 ), -1 );
        }

        // Display the matches
        cv::imshow("DrawKeyPoints", img_src_color);
        cv::waitKey(1);
    }

    void DrawMatches(const cv::Mat& src, const cv::Mat& dst, 
        const std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>& key_points) {

        std::vector<cv::KeyPoint> key_points_1, key_points_2;

        for (const auto& point : key_points.first) {
            key_points_1.push_back( cv::KeyPoint( point, 1.0f ) );
        }
        for (const auto& point : key_points.second){
            key_points_2.push_back(cv::KeyPoint( point, 1.0f ) );
        }

        std::vector<cv::DMatch> matches;
        for ( size_t i = 0; i < key_points_1.size(); ++i) {
            matches.push_back(cv::DMatch(i, i, 0));
        }

        cv::Mat img_matches;
        cv::drawMatches(src, key_points_1, dst, key_points_2, matches, img_matches);

        cv::imshow("Matches", img_matches);
        cv::waitKey(0);
    }

};

}