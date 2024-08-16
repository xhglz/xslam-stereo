#include <cmath>
#include <opencv2/opencv.hpp>

#include "odometry/Algorithm.h"
#include "odometry/Feature.h"
#include "odometry/Frontend.h"
#include "odometry/Viewer.h"
#include "odometry/Backend.h"
#include "odometry/g2o_types.h"

#include "common/PrintDebug.h"
#include "common/YamlConfig.h"


namespace myslam {

Frontend::Frontend(const OdometryOptions &params) {
    if (params.track_method == "GFTT") {
        trackFEATS = std::make_shared<FeatsGFTT>(params.num_features);
        // std::shared_ptr<FeatsBase>(new FeatsGFTT(params.num_features));
    } else if (params.track_method == "SUPERPOINT") {
        auto cfg = std::make_shared<YamlConfig>(params.dl_yaml_path);
        trackFEATS = std::make_shared<FeatsSuperPoint>(cfg);
    }
    // gftt_ = cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    // num_features_init_ = Config::Get<int>("num_features_init");
    // num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    current_frame_ = std::move(frame);

    switch (status_) {
        case FrontendStatus::INITING: {
            PRINT_DEBUG("Status: INITING\n");
            StereoInit();
            break;
        }
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD: {
            Track();
            break;
        }
        case FrontendStatus::LOST: {
            Reset();
            break;
        }
    }

    last_frame_ =  std::move(current_frame_);
    return true;
}

bool Frontend::Track() {
    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    int num_track_last = trackFEATS->TrackLastFrame(last_frame_, current_frame_, camera_left_);
    PRINT_DEBUG("TrackLastFrame num track last %d\n", num_track_last);

    tracking_inliers_ = EstimateCurrentPose();

    PRINT_DEBUG(YELLOW "tracking_inliers %d\n" RESET, tracking_inliers_);
    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // std::exit(0);
        // lost
        status_ = FrontendStatus::LOST;
    }

    InsertKeyframe();
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    if (viewer_) {
        viewer_->AddCurrentFrame(current_frame_);
    }
    return true;
}

bool Frontend::InsertKeyframe() {
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, don't insert keyframe
        return false;
    }
    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    PRINT_DEBUG("Set frame %d as keyframe %d\n", current_frame_->id_, current_frame_->keyframe_id_);

    SetObservationsForKeyFrame();

    // detect new features
    trackFEATS->DetectFeatures(current_frame_);  

    // track in right image
    trackFEATS->FindFeaturesInRight(current_frame_, camera_right_);

    // triangulate map points
    TriangulateNewPoints();

    // update backend because we have a new keyframe
    backend_->UpdateMap();

    if (viewer_) {
        viewer_->UpdateMap();
    }

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

int Frontend::TriangulateNewPoints() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
            std::vector<Vec3> points {
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;
                float px = pworld(0, 0);
                float py = pworld(1, 0);
                float pz = pworld(2, 0);
                if (std::isnan(px) || std::isnan(py)  || std::isnan(pz)) {
                    continue;
                } else {
                    new_map_point->SetPos(pworld);
                    new_map_point->AddObservation(current_frame_->features_left_[i]);
                    new_map_point->AddObservation(current_frame_->features_right_[i]);

                    current_frame_->features_left_[i]->map_point_ = new_map_point;
                    current_frame_->features_right_[i]->map_point_ = new_map_point;
                    // PRINT_DEBUG(YELLOW "map point \t" RESET)
                    // std::cout << new_map_point->Pos() << std::endl;
                    map_->InsertMapPoint(new_map_point);
                    cnt_triangulated_pts++;
                }
            }
        }
    }
    PRINT_DEBUG("new landmarks: %d\n", cnt_triangulated_pts);
    return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(toVec2(current_frame_->features_left_[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    // PRINT_INFO("Outlier/Inlier in pose estimating: %d /\ %d\n", cnt_outlier, features.size() - cnt_outlier);
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());

    // LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
    return 0;
}

// int Frontend::TrackLastFrame() {
    // // use LK flow to estimate points in the right image
    // std::vector<cv::Point2f> kps_last, kps_current;
    // for (auto &kp : last_frame_->features_left_) {
    //     if (kp->map_point_.lock()) {
    //         // use project point
    //         auto mp = kp->map_point_.lock();
    //         auto px =
    //             camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
    //         kps_last.push_back(kp->position_.pt);
    //         kps_current.push_back(cv::Point2f(px[0], px[1]));
    //     } else {
    //         kps_last.push_back(kp->position_.pt);
    //         kps_current.push_back(kp->position_.pt);
    //     }
    // }

    // std::vector<uchar> status;
    // cv::Mat error;
    // cv::calcOpticalFlowPyrLK(
    //     last_frame_->left_img_, current_frame_->left_img_, kps_last,
    //     kps_current, status, error, cv::Size(11, 11), 3,
    //     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
    //     cv::OPTFLOW_USE_INITIAL_FLOW);

    // int num_good_pts = 0;

    // for (size_t i = 0; i < status.size(); ++i) {
    //     if (status[i]) {
    //         cv::KeyPoint kp(kps_current[i], 7);
    //         Feature::Ptr feature(new Feature(current_frame_, kp));
    //         feature->map_point_ = last_frame_->features_left_[i]->map_point_;
    //         current_frame_->features_left_.push_back(feature);
    //         num_good_pts++;
    //     }
    // }

    // PRINT_DEBUG("Find %d in the last image.\n", num_good_pts);
    // return num_good_pts;
// }

bool Frontend::StereoInit() {
    int num_features_left = trackFEATS->DetectFeatures(current_frame_);
    PRINT_DEBUG("DetectFeatures num features left %d\n", num_features_left);

    int num_coor_features = trackFEATS->FindFeaturesInRight(current_frame_, camera_right_);
    if (num_coor_features < num_features_init_) {
        return false;
    }

    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }
    return false;
}

// int Frontend::DetectFeatures() {
    // cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    // for (auto &feat : current_frame_->features_left_) {
    //     cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
    //                   feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    // }

    // std::vector<cv::KeyPoint> keypoints;
    // gftt_->detect(current_frame_->left_img_, keypoints, mask);
    // int cnt_detected = 0;
    // for (auto &kp : keypoints) { 
    //     current_frame_->features_left_.push_back(
    //         Feature::Ptr(new Feature(current_frame_, kp)));
    //     cnt_detected++;
    // }

    // PRINT_DEBUG("Detect %d new features\n", cnt_detected)
    // return cnt_detected;
// }

// int Frontend::FindFeaturesInRight() {
    // // use LK flow to estimate points in the right image
    // std::vector<cv::Point2f> kps_left, kps_right;
    // for (auto &kp : current_frame_->features_left_) {
    //     kps_left.push_back(kp->position_.pt);
    //     auto mp = kp->map_point_.lock();
    //     if (mp) {
    //         // use projected points as initial guess
    //         auto px = camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
    //         kps_right.push_back(cv::Point2f(px[0], px[1]));
    //     } else {
    //         // use same pixel in left iamge
    //         kps_right.push_back(kp->position_.pt);
    //     }
    // }

    // std::vector<uchar> status;
    // cv::Mat error;
    // cv::calcOpticalFlowPyrLK(
    //     current_frame_->left_img_, current_frame_->right_img_, kps_left,
    //     kps_right, status, error, cv::Size(11, 11), 3,
    //     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
    //     cv::OPTFLOW_USE_INITIAL_FLOW);

    // int num_good_pts = 0;
    // for (size_t i = 0; i < status.size(); ++i) {
    //     if (status[i]) {
    //         cv::KeyPoint kp(kps_right[i], 7);
    //         Feature::Ptr feat(new Feature(current_frame_, kp));
    //         feat->is_on_left_image_ = false;
    //         current_frame_->features_right_.push_back(feat);
    //         num_good_pts++;
    //     } else {
    //         current_frame_->features_right_.push_back(nullptr);
    //     }
    // }
    // PRINT_DEBUG("Find %d in the right image.\n", num_good_pts);
    // return num_good_pts;
// }

bool Frontend::BuildInitMap() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr) continue;
        // create map point from triangulation
        std::vector<Vec3> points{
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero();

        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    backend_->UpdateMap();

    PRINT_DEBUG("Initial map created with %d map points\n", cnt_init_landmarks);
    return true;
}

bool Frontend::Reset() {
    PRINT_DEBUG("Reset is not implemented.\n");
    return true;
}

}  // namespace myslam