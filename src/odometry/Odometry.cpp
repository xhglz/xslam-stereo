#include <chrono>
#include "odometry/Odometry.h"
#include "common/DataBase.h"

namespace myslam {

Odometry::Odometry(const OdometryOptions &params, const std::shared_ptr<DataBase> &data) {
    this->params_ = params;
    this->data_ = data;
    Init();
}

bool Odometry::Init() {
    // create components and links
    frontend_ = std::make_shared<Frontend>(params_);
    backend_ = std::make_shared<Backend>();
    map_ = std::make_shared<Map>();
    viewer_ = std::make_shared<Viewer>();

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetViewer(viewer_);
    frontend_->SetCameras(data_->GetCamera(0), data_->GetCamera(1));

    backend_->SetMap(map_);
    backend_->SetCameras(data_->GetCamera(0), data_->GetCamera(1));

    viewer_->SetMap(map_);

    return true;
}

void Odometry::Run() {
    PRINT_DEBUG("VO is running\n");
    while (true) {
        if (Step() == false) {
            break;
        }
    }
    backend_->Stop();
    viewer_->Close();

    PRINT_DEBUG("\nVO exit\n");
}

bool Odometry::Step() {
    Frame::Ptr new_frame = data_->NextFrame();
    if (new_frame == nullptr) {
        PRINT_DEBUG("No input data\n");
        return false;
    }

    // cv::imshow("frame", new_frame->left_img_);
    // cv::waitKey(0);
    // PRINT_DEBUG("VO Step\n");
    // return false;

    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend_->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    PRINT_INFO("VO cost time: %.3f seconds.\n", time_used.count());
    return success;
}

}  // namespace myslam
