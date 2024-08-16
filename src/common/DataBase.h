#pragma once

#include "odometry/Camera.h"
#include "odometry/MathTypes.h"
#include "odometry/Frame.h"

namespace myslam {

class DataBase {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;    
    typedef std::shared_ptr<DataBase> Ptr;
    DataBase(const std::string& dataset_path): dataset_path_(dataset_path) {}
    
    virtual ~DataBase() {}

    virtual bool Init() = 0;

    virtual Frame::Ptr NextFrame() = 0; 
    
    Camera::Ptr GetCamera(int camera_id) const {
        return cameras_.at(camera_id);
    }

protected:
    DataBase() = default;
    std::string dataset_path_;
    int current_image_index_ = 0;

    std::vector<Camera::Ptr> cameras_;
};

}