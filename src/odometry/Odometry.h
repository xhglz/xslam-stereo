#pragma once

#include "common/DataBase.h"
#include "odometry/Viewer.h"
#include "odometry/MathTypes.h"
#include "odometry/Frontend.h"
#include "odometry/Backend.h"
#include "odometry/OdometryOptions.h"

namespace myslam {

class Odometry {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Odometry> Ptr;

    Odometry(const OdometryOptions &params, const std::shared_ptr<DataBase> &data);

    void Run();

    bool Step();

    /// 获取前端状态
    FrontendStatus GetFrontendStatus() const {return frontend_->GetStatus();}

private:
    bool Init();

    bool inited_ = false;

    OdometryOptions params_;

    std::string config_file_path_;

    Frontend::Ptr frontend_ = nullptr;

    Backend::Ptr backend_ = nullptr;

    Map::Ptr map_ = nullptr;
    
    Viewer::Ptr viewer_ = nullptr;

    DataBase::Ptr data_ = nullptr;
};

}