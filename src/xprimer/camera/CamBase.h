#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>
#include "basic/math_util.h"

class CamBase {
  public:
    virtual ~CamBase() = default;

    CamBase(int width, int height) : width_(width), height_(height) {}

    CamBase(const std::string &name, int width, int height,
          const Eigen::Matrix4f &intrinsic,
          const Eigen::Matrix3f &extrinsic_r,
          const Eigen::Vector3f &extrinsic_t)
        : name_(name), width_(width), height_(height),
          intrinsic_(intrinsic), extrinsic_r_(extrinsic_r), extrinsic_t_(extrinsic_t) {}

    virtual void set_value(const Eigen::MatrixXd &calib);
    
    void set_intrinsic(int width, int height, double fx, double fy, double cx,
                       double cy, bool perspective = true);
    void set_intrinsic(const Eigen::Matrix3f &mat, bool perspective = true);
    Eigen::Matrix3f intrinsic33() const;


    Eigen::MatrixXd get_value() { return camera_values; }

    Eigen::Matrix3f get_K() { return camera_k; }

    Eigen::Vector4f get_D() { return camera_d; }

    int w() { return width_; }

    int h() { return height_; }

  protected:
    CamBase() = default;

    std::string name_;
    Eigen::MatrixXd camera_values;
    Eigen::Matrix3f camera_k;
    Eigen::Vector4f camera_d;

    Eigen::Matrix3f intrinsic_;
    Eigen::Matrix3f extrinsic_r_;
    Eigen::Vector3f extrinsic_t_;
    int width_;
    int height_;
};
