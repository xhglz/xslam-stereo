#include <Eigen/Eigen>

#include "camera/CamBase.h"

void CamBase::set_intrinsic(int width, int height, double fx,
                                        double fy, double cx, double cy,
                                        bool perspective) {
    width_ = width;
    height_ = height;
    Eigen::Matrix3f insic;
    insic.setIdentity();
    insic(0, 0) = fx;
    insic(1, 1) = fy;
    insic(0, 2) = cx;
    insic(1, 2) = cy;
    set_intrinsic(insic, perspective);
}

void CamBase::set_intrinsic(const Eigen::Matrix3f &mat,
                                        bool perspective) {
    intrinsic_.setZero();
    intrinsic_(0, 0) = mat(0, 0);
    intrinsic_(1, 1) = mat(1, 1);
    intrinsic_(0, 2) = mat(0, 2);
    intrinsic_(1, 2) = mat(1, 2);
    intrinsic_(2, 3) = 1;
    intrinsic_(3, 2) = 1;
}

Eigen::Matrix3f CamBase::intrinsic33() const {
    Eigen::Matrix3f mat;
    mat.setIdentity();

    mat(0, 0) = intrinsic_(0, 0);
    mat(1, 1) = intrinsic_(1, 1);

    mat(0, 2) = intrinsic_(0, 2);
    mat(1, 2) = intrinsic_(1, 2);

    return mat;
}