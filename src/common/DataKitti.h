#pragma once

#include <unistd.h>
#include <fstream>
#include <sstream>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>

#include "common/DataBase.h"
#include "common/PrintDebug.h"

namespace myslam {

/**
 * 数据集读取
 * 构造时传入配置文件路径，配置文件的dataset_dir为数据集路径
 * Init之后可获得相机和下一帧图像
 */
class DataKitti : public DataBase {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<DataKitti> Ptr;
    DataKitti(const std::string& dataset_path) : DataBase(dataset_path) {}

    ~DataKitti() {}

    /// 初始化，返回是否成功
    bool Init() override {
        // read camera intrinsics and extrinsics
        std::ifstream fin(dataset_path_ + "/calib.txt");
        if (!fin) {
            PRINT_ERROR(RED "cannot find %s\n" RESET, dataset_path_.c_str());
            return false;
        }   

        for (int i = 0; i < 4; ++i) {
            if (fin.eof()) {
                break;
            }
            
            char camera_name[3];
            for (int k = 0; k < 3; ++k) {
                fin >> camera_name[k];
            }
            double projection_data[12];
            for (int k = 0; k < 12; ++k) {
                fin >> projection_data[k];
            }

            Mat33 K;
            K << projection_data[0], projection_data[1], projection_data[2],
                projection_data[4], projection_data[5], projection_data[6],
                projection_data[8], projection_data[9], projection_data[10];
            
            Vec3 t;
            t << projection_data[3], projection_data[7], projection_data[11];
            t = K.inverse() * t;
            K = K * 0.5;
            
            Camera::Ptr new_camera(new Camera(
                K(0, 0), K(1, 1), K(0, 2), K(1, 2),t.norm(), SE3(SO3(), t)));
            cameras_.push_back(new_camera);

            std::stringstream ss;
            ss << "Cam " << i << " extrinsics: " << t.transpose() << std::endl;
            PRINT_DEBUG(ss.str().c_str());
        }
        fin.close();
        current_image_index_ = 0;
        return true;
    }

    Frame::Ptr NextFrame() override {
        boost::format fmt("%s/image_%d/%06d.png");
        cv::Mat image_left, image_right;
        
        image_left  = cv::imread((fmt % dataset_path_ % 0 % current_image_index_).str(), cv::IMREAD_GRAYSCALE);
        image_right = cv::imread((fmt % dataset_path_ % 1 % current_image_index_).str(), cv::IMREAD_GRAYSCALE);

        if (image_left.data == nullptr || image_right.data == nullptr) {
            PRINT_WARNING(YELLOW "cannot find images at index %d\n" RESET, current_image_index_);
            return nullptr;
        }

        cv::Mat image_left_resized, image_right_resized;
        cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,cv::INTER_NEAREST);

        auto new_frame = Frame::CreateFrame();
        new_frame->left_img_ = image_left_resized;
        new_frame->right_img_ = image_right_resized;
        current_image_index_++;

        // 100ms 
        // usleep(50000 * nDelayTimes);
        return new_frame;
    }

private:
    int nDelayTimes = 2;
};

}

