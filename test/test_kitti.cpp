#include <opencv2/opencv.hpp>

#include "common/DataKitti.h"
#include "common/PrintDebug.h"
#include "odometry/Odometry.h"

using namespace myslam;

int main(int argc, char **argv) {
    std::string verbosity = "DEBUG";
    Printer::setPrintLevel(verbosity);

    std::string config_file = "config/kitti.yaml";
    if ( argc == 2 ) {
        config_file = argv[1];
    }
        
    auto parser = std::make_shared<YamlParser>(config_file);
    OdometryOptions params;
    params.print_and_load(parser);

    std::string dataset_path = params.dataset_path;
    auto data = std::make_shared<DataKitti>(dataset_path);
    data->Init();
    
    auto odom = std::make_shared<Odometry>(params, data);
    odom->Run();
    
    return EXIT_SUCCESS;
}