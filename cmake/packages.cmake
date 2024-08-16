# 引入该目录下的.cmake文件
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# eigen 3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# sophus
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/sophus)

# suitesparse
include_directories("/usr/include/suitesparse")
link_directories(/usr/lib/x86_64-linux-gnu) 

# opencv
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# yaml-cpp
find_package(yaml-cpp REQUIRED)
include_directories(${yaml-cpp_INCLUDE_DIRS})

# csparse
# find_package(CSparse REQUIRED)
# include_directories(${CSPARSE_INCLUDE_DIR})

# cholmod
# find_package(Cholmod REQUIRED)
# include_directories(${CHOLMOD_INCLUDE_DIRS})

# Pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# pcl
# find_package(PCL REQUIRED)
# include_directories(${PCL_INCLUDE_DIRS})
# message(STATUS "PCL library status:")
# message(STATUS "    config: ${PCL_DIR}")
# message(STATUS "    version: ${PCL_VERSION}")
# message(STATUS "    libraries: ${PCL_LIBRARY_DIRS}")
# message(STATUS "    include path: ${PCL_INCLUDE_DIRS}")

# fmt
find_package(fmt REQUIRED)

# g2o 使用thirdparty中的
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/g2o/include)

find_package(CUDA REQUIRED)
set(ONNXRUNTIME_ROOTDIR /home/ly/opt/onnxruntime/onnxruntime-linux-x64-gpu-cuda12-1.17.3)
message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")
include_directories(${ONNXRUNTIME_ROOTDIR}/include)
link_directories(${ONNXRUNTIME_ROOTDIR}/lib)

set(g2o_libs
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_stuff.so
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_core.so
    # ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_solver_cholmod.so
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_solver_dense.so
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_solver_csparse.so
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_csparse_extension.so
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_types_sba.so
    # ${CSPARSE_LIBRARY}
    # ${CHOLMOD_LIBRARY}
)

set(thirdparty_libraries
    ${g2o_libs}
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
    ${Pangolin_LIBRARIES}
    # ${PCL_LIBRARIES}
    pthread
    fmt::fmt
    GL GLU GLEW glut
    onnxruntime
)
