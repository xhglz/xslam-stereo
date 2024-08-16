### https://zhuanlan.zhihu.com/p/675418205
# 配置阶段 
## cmake -B build -DCMAKE_BUILD_TYPE=Release
## -D 选项 cmake -B build -DCMAKE_BUILD_TYPE=Release 
## -G 选项 cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
# 构建阶段 
## cmake --build build -j4
## cmake --build build -j8 --target install


cmake_minimum_required(VERSION 3.10.2)
project(slam_stereo CXX)

IF (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized -fno-omit-frame-pointer")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

option(SLAM_DEBUG "option for debug" ON)
if (SLAM_DEBUG) 
  add_definitions(-DSLAM_DEBUG)
endif(SLAM_DEBUG)

# find_package(Eigen3 REQUIRED)
# find_package(OpenCV 4 REQUIRED)
# find_package(yaml-cpp REQUIRED)
# find_package(fmt REQUIRED)
# find_package(Pangolin REQUIRED)

# include_directories(
#   src
#   ${EIGEN3_INCLUDE_DIR}
#   ${OpenCV_INCLUDE_DIRS}
#   ${YAML_CPP_INCLUDE_DIR}
#   ${Pangolin_INCLUDE_DIRS}
# )

# list(APPEND thirdparty_libraries
#   pthread
#   fmt::fmt
#   GL GLU GLEW glut
#   ${OpenCV_LIBRARIES}
#   ${YAML_CPP_LIBRARIES}
#   ${Pangolin_LIBRARIES} 
# )

include(cmake/packages.cmake)
include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src)
add_subdirectory(test)

# install(DIRECTORY src/
#     DESTINATION ${CMAKE_RUNTIME_INCLUDE_DIRECTORY}
#     FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
# )