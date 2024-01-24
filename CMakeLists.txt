cmake_minimum_required(VERSION 3.10.2)
project(xslam_stereo CXX)

IF (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized -fno-omit-frame-pointer")

option(XSLAM_DEBUG "option for debug" ON)
if (XSLAM_DEBUG) 
  add_definitions(-DXSLAM_DEBUG)
endif(XSLAM_DEBUG)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(yaml-cpp REQUIRED)
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

list(APPEND thirdparty_libraries
  pthread
  ${OpenCV_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

add_subdirectory(src/xprimer)

add_executable(run_euroc test/run_euroc.cc)
target_link_libraries(run_euroc PUBLIC xprimer ${thirdparty_libraries})





# install(TARGETS robot_lib run_robot run_ros1 simple_goal run_udp_client 
#     ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#     LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#     RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
# install(DIRECTORY src/
#     DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
#     FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
# )
# install(DIRECTORY launch/
#     DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
# )