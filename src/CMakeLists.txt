list(APPEND LIBRARY_SOURCES
  common/PrintDebug.cpp
  odometry/Frame.cpp
  odometry/Camera.cpp
  odometry/Feature.cpp
  odometry/Frontend.cpp
  odometry/Map.cpp
  odometry/Mappoint.cpp
  odometry/Odometry.cpp
  odometry/Viewer.cpp
  odometry/Backend.cpp
  image/ImageProcess.cpp
  onnx/SPGLEnd2EndORT.cpp
  onnx/SuperPointOrt.cpp
  onnx/LightGlueOrt.cpp
)

list(APPEND LIBRARY_HEADERS
  common/Timer.h
  common/DataBase.h
  common/DataKitti.h
  common/CVYamlParse.h
  common/YamlConfig.h
  common/PrintDebug.h
  common/Visualizer.h
  odometry/Backend.h
  odometry/MathTypes.h
  odometry/Frame.h
  odometry/Camera.h
  odometry/FeatsBase.h
  odometry/Algorithm.h
  image/ImageProcess.h
  image/KeyPointsType.h
  onnx/BaseOnnxRunner.h
  onnx/SPGLEnd2EndORT.h
  onnx/SuperPointOrt.h
  onnx/LightGlueOrt.h
  odometry/FeatsSuperPoint.h
)

add_library(odometry SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_link_libraries(odometry ${thirdparty_libraries})
