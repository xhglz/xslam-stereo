add_executable(test_kitti test_kitti.cpp)
target_link_libraries(test_kitti odometry)

add_executable(test_lightglue test_lightglue.cpp)
target_link_libraries(test_lightglue odometry)

install(TARGETS odometry test_kitti
  ARCHIVE DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
  LIBRARY DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  RUNTIME DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)