set(XPRIMER_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${XPRIMER_DIR})

set(Headers
  logger/print.h
)
source_group("Headers" FILES ${Headers})

set(Sources
  logger/print.cc
)
source_group("Sources" FILES ${Sources})

set(target xprimer)
add_library(${target} ${Headers} ${Sources})
target_compile_options(${target} PRIVATE -fPIC)
target_link_libraries(${target} PUBLIC ${thirdparty_libraries})
target_include_directories(${target} PUBLIC ${XPRIMER_DIR})

# include(GenerateExportHeader)
# generate_export_header(
#     ${target} EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/xrprimer_export.h
# )

# target_include_directories(
#     ${target}
#     PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cpp/xrprimer>
#            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/>
#            $<INSTALL_INTERFACE:include/xrprimer>
# )

# target_link_libraries(
#     ${target}
#     PUBLIC $<$<NOT:$<PLATFORM_ID:IOS>>:PnpSolver::pnpsolver> OpenCV::OpenCV
#            Eigen3::Eigen Ceres::ceres jsoncpp_static spdlog::spdlog_header_only
# )

# install(TARGETS ${target} DESTINATION lib EXPORT ${target}-targets)
# install(EXPORT ${target}-targets DESTINATION lib/cmake/xrprimer
#         NAMESPACE XRPrimer::
# )
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING
#         PATTERN "*.h"
# )

# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#     ${CMAKE_SOURCE_DIR}/cmake/XRPrimerConfig.cmake.in
#     "${CMAKE_CURRENT_BINARY_DIR}/XRPrimerConfig.cmake"
#     INSTALL_DESTINATION lib/cmake/xrprimer NO_CHECK_REQUIRED_COMPONENTS_MACRO
# )

# file(READ ${CMAKE_SOURCE_DIR}/cmake/dependencies.cmake DEPS_FILE_CONTENTS)
# string(REPLACE "find_package" "find_dependency" DEPS_FILE_CONTENTS
#                ${DEPS_FILE_CONTENTS}
# )
# file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dependencies.cmake ${DEPS_FILE_CONTENTS})

# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/XRPrimerConfigVersion.cmake"
#     VERSION
#         "${XRPRIMER_VERSION_MAJOR}.${XRPRIMER_VERSION_MINOR}.${XRPRIMER_VERSION_PATCH}"
#     COMPATIBILITY AnyNewerVersion
# )

# install(
#     FILES "${CMAKE_CURRENT_BINARY_DIR}/XRPrimerConfig.cmake"
#           "${CMAKE_CURRENT_BINARY_DIR}/XRPrimerConfigVersion.cmake"
#           "${CMAKE_CURRENT_BINARY_DIR}/dependencies.cmake"
#     DESTINATION lib/cmake/xrprimer
# )


# install(FILES ${CMAKE_CURRENT_BINARY_DIR}/xrprimer_export.h
#         DESTINATION include/xrprimer
# )
