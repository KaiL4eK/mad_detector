cmake_minimum_required(VERSION 3.5)
project(mad_detector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  image_transport
  std_msgs
  geometry_msgs
  message_generation
)

find_package(OpenCV REQUIRED)
find_package(Boost 1.6.5 REQUIRED COMPONENTS program_options filesystem)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_message_files(
  FILES
  Detection.msg
  Detections.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mad_autonet
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
  CATKIN_DEPENDS message_runtime
)

## Application specific

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

set(YOLO_LIB yolo_cpp)
add_library(${YOLO_LIB} SHARED
            src/yolo.cpp
)
target_include_directories(${YOLO_LIB}
PUBLIC
    include
    ${Boost_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
)
target_link_libraries(${YOLO_LIB}
PUBLIC
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)


find_package(InferenceEngine 2.0 REQUIRED)

set(OPENVINO_APP app)
add_executable(${OPENVINO_APP}
    src/main.cpp
    src/yolo_ov.cpp
)
target_link_libraries(${OPENVINO_APP}
    ${InferenceEngine_LIBRARIES}
    IE::ie_cpu_extension
    ${YOLO_LIB}
    ${catkin_LIBRARIES}
)

add_dependencies(${OPENVINO_APP} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
