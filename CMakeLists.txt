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

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_message_files(
  FILES
  Detection.msg
  Detections.msg
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
  execute_process(
    COMMAND mkdir models
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  execute_process(
    COMMAND wget -q https://github.com/KaiL4eK/mad_detector/releases/download/v0.0/Yolov4Tiny_mad_model.pth -O models/Yolov4Tiny_mad_model.pth
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()
