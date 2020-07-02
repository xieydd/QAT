cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

PROJECT(face_detection)

add_definitions(-DJNI_DEBUG)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(NCNN_INSTALL_SDK OFF CACHE BOOL "Install ncnn library and headers")

add_subdirectory(ncnn)
set(OpenCV_DIR /opt/opencv-3.4.2_install/share/OpenCV)

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs video)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ncnn/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/ncnn/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(Extract ncnn_forward.cpp)
target_link_libraries(Extract ncnn ${OpenCV_LIBS})