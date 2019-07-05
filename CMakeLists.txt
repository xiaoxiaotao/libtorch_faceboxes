cmake_minimum_required(VERSION 2.8)
project(faceboxes)

set(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -O3")

set(Torch_DIR /home/tao/libtorch/share/cmake/Torch)
set(OpenCV_DIR /home/tao/opencv-3.4.2/build)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(facebox prediction.cpp)
target_link_libraries(facebox ${OpenCV_LIBS} ${TORCH_LIBRARIES})
