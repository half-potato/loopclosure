cmake_minimum_required(VERSION 2.8.11)
project(rgbd_dataset)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_definitions(
	-Wall
	#-std=c++11
	#-std=c++0x
	-O0
	-g
)

set(ALL_LIBS
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
)

set(CORE_CPP
  "src/util.cpp"
  "src/tum.cpp"
  "src/scenenn.cpp"
  "src/processing.cpp"
)

add_executable(rgbd_dataset src/main.cpp ${CORE_CPP})
target_link_libraries(rgbd_dataset ${ALL_LIBS})
