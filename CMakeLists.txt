cmake_minimum_required(VERSION 3.13)
project(kjoeb)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR "../opencv/installation/OpenCV-master/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

add_executable(kjoeb main.cpp maths.cpp clustering.cpp)
target_link_libraries(kjoeb ${OpenCV_LIBS})
