project (facedetect)
cmake_minimum_required (VERSION 2.8.7)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(CASCADE_XML "${PROJECT_ROOT}/data/lbpcascade_frontalface.xml" 
    CACHE STRING "cascade file, can be found in /data")

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
   message( FATAL_ERROR "Cannot find OpenCV library" )
   add_definitions(-DHAVE_OPENCV)
endif(NOT OpenCV_FOUND)

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} ${PROJECT_ROOT}/include)

set(CASCADE_H   "${PROJECT_ROOT}/include/cascade.h")
include("${PROJECT_ROOT}/script/xml2header.cmake")
file(GLOB facedetect_h "${PROJECT_ROOT}/include/*.h")

add_subdirectory(src)
add_subdirectory(test)
