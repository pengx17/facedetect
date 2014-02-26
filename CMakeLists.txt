project (facedetect)
cmake_minimum_required (VERSION 2.8.7)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(CASCADE_XML "${PROJECT_ROOT}/data/lbpcascade_frontalface.xml" 
    CACHE STRING "cascade file, can be found in /data")

OPTION(BUILD_SHARED "Build the library as shared library" OFF)

if(BUILD_SHARED)
    add_definitions(-DBUILD_SHARED)
endif()

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
   message( FATAL_ERROR "Cannot find OpenCV library" )
   add_definitions(-DHAVE_OPENCV)
endif(NOT OpenCV_FOUND)

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} ${PROJECT_ROOT}/include)

# reset output directories
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR} )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR} )
# Second, for multi-config builds (e.g. msvc)
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_CAP )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_CAP} ${PROJECT_BINARY_DIR}/${OUTPUTCONFIG} )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_CAP} ${PROJECT_BINARY_DIR}/${OUTPUTCONFIG} )
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_CAP} ${PROJECT_BINARY_DIR}/${OUTPUTCONFIG} )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

# generate cascade header
set(CASCADE_H   "${PROJECT_ROOT}/include/cascade.h")
include("${PROJECT_ROOT}/script/xml2header.cmake")
file(GLOB facedetect_h "${PROJECT_ROOT}/include/*.h")

add_subdirectory(src)
add_subdirectory(test)
