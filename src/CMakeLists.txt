file(GLOB src "${CMAKE_CURRENT_SOURCE_DIR}/facedetect.cpp")

if(BUILD_SHARED)
    ADD_LIBRARY(facedetect SHARED ${facedetect_h} ${src})
else()
    ADD_LIBRARY(facedetect STATIC ${facedetect_h} ${src})
endif()
target_link_libraries(facedetect ${OpenCV_LIBRARIES})
