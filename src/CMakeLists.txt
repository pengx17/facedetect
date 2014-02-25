file(GLOB src "${CMAKE_CURRENT_SOURCE_DIR}/facedetect.cpp")
              
ADD_LIBRARY(facedetect STATIC ${facedetect_h} ${src})
target_link_libraries(facedetect ${OpenCV_LIBRARIES})
