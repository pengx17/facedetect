file(GLOB detect_src "${CMAKE_CURRENT_SOURCE_DIR}/detect.cpp")

add_executable(detect ${detect_src})
target_link_libraries(detect facedetect)
