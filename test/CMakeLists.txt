file(GLOB test_src "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp")

add_executable(facedetect_test ${test_src})
target_link_libraries(facedetect_test facedetect)