facedetect
==========

A static library wrapper for OpenCV. Detect faces in YUV images and output a bitmap for the found area.
Run the `detect` project as a demo.

####Features:

1.  preloaded both old/new cascade file
2.  utility for YUV420/422 image/video input.

Other platform than Windows is not tested.


Hint:
For ease of distribution, you may need to build OpenCV as static:
In OpenCV CMake, uncheck `BUILD_SHARD_LIBS` and `BUILD_WITH_STATIC_CRT`.
