#ifndef __FACEDETECT_H__
#define __FACEDETECT_H__

#if (defined WIN32 || defined _WIN32 || defined WINCE) && defined BUILD_SHARED
#  define FD_EXPORTS __declspec(dllexport)
#else
#  define FD_EXPORTS
#endif

namespace fd
{
enum ImageFormat
{
    GRAY,        //Signle channel gray image
    LUMA = GRAY,
    BGR,         //Default 3-channel color order in OpenCV
    RGB,
    IYUV,        //YUV420 image
    I420 = IYUV,
    UYVY,
    YUV422 = UYVY,
};

struct FD_EXPORTS FDSize
{
    FDSize(int _w = 0, int _h = 0): w(_w), h(_h) {}
    int w;
    int h;
};

// Detect faces in the input
// return a bitmap of detected face rectangles
//
// note, the bitmap is not refering to Window bitmap image format
// it is a 8bit single channel image with the same image size with the input,
// the non-negative area indicates the found area.
// also, it MUST be pre-allocated with proper size
FD_EXPORTS void detectBitmap(const void *src, const FDSize &imgSize, void *bitmap,
                             ImageFormat format = I420, double scaleFactor = 1.1, int minNeighbors = 3,
                             FDSize minSize = FDSize(), FDSize maxSize = FDSize());
}
#endif /* __FACEDETECT_H__ */
