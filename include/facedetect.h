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
// it is a single channel image with the same image size (or 1/8 + padding if output1bit is true) with the input
// a non-negative area indicates a found area.
// if output1bit is true, 8 pixels in a row are grouped in one byte of the output
// and significantly reduce the width to 1/8. If the width is not a multiple of 8,
// the bitmap will be padded with 0's at the end of each row; otherwise, a pixel possesses one byte 
// also, it MUST be pre-allocated with proper size
FD_EXPORTS void detectBitmap(const void *src, const FDSize &imgSize, void *bitmap, bool output1bit = true,
                             ImageFormat format = I420, double scaleFactor = 1.1, int minNeighbors = 3,
                             FDSize minSize = FDSize(), FDSize maxSize = FDSize());
}
#endif /* __FACEDETECT_H__ */
