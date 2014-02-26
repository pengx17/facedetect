#ifndef __FACEDETECT_H__
#define __FACEDETECT_H__

#include "opencv2/core/core.hpp"

namespace fd
{
enum ImageFormat
{
    GRAY,       //Signle channel gray image
    LUMA = GRAY,
    BGR,        //Default 3-channel color order in OpenCV
    RGB,
    IYUV,        //YUV420 image
    I420 = IYUV,
    UYVY,
    YUV422 = UYVY,
};

// construct a Mat header for the input data pointer. its data validity is not checked.
cv::Mat createMatWithPtr(int width, int height, int strip, void *ptr, ImageFormat format);

class Facedetect
{
public:
    Facedetect();
    ~Facedetect();
    // Detect faces in the input
    // return a bitmap of detected face rectangles
    //
    // note, the bitmap is not refering to Window bitmap image format
    // it is a 8bit single channel image with the same image size with the input,
    // the non-negative area indicates the found area.
    cv::Mat detectBitmap(const cv::Mat& src, ImageFormat format = I420,
        double scaleFactor = 1.1, int minNeighbors = 3,
        cv::Size minSize = cv::Size(), cv::Size maxSize = cv::Size());
private:
    struct Impl;
    std::auto_ptr<Impl> impl;
};
}
#endif /* __FACEDETECT_H__ */
