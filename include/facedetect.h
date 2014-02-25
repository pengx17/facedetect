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
    YUV420,     //YUV420 image
    YUV422,
};

// construct a Mat header for the input data pointer. its data validity is not checked.
cv::Mat createMatWithPtr(int width, int height, int strip, void *ptr, ImageFormat format);

class Facedetect
{
public:
    Facedetect();
    ~Facedetect();
    // detect faces in the input
    // return a bitmap of detected face rectangles
    cv::Mat detectBitmap(const cv::Mat& src, ImageFormat format = YUV420,
        double scaleFactor = 1.1, int minNeighbors = 3,
        cv::Size minSize = cv::Size(), cv::Size maxSize = cv::Size());
private:
    struct Impl;
    std::auto_ptr<Impl> impl;
};
}
#endif /* __FACEDETECT_H__ */
