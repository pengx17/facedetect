#include <numeric>
#include "opencv2/opencv.hpp"
#include "facedetect.h"
#include "cascade.h"

using namespace cv;

namespace fd
{
namespace impl
{
string cascade_string = "";
CascadeClassifier classifier;

void initCascade();
// construct a Mat header for the input data pointer. its data validity is not checked.
Mat createMatWithPtr(int width, int height, int strip, const void *ptr, ImageFormat format);
void detectBitmapHelper(const Mat& _src, Mat &bitmap, ImageFormat format,
                        double scaleFactor, int minNeighbors, Size minSize, Size maxSize);
void initCascade()
{
    if(cascade_string.size() == 0)
    {
        const int num_sub = sizeof(cascade_strings) / sizeof(*cascade_strings);
        cascade_string = std::accumulate(cascade_strings, cascade_strings + num_sub, cascade_string);

        FileStorage fs(cascade_string, FileStorage::READ | FileStorage::MEMORY | FileStorage::FORMAT_XML);
        if(!classifier.read(fs.getFirstTopLevelNode()))
        {
            //this should never happen in real product, so dont worry about throw 
            //an exception in constructor
            std::cout << "Cannot load cascade file" << std::endl;
            throw std::exception();
        }
    }
}

Mat createMatWithPtr(int width, int height, int strip, const void *ptr, ImageFormat format)
{
    int cvformat = 0;
    switch(format)
    {
    case IYUV:
        height = height / 2 * 3;
        cvformat = CV_8UC1;
        break;
    case UYVY:
        width *= 2;
        cvformat = CV_8UC1; // OpenCV seems to treat UYVY images as 8bit instead of 16bit
        break;
    case GRAY:
        cvformat = CV_8UC1;
        break;
    case RGB:
    case BGR:
        strip *= 3;
        cvformat = CV_8UC3;
        break;
    }
    // we know that we will not change the content of input .. so const_cast should work
    return Mat(height, width, cvformat, const_cast<void *>(ptr), strip);
}

void detectBitmapHelper(const Mat& _src, Mat &bitmap, ImageFormat format, 
                        double scaleFactor, int minNeighbors, Size minSize, Size maxSize)
{
    Mat src;
    bitmap.setTo(0);
    switch(format)
    {
    case IYUV:
        cvtColor(_src, src, CV_YUV420p2BGR);
        cvtColor(src, src, COLOR_BGR2GRAY);
        break;
    case UYVY:
        cvtColor(_src, src, CV_YUV2BGR_UYVY);
        cvtColor(src, src, COLOR_BGR2GRAY);
        break;
    case GRAY:
        break;
    case RGB:
        cvtColor(_src, src, COLOR_RGB2GRAY);
        break;
    case BGR:
        cvtColor(_src, src, COLOR_BGR2GRAY);
        break;
    }
    equalizeHist(src, src);

    std::vector<Rect> faces;
    classifier.detectMultiScale(src, faces,
        scaleFactor, minNeighbors, CV_HAAR_SCALE_IMAGE, minSize, maxSize);

    //iteratively set the areas to 1's
    for(std::vector<Rect>::iterator it = faces.begin(); it < faces.end(); ++it)
    {
        bitmap(*it).setTo(255);
    }
}
} // namespace impl

void detectBitmap(const void *_src, const FDSize &_imgSize, void *_bitmap,
                  ImageFormat format, double scaleFactor, int minNeighbors,
                  FDSize _minSize, FDSize _maxSize)
{
    impl::initCascade();

    Mat src    = impl::createMatWithPtr(_imgSize.w, _imgSize.h, _imgSize.w, _src, format);
    Mat bitmap = impl::createMatWithPtr(_imgSize.w, _imgSize.h, _imgSize.w, _bitmap, GRAY);

    impl::detectBitmapHelper(src, bitmap, format, 
        scaleFactor, minNeighbors, Size(_minSize.w, _minSize.h),  Size(_maxSize.w, _maxSize.h));
}
} // namespace fd
