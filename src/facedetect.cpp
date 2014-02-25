#include "opencv2/opencv.hpp"
#include "facedetect.h"

using namespace cv;

namespace fd
{
Mat createMatWithPtr(int width, int height, int strip, void *ptr, ImageFormat format)
{
    int cvformat = 0;
    switch(format)
    {
    case YUV420:
        height = height / 2 * 3;
        cvformat = CV_8UC1;
        break;
    case YUV422:
        width *= 2;
        cvformat = CV_8UC1;
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
    return Mat(height, width, cvformat, ptr, strip);
}

struct Facedetect::Impl
{
    Impl();
    ~Impl();

    CascadeClassifier classifier;
};

Facedetect::Impl::Impl()
{
    //init classifier
    //hard coded for testing
    String cascade_name = "haarcascade_frontalface_alt.xml";
    if(!classifier.load(cascade_name))
    {
        //this should never happen in real product, so dont worry about throw 
        //an exception in constructor
        std::cout << "Cannot find cascade file" << std::endl;
        throw std::exception();
    }
}

Facedetect::Impl::~Impl()
{
}

Facedetect::Facedetect()
    :impl(new Impl())
{
}

Facedetect::~Facedetect()
{
}

Mat Facedetect::detectBitmap(const Mat& _src, ImageFormat format,
    double scaleFactor, int minNeighbors, Size minSize, Size maxSize)
{
    Mat src;
    switch(format)
    {
    case YUV420:
        cvtColor(_src, src, CV_YUV420sp2BGR);
        cvtColor(src, src, COLOR_BGR2GRAY);
        break;
    case YUV422:
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
    impl->classifier.detectMultiScale(src, faces, 
        scaleFactor, minNeighbors, CV_HAAR_SCALE_IMAGE, minSize, maxSize);

    Mat res(src.rows, src.cols, CV_8UC1, Scalar::all(0));
    //iteratively set the areas to 1's
    for(std::vector<Rect>::iterator it = faces.begin(); it < faces.end(); ++it)
    {
        res(*it).setTo(255);
    }
    return res;
}
}
