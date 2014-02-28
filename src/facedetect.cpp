#include <numeric>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "facedetect.h"
#include "cascade.h"

using namespace cv;

namespace fd
{
namespace impl
{
string cascade_string = "";

class MyCascadeClassifier: public CascadeClassifier
{
public:
    //overload load
    bool load(const string& cascade);
};

struct CvFileStorageFix
{
    int i_holder[4];
    void* p_holder[4];
    CvSeq* roots;
    //we do not need the rest ...
};
void *loadOldCascade( cv::FileStorage& fs )
{
    void* ptr = 0;
    CvFileNode* node = 0;

    if( !fs.isOpened() )
        return 0;

    int i, k;
    CvFileStorageFix *cvfs = (CvFileStorageFix *)fs.fs.obj;
    for( k = 0; k < cvfs->roots->total; k++ )
    {
        CvSeq* seq;
        CvSeqReader reader;

        node = (CvFileNode*)cvGetSeqElem( cvfs->roots, k );
        if( !CV_NODE_IS_MAP( node->tag ))
            return 0;
        seq = node->data.seq;
        node = 0;

        cvStartReadSeq( seq, &reader, 0 );

        // find the first element in the map
        for( i = 0; i < seq->total; i++ )
        {
            if( CV_IS_SET_ELEM( reader.ptr ))
            {
                node = (CvFileNode*)reader.ptr;
                goto stop_search;
            }
            CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
        }
    }
stop_search:

    if( !node )
        CV_Error( CV_StsObjectNotFound, "Could not find the/an object in file storage" );

    ptr = cvRead( *fs, node, 0 );

    if( cvGetErrStatus() < 0 )
    {
        cvRelease( (void**)&ptr );
    }
    return ptr;
}

bool MyCascadeClassifier::load(const string& cascade)
{
    oldCascade.release();
    data = Data();
    featureEvaluator.release();

    FileStorage fs(cascade, FileStorage::READ | FileStorage::MEMORY);
    if( !fs.isOpened() )
        return false;

    if( read(fs.getFirstTopLevelNode()) )
        return true;

    oldCascade = Ptr<CvHaarClassifierCascade>((CvHaarClassifierCascade*)loadOldCascade(fs));
    return !oldCascade.empty();
}

MyCascadeClassifier classifier;

void initCascade();
// construct a Mat header for the input data pointer. its data validity is not checked.
Mat createMatWithPtr(int width, int height, int strip, const void *ptr, ImageFormat format);
void detectBitmapHelper(const Mat& _src, std::vector<Rect>& faces, ImageFormat format,
                        double scaleFactor, int minNeighbors, Size minSize, Size maxSize);
void initCascade()
{
    if(cascade_string.size() == 0)
    {
        const int num_sub = sizeof(cascade_strings) / sizeof(*cascade_strings);
        cascade_string = std::accumulate(cascade_strings, cascade_strings + num_sub, cascade_string);
        if(!classifier.load(cascade_string))
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

void detectBitmapHelper(const Mat& _src, std::vector<Rect>& faces, ImageFormat format, 
                        double scaleFactor, int minNeighbors, Size minSize, Size maxSize)
{
    Mat src;
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

    classifier.detectMultiScale(src, faces,
        scaleFactor, minNeighbors, CV_HAAR_SCALE_IMAGE, minSize, maxSize);
}
} // namespace impl

void detectBitmap(const void *_src, const FDSize &imgSize, void *_bitmap,
                  bool output1bit, ImageFormat format, double scaleFactor, int minNeighbors,
                  FDSize _minSize, FDSize _maxSize)
{
    impl::initCascade();

    Mat src    = impl::createMatWithPtr(imgSize.w, imgSize.h, imgSize.w, _src, format);
    std::vector<Rect> faces;

    impl::detectBitmapHelper(src, faces, format, 
        scaleFactor, minNeighbors, Size(_minSize.w, _minSize.h),  Size(_maxSize.w, _maxSize.h));

    const int bitmapWidth = output1bit ? (imgSize.w + 7) / 8 : imgSize.w;
    Mat bitmap = impl::createMatWithPtr(bitmapWidth, imgSize.h, imgSize.w, _bitmap, GRAY);
    bitmap.setTo(0);

    if (!output1bit)
    {
        //iteratively set the areas to 255's
        for(std::vector<Rect>::iterator it = faces.begin(); it < faces.end(); ++it)
        {
            bitmap(*it).setTo(255);
        }
    }
    else
    {
        //TODO
        //optimize me if needed
        const int bits[] = {1, 2, 4, 8, 16, 32, 64, 128};
        for(int y = 0; y < imgSize.h; ++y)
        {
            for(int x = 0; x < imgSize.w; ++x)
            {
                //low efficiency but should work...
                for(std::vector<Rect>::iterator it = faces.begin(); it < faces.end(); ++it)
                {
                    if (it->contains(Point(y, x)))
                    {
                        bitmap.at<unsigned char>(y, x / 8) += bits[(7 - (x % 8))];
                        break;
                    }
                }
            }
        }
    }
}
} // namespace fd
