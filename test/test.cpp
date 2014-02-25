#include "opencv2/opencv.hpp"
#include "facedetect.h"

const std::string testImage = "lena.jpg";

int main(int argc, char **argv)
{
    cv::Mat m = cv::imread(testImage);
    if(m.empty())
    {
        std::cout << "cannot open input image" << std::endl;
        return 0;
    }
    fd::Facedetect detector;
    cv::Mat bitmap = detector.detectBitmap(m, fd::BGR);
    cv::imshow("src", m);
    cv::imshow("res", bitmap);
    cv::waitKey();
    return 0;
}
