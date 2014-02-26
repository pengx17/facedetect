#include <fstream>
#include "opencv2/opencv.hpp"
#include "facedetect.h"

const std::string testImage = "lena.jpg";
const std::string testImageI420 = "lena.I420";

static cv::Mat readI420Image(const std::string& filename)
{
    std::fstream fin(filename, std::fstream::in);
    cv::Mat res;
    if (!fin.is_open())
    {
        std::cout << "cannot open input image file" << std::endl;
        return res;
    }
    char strbuf [10];
    int rows, cols;
    char space;
    fin >> strbuf >> rows >> cols;
    space = fin.get();
    if (strcmp(strbuf, "I420") == 0 && space != ' ')
    {
        std::cout << "image file header is incorrect" << std::endl;
        return res;
    }
    std::auto_ptr<char> buf(new char [rows * cols * 3 / 2]);
    fin.read(buf.get(), rows * cols * 3 / 2);
    cv::Mat src_i420_h = fd::createMatWithPtr(cols, rows, cols, buf.get(), fd::I420);
    
    src_i420_h.copyTo(res);
    return res;
}

int main(int argc, char **argv)
{
    cv::Mat src_i420 = readI420Image(testImageI420);
    if(src_i420.empty())
    {
        std::cout << "cannot open input image" << std::endl;
        return 0;
    }
    fd::Facedetect detector;
    cv::Mat bitmap = detector.detectBitmap(src_i420, fd::I420);
    cv::Mat src_bgr;
    cv::cvtColor(src_i420, src_bgr, CV_YUV2BGR_I420);
    cv::imshow("src", src_bgr);
    cv::imshow("res", bitmap);
    cv::waitKey();
    /*
    //write the raw yuv image
    cv::Mat src = cv::imread("lena.jpg");
    cv::Mat out_yuv420;
    cv::cvtColor(src, out_yuv420, CV_BGR2YUV_I420);
    std::fstream fout("lena.I420", std::fstream::out);
    fout << "I420 " << (out_yuv420.rows / 3 * 2) << " " << out_yuv420.cols << " ";
    fout.write((const char *)out_yuv420.data, out_yuv420.rows * out_yuv420.cols);
    fout.close();
    */
    return 0;
}
