#include <fstream>
#include "opencv2/opencv.hpp"
#include "facedetect.h"

static void help()
{
    std::cout << "Usage: ./detect img.I420" << std::endl;
    std::cout << "  support image formats defined in cv::imread and I420" << std::endl;
}

//I420 image is customized as:
//width height I420 image_data_in_binary
static cv::Mat readI420Image(const std::string& filename)
{
    std::fstream fin(filename, std::fstream::in);
    cv::Mat res;
    if (!fin.is_open())
    {
        return res;
    }
    char strbuf [10];
    int rows, cols;
    char space;
    fin >> strbuf >> cols >> rows;
    space = fin.get();
    if (strcmp(strbuf, "I420") == 0 && space != ' ')
    {
        std::cout << "Error: image file header is incorrect" << std::endl;
        return res;
    }
    std::auto_ptr<char> buf(new char [rows * cols * 3 / 2]);
    fin.read(buf.get(), rows * cols * 3 / 2);
    cv::Mat src_i420_h = fd::createMatWithPtr(cols, rows, cols, buf.get(), fd::I420);
    
    src_i420_h.copyTo(res);
    return res;
}

static void writeI420Image(const cv::Mat &src, const std::string& filename)
{
    cv::Mat out_yuv420;
    cv::cvtColor(src, out_yuv420, CV_BGR2YUV_I420);
    std::fstream fout("lena.I420", std::fstream::out);
    fout << "I420 " << out_yuv420.cols << " " << (out_yuv420.rows / 3 * 2)  << " ";
    fout.write((const char *)out_yuv420.data, out_yuv420.rows * out_yuv420.cols);
    fout.close();
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        help();
        return 0;
    }
    fd::ImageFormat format = strstr(argv[1], "I420") != 0 ? fd::I420 : fd::BGR;
    cv::Mat src, src_bgr;
    if (format == fd::I420)
    {
        cv::Mat src_i420 = readI420Image(argv[1]);
        cv::cvtColor(src_i420, src_bgr, CV_YUV2BGR_I420);
        src = src_i420;
    }
    else
    {
        src = cv::imread(argv[1]);
        src_bgr = src;
    }
    if(src.empty())
    {
        std::cout << "Error: cannot open input image!" << std::endl;
        return 0;
    }
    fd::Facedetect detector;
    cv::Mat bitmap = detector.detectBitmap(src, format);
    cv::imshow("src", src_bgr);
    cv::imshow("bitmap", bitmap);
    cv::Mat clip;
    cv::cvtColor(bitmap, bitmap, CV_GRAY2BGR);
    cv::bitwise_and(bitmap, src_bgr, clip);
    cv::imshow("clip", clip);
    cv::waitKey();
    return 0;
}
