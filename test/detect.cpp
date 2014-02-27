#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <fstream>
#include "opencv2/opencv.hpp"
#include "facedetect.h"


// YUV420 videos can be downloaded at http://trace.eas.asu.edu/yuv/.
// please rename video name to *.widthxheight.I420
static void help()
{
    std::cout << "Usage: ./detect img.I420" << std::endl;
    std::cout << "  support image formats defined in cv::imread and I420" << std::endl;
    std::cout << "  support I420 YUV videos" << std::endl;
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
    cv::Mat src_i420_h = cv::Mat(rows * 3 / 2, cols, CV_8UC1, buf.get());
    
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

static bool constructMatFromString(const char *fpath, cv::Mat &img)
{
    int height = 0, width = 0;
    char fname_buf[255];
    strcpy(fname_buf, fpath);
    char *strptr = strtok(fname_buf, ".x");
    strptr = strtok(NULL, ".x");
    width  = atoi(strptr);
    strptr = strtok(NULL, ".x");
    height = atoi(strptr);
    if (height == 0 || width == 0)
    {
        return false;
    }
    img.create(height * 3 / 2, width, CV_8UC1);
    return true;
}

static void detectAndShow(const cv::Mat &src, fd::ImageFormat format, int wait = 0)
{
    cv::Mat src_bgr;
    if (format == fd::I420)
    {
        cv::cvtColor(src, src_bgr, CV_YUV2BGR_I420);
    }
    else
    {
        src_bgr = src;
    }
    int width = src_bgr.cols, height = src_bgr.rows;
    cv::Mat bitmap(src_bgr.rows, src_bgr.cols, CV_8UC1);
    fd::detectBitmap(src.ptr<void>(), fd::FDSize(width, height), bitmap.ptr<void>(), false, format);
    cv::imshow("src", src_bgr);
    cv::imshow("bitmap", bitmap);
    cv::Mat clip;
    cv::cvtColor(bitmap, bitmap, CV_GRAY2BGR);
    cv::bitwise_and(bitmap, src_bgr, clip);
    cv::imshow("clip", clip);
    cv::waitKey(wait);
}

// read a frame from input fstream
// frame width and height is needed to be given in the img
static bool readI420Video(std::fstream& fs, cv::Mat &img)
{
    if(!fs.good())
    {
        return false;
    }
    fs.read((char *)img.data, img.rows * img.cols);
    return true;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        help();
        return 0;
    }
    fd::ImageFormat format = strstr(argv[1], "I420") != 0 ? fd::I420 : fd::BGR;
    bool isI420Video = strstr(argv[1], "I420.yuv") != 0;
    cv::Mat src, src_bgr;
    
    if (isI420Video)
    {
        std::fstream fs(argv[1], std::fstream::in | std::fstream::binary);
        if (!fs.good())
        {
            std::cout << "Error: cannot open input video!" << std::endl;
        }
        if (!constructMatFromString(argv[1], src))
        {
            std::cout << "Error: cannot get valid frame size from file name!" << std::endl;
        }
        while(readI420Video(fs, src))
        {
            detectAndShow(src, format, 30);
        }
    }
    else
    {
        if (format == fd::I420)
        {
            src = readI420Image(argv[1]);
        }
        else
        {
            src = cv::imread(argv[1]);
        }
        if(src.empty())
        {
            std::cout << "Error: cannot open input image!" << std::endl;
            return 0;
        }
        detectAndShow(src, format);
    }
    return 0;
}
