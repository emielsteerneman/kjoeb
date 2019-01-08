#include "opencv2/opencv.hpp"

#ifndef KJOEB_UTILS_H
#define KJOEB_UTILS_H

struct ExtendedRect : cv::Rect {
    double angle;
    double sizeNorm;
    ExtendedRect(const cv::Rect& boundingBox, double sizeNorm = 0.0, double angle = 0.0)
            : cv::Rect(boundingBox), sizeNorm(sizeNorm), angle(angle){};
};

using Line = std::tuple<int, const cv::Rect&, const cv::Rect&, double>;

template<class T> using Cluster = std::vector<T>;
template<class T> using SuperCluster = std::vector<std::vector<T>>;




class Timer {
    std::chrono::steady_clock::time_point t_begin;
    std::chrono::steady_clock::time_point t_end;
public:
    Timer();
    void start();
    void stop();
    double get();
};

class VidWriter {
    int width, height, nWidth, nHeight;
    int subWidth, subHeight, at;
    int fps;
    cv::VideoWriter writer;
    cv::Mat frame, workFrame;
    bool disabled = false;
public:
    VidWriter(std::string filename, int w, int h, int nW, int nH, int fps);
    void disable();
    void reset();
    void add(const cv::Mat& subFrame, std::string text = "");
    void show();
    void flush();
};


#endif //KJOEB_UTILS_H
