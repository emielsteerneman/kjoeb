#include "opencv2/opencv.hpp"

#ifndef KJOEB_UTILS_H
#define KJOEB_UTILS_H

struct ExtendedRect : cv::Rect {
    double angle;
    double sizeNorm;
    int id;

    ExtendedRect(const cv::Rect& boundingBox, int id = 0, double sizeNorm = 0.0, double angle = 0.0)
    : cv::Rect(boundingBox), id(id), sizeNorm(sizeNorm), angle(angle){};

    bool operator < (const ExtendedRect& r) const {
        return area() < r.area();
    }
};

struct Line {
    double distance;
    double angle;
    ExtendedRect r1;
    ExtendedRect r2;

    Line(double distance, double angle, ExtendedRect& r1, ExtendedRect& r2)
    : distance(distance), angle(angle), r1(r1), r2(r2){};
};

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
    double now();
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
