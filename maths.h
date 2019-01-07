//
// Created by emiel on 7-1-19.
//

#ifndef KJOEB_MATHS_H
#define KJOEB_MATHS_H

#include <tuple>
#include <opencv2/opencv.hpp>

using Line = std::tuple<int, const cv::Rect&, const cv::Rect&, double>;

namespace maths {

    bool inRangeRel(double a, double b, double c = 0.85);
    bool inRangeAbs(double a, double b, double c);

    double average(const std::vector<int>& vec);
    double variance(const std::vector<int>& vec);

    unsigned int fastroot(unsigned int x);
    double normalizeAngle90(double angle);

    int distance(const cv::Rect &r1, const cv::Rect &r2);
    double distanceRoot(const cv::Rect& r1, const cv::Rect& r2);

    bool intersect(const cv::Rect& r1, const cv::Rect& r2);
    bool intersect(const cv::Point& a11, const cv::Point& a12, const cv::Point& b11, const cv::Point& b12);
    bool intersect(const cv::Rect& r11, const cv::Rect& r12, const cv::Rect& r21, const cv::Rect& r22);
    bool intersect(const Line& l1, const Line& l2);

}


#endif //KJOEB_MATHS_H
