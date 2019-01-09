//
// Created by emiel on 7-1-19.
//

#include "maths.h"

#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>

namespace maths {

    // ====================================================================================
    // ==================================== PURE MATHS ====================================
    // ====================================================================================

    // TODO What if either a of b equals 0
    bool inRangeRel(double a, double b, double c){
        return c < a/b && c < b/a;
    }

    bool inRangeAbs(double a, double b, double c){
        return fabs(a-b) <= c;
    }

    double average(const std::vector<int>& vec){
        int sum = 0;
        for(const auto& item : vec)
            sum += item;
        return sum / (double)vec.size();
    }

    double variance(const std::vector<int>& vec){
        double avg = average(vec);
        double sum = 0.0;
        for(const auto& item : vec)
            sum += std::pow(item - avg, 2);
        return sum / (double)(vec.size() - 2);
    }

    //http://supp.iar.com/FilesPublic/SUPPORT/000419/AN-G-002.pdf
    unsigned int fastroot(unsigned int x){
        unsigned int a,b;
        b     = x;
        a = x = 0x3f;
        x     = b/x;
        a = x = (x+a)>>1;
        x     = b/x;
        a = x = (x+a)>>1;
        x     = b/x;
        x     = (x+a)>>1;
        return(x);
    }

    double normalizeAngle90(double angle){
//        angle *= 57.29;
        double step = M_PI / 4.0;

//        angle = fabs(angle);
        while(angle < 0.0)
            angle += M_PI;

        while(step <= angle)
            angle -= step;

        if(step / 2 < angle)
            angle -= step;

        return angle;
    }



    // ====================================================================================
    // ==================================== CUBE MATHS ====================================
    // ====================================================================================

    // Distance calculation without root
    int distance(const cv::Rect &r1, const cv::Rect &r2){
        int dx = r1.x - r2.x;
        int dy = r1.y - r2.y;
        return dx * dx + dy * dy;
    }

    double distanceRoot(const cv::Rect& r1, const cv::Rect& r2){
        return sqrt(distance(r1, r2));
    }

    bool intersect(const cv::Rect& r1, const cv::Rect& r2){
        return r1.x < (r2.x + r2.width)
               && r2.x < (r1.x + r1.width)
               && r1.y < (r2.y + r2.height)
               && r2.y < (r1.y + r1.height);
    }

    bool intersect(const cv::Point& a11, const cv::Point& a12, const cv::Point& b11, const cv::Point& b12){
        return intersect({a11, a12}, {b11, b12});
    }

    bool intersect(const cv::Rect& r11, const cv::Rect& r12, const cv::Rect& r21, const cv::Rect& r22){
        return intersect(r11.tl(), r12.tl(), r21.tl(), r22.tl());
    }

    bool intersect(const Line& l1, const Line& l2){
        return intersect(l1.r1, l1.r2, l2.r1, l2.r2);
    }

}