//
// Created by emiel on 9-1-19.
//

#include "drawing.h"
#include "utils.h"
#include <opencv2/opencv.hpp>


void drawCluster(cv::Mat& frame, const Cluster<ExtendedRect>& cluster, cv::Scalar colour, int thickness){
    for(const ExtendedRect& rect : cluster)
        cv::rectangle(frame, rect, colour, thickness);
}

void drawCluster(cv::Mat& frame, const Cluster<Line>& cluster, cv::Scalar colour, int thickness, bool drawLines, bool drawRects){
    for(const Line& line : cluster){
        cv::Point p1(line.r1.x + line.r1.width / 2, line.r1.y + line.r1.height / 2);
        cv::Point p2(line.r2.x + line.r2.width / 2, line.r2.y + line.r2.height / 2);
        if(drawLines)
            cv::line(frame, p1, p2, colour, thickness);

        if(drawRects) {
            cv::rectangle(frame, line.r1, colour, thickness);
            cv::rectangle(frame, line.r2, colour, thickness);
        }
    }
}

void drawSuperCluster(cv::Mat& frame, const SuperCluster<ExtendedRect>& superCluster){
    for(int i = 0; i < superCluster.size(); i++){
        double iColour = 50 + (200 * i / superCluster.size());
        drawCluster(frame, superCluster[i], {iColour, 0, iColour});
    }
}

void drawSuperCluster(cv::Mat& frame, const SuperCluster<Line>& superCluster){
    for(int i = 0; i < superCluster.size(); i++){
        double iColour = 50 + (200 * i / superCluster.size());
        drawCluster(frame, superCluster[i], {iColour, 0, iColour});
    }
}