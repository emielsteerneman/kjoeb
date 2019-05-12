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

void DrawRotatedRectangle(cv::Mat& image, const ExtendedRect& rect, int thickness){
    cv::Scalar color = cv::Scalar(255.0, 255.0, 255.0); // white

    // Create the rotated rectangle
    cv::Point centerPoint((int)(rect.x + rect.sizeNorm / 2), (int)(rect.y + rect.sizeNorm / 2));
    cv::RotatedRect rotatedRectangle(centerPoint, {(float)rect.sizeNorm, (float)rect.sizeNorm}, (int)(rect.angle * 57.2958));

    // We take the edges that OpenCV calculated for us
    cv::Point2f vertices2f[4];
    rotatedRectangle.points(vertices2f);

    // Convert them so we can use them in a fillConvexPoly
    cv::Point vertices[4];
    for(int i = 0; i < 4; ++i){
        vertices[i] = vertices2f[i];
    }

    for(int i = 0; i < 4; i++){
        cv::line(image, vertices[i], vertices[(i+1)%4], color, 1);
    }

    // Now we can fill the rotated rectangle with our specified color
//    cv::fillConvexPoly(image,vertices,4,color);
}