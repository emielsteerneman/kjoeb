//
// Created by emiel on 9-1-19.
//

#ifndef KJOEB_DRAWING_H
#define KJOEB_DRAWING_H

#include "utils.h"
#include <opencv2/opencv.hpp>

void drawCluster(cv::Mat& frame, const Cluster<ExtendedRect>& cluster, cv::Scalar colour = {255, 255, 255}, int thickness = 2);
void drawCluster(cv::Mat& frame, const Cluster<Line>& cluster        , cv::Scalar colour = {255, 255, 255}, int thickness = 2, bool drawLines = true, bool drawRects = true);

void drawSuperCluster(cv::Mat& frame, const SuperCluster<ExtendedRect>& superCluster);
void drawSuperCluster(cv::Mat& frame, const SuperCluster<Line>& superCluster);

#endif //KJOEB_DRAWING_H
