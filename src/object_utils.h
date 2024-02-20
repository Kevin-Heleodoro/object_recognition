// Author: Kevin Heleodoro
// Date: February 18, 2024
// Purpose: A collection of utils used for object recognition

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#ifndef OBJECT_UTILS_H
#define OBJECT_UTILS_H

// void thresholdDemo(int, void *);

// void morphologicalFilter(cv::Mat &img, int operation, int kernelSize);

int imageThresholding(std::string imgPath);

int videoThresholding();

// int customThreshold(const cv::Mat &src, cv::Mat &dst, double threshValue);

#endif