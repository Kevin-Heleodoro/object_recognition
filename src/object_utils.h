// Author: Kevin Heleodoro
// Date: February 18, 2024
// Purpose: A collection of utils used for object recognition

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#ifndef OBJECT_UTILS_H
#define OBJECT_UTILS_H

int imageThresholding(std::string imgPath);

int videoThresholding();

#endif