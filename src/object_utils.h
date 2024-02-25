// Author: Kevin Heleodoro
// Date: February 18, 2024
// Purpose: A collection of utils used for object recognition

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#ifndef OBJECT_UTILS_H
#define OBJECT_UTILS_H

/**
 * @brief Applies thresholding to an image. The user can choose the threshold value and the morphological kernel size
 * using trackbars.
 *
 * @param imgPath The path to the image to apply thresholding to
 * @return int
 */
int imageThresholding(std::string imgPath);

/**
 * @brief Applies thresholding to a video stream. The user can choose the threshold value and the morphological kernel
 * size using trackbars.
 *
 * @return int
 */
int videoThresholding();

#endif