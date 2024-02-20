// Author: Kevin Heleodoro
// Date: February 18, 2024
// Purpose: A collection of utils used for object recognition

// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/videoio.hpp>

#include "object_utils.h"

using namespace cv;

////////// Global variables
cv::Mat gray;
cv::Mat dst;
int threshValue = 0;
int threshType = 1;
// int threshType = 3;
int const maxVal = 255;
int const maxType = 4;
int const maxBinaryValue = 255;
const char *windowDetectName = "Object Detection";
const char *windowName = "Capture";
const char *trackbarType =
    "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char *trackbarValue = "Value";
/////////

void thresholdDemo(int, void *)
{
    int currentTrackbarValue = cv::getTrackbarPos(trackbarValue, windowName);
    int currentTrackbarType = cv::getTrackbarPos(trackbarType, windowName);

    cv::threshold(gray, dst, currentTrackbarValue, maxBinaryValue,
                  currentTrackbarType); // Will need to make my own custom version eventually
    cv::imshow(windowDetectName, dst);
}

int imageThresholding(std::string imgPath)
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Could not read the image: " << imgPath << std::endl;
        return -1;
    }

    // Preprocessing
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // convert to grayscale

    namedWindow(windowName, cv::WINDOW_AUTOSIZE); // Create Window for results
    cv::createTrackbar(trackbarType, windowName, NULL, maxType,
                       thresholdDemo); // Create Trackbar to choose type of Threshold
    cv::createTrackbar(trackbarValue, windowName, NULL, maxVal,
                       thresholdDemo); // Create Trackbar to choose Threshold value
    thresholdDemo(0, 0);               // Call the function to initialize
                                       // waitKey(0);                        // Wait for any keystroke in the window
    char key = (char)waitKey(0);
    if (key == 'q')
    {
        std::cout << "User terminated program" << std::endl;
    }
    if (key == 's')
    {
        std::cout << "Saving frame" << std::endl;
        imwrite("frame.png", dst);
    }
    return 0;
}

int videoThresholding()
{
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    namedWindow(windowDetectName, cv::WINDOW_AUTOSIZE);

    createTrackbar(trackbarType, windowName, NULL, maxType, thresholdDemo);
    createTrackbar(trackbarValue, windowName, NULL, maxVal, thresholdDemo);

    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Error: Could not capture frame" << std::endl;
            break;
        }

        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        thresholdDemo(0, 0);
        imshow(windowName, frame);

        char key = (char)waitKey(30);
        if (key == 'q')
        {
            std::cout << "User terminated program" << std::endl;
            break;
        }
        if (key == 's')
        {
            std::cout << "Saving frame" << std::endl;
            imwrite("frame.png", frame);
        }
    }
    return 0;
}

int customThreshold(const cv::Mat &src, cv::Mat &dst, double threshValue)
{
    // Convert to grayscale
    cv::Mat gray;
    if (src.channels() == 3)
    {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = src.clone();
    }

    // Create destination image and initialize to black
    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    // Apply thresholding
    for (int i = 0; i < gray.rows; i++)
    {
        for (int j = 0; j < gray.cols; j++)
        {
            uchar pixel = gray.at<uchar>(i, j);

            // If the pixel is above the trehsold, set it to white
            if (pixel > threshValue)
            {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }

    return 0;
}
