// Author: Kevin Heleodoro
// Date: February 18, 2024
// Purpose: A collection of utils used for object recognition

#include "object_utils.h"

using namespace cv;

////////// Global variables
cv::Mat gray;
cv::Mat dst;
int threshValue = 0;
int const maxVal = 255;
int const maxBinaryValue = 255;
const char *trackbarValue = "Value";

int morphKernelSize = 1;
int maxMorphKernelSize = 25;
const char *morphKernel = "Morph Kernel";
int threshType = 1; // Binary Inverted threshold
const char *windowDetectName = "Object Detection";
const char *windowName = "Capture";
/////////

void morphologicalFilter(cv::Mat &img, int operation, int kernelSize)
{
    // Create the structuring element for the morphological operation

    /**
     *  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
     *  This is the original line of code, but it is not flexible enough to allow for different kernel sizes.
     *  It also does not guarantee that the kernel will be centered.
     */

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                                                cv::Point(kernelSize, kernelSize));

    // Apply the morphological operation
    cv::morphologyEx(img, img, operation, element);
}

void thresholdDemo(int, void *)
{
    int currentTrackbarValue = cv::getTrackbarPos(trackbarValue, windowName);
    int currentTrackbarMorphKernel = cv::getTrackbarPos(morphKernel, windowName);

    cv::threshold(gray, dst, currentTrackbarValue, maxBinaryValue,
                  threshType); // Will need to make my own custom version eventually

    // Apply morphological opening to reduce noise
    // morphologicalFilter(dst, cv::MORPH_OPEN, 5);

    // Apply morphological closing to fill in gaps
    morphologicalFilter(dst, cv::MORPH_CLOSE, currentTrackbarMorphKernel);

    // Display the result
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

    // Convert to grayscale
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Create a window to display the image
    namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Create Trackbar to choose Morph Kernel size
    cv::createTrackbar(morphKernel, windowName, NULL, maxMorphKernelSize, thresholdDemo);
    // Create Trackbar to choose Threshold value
    cv::createTrackbar(trackbarValue, windowName, NULL, maxVal, thresholdDemo);

    // Call the function to initialize
    thresholdDemo(0, 0);

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

    // Create the input and output windows
    namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    namedWindow(windowDetectName, cv::WINDOW_AUTOSIZE);

    // Create Trackbar to choose Morph Kernel size
    createTrackbar(morphKernel, windowName, NULL, maxMorphKernelSize, thresholdDemo);
    // Create Trackbar to choose Threshold value
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
