// Author: Kevin Heleodoro
// Date: February 18, 2024
// Purpose: A collection of utils used for object recognition

#include <fstream>
#include <string>
#include <vector>

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

int MIN_AREA = 300;

const char *windowDetectName = "Object Detection";
const char *windowSegmentName = "Region Segmentation";
const char *windowName = "Capture";

bool trainingMode = false;
/////////

cv::RotatedRect calcFeatures(const cv::Mat &regionMap, const cv::Mat &stats, int regionId)
{
    int x = stats.at<int>(regionId, cv::CC_STAT_LEFT);
    int y = stats.at<int>(regionId, cv::CC_STAT_TOP);
    int width = stats.at<int>(regionId, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(regionId, cv::CC_STAT_HEIGHT);

    cv::Mat mask = regionMap == regionId;
    cv::Moments m = cv::moments(mask, true);
    cv::Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);
    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    cv::RotatedRect box = cv::minAreaRect(points);
    printf("Centroid: (%f, %f)\n", centroid.x, centroid.y);
    printf("Width: %d, Height: %d\n", width, height);
    printf("Angle: %f\n", box.angle);
    printf("Area: %f\n", m.m00);
    printf("Bounding Box: (%f, %f), (%f, %f)\n", box.size.width, box.size.height, box.center.x, box.center.y);
    // print

    return box;
}

// calculates area, coordinates of centroid, and aspect ratio
std::vector<float> calcFeatures(const cv::Mat &regionMap, int regionId)
{
    std::vector<float> features;
    cv::Mat mask = regionMap == regionId;
    cv::Moments m = cv::moments(mask, true);
    // Area
    features.push_back(m.m00);
    // Centroid
    if (m.m00 != 0)
    {
        features.push_back(m.m10 / m.m00);
        features.push_back(m.m01 / m.m00);
    }
    else
    {
        features.push_back(0);
        features.push_back(0);
    }
    // Aspect Ratio
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (!contours.empty())
    {
        cv::Rect boundingBox = cv::boundingRect(contours[0]);
        float aspectRatio = static_cast<float>(boundingBox.width) / boundingBox.height;
        features.push_back(aspectRatio);
    }
    else
    {
        features.push_back(0);
    }

    return features;
}

void saveFeatures(const std::string &label, const std::vector<float> &features, const std::string &filename)
{
    std::ofstream file(filename, std::ios::app);
    if (file.is_open())
    {
        file << label;
        for (const auto &feature : features)
        {
            file << "," << feature;
        }
        file << "\n";
        file.close();
    }
    else
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
    trainingMode = false;
}

void labelAndSaveFeatures(const cv::Mat &regionMap, int regionId, const std::string &filename)
{
    std::string label;
    std::cout << "Enter the label for the region: ";
    std::cin >> label;

    auto features = calcFeatures(regionMap, regionId);
    saveFeatures(label, features, filename);
}

void drawFeatures(cv::Mat &src, const std::vector<cv::RotatedRect> &boxes)
{
    for (auto box : boxes)
    {
        cv::Point2f vertices[4];
        box.points(vertices);
        for (int i = 0; i < 4; i++)
        {
            cv::line(src, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
    // cv::imshow("Features", src);
}

void morphologicalFilter(cv::Mat &img, int operation, int kernelSize)
{
    // Create the structuring element for the morphological operation
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                                                cv::Point(kernelSize, kernelSize));

    // Apply the morphological operation
    cv::morphologyEx(img, img, operation, element);
}

void segmentedRegions(const cv::Mat &img, bool trainingMode = false)
{
    // Ensure img has been tresholded
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(img, labels, stats, centroids);

    // Create a new image to display the result
    cv::Mat output = cv::Mat::zeros(img.size(), CV_8UC3);

    std::vector<cv::RotatedRect> boxes;

    // Generate random colors for each region
    std::vector<cv::Vec3b> colors(nLabels);
    colors[0] = cv::Vec3b(0, 0, 0); // background
    for (int i = 1; i < nLabels; i++)
    {
        colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= MIN_AREA)
        {
            // do not include regions that touch the border of the frame
            if (stats.at<int>(i, cv::CC_STAT_LEFT) == 0 || stats.at<int>(i, cv::CC_STAT_TOP) == 0 ||
                stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH) == img.cols ||
                stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT) == img.rows)
            {
                continue;
            }

            boxes.push_back(calcFeatures(labels, stats, i));
        }
    }

    // Paint each region with a random color
    for (int i = 0; i < labels.rows; i++)
    {
        for (int j = 0; j < labels.cols; j++)
        {
            // only paint the region if it is within the boxes array

            // ###
            // for (auto box : boxes)
            // {
            //     cv::Point2f vertices[4];
            //     box.points(vertices);
            //     std::vector<cv::Point2f> points(vertices, vertices + 4); // Convert array to vector
            //     cv::Point2f point(j, i);
            //     if (cv::pointPolygonTest(points, point, false) >= 0) // Use the vector of points
            //     {
            //         cv::Vec3b &pixel = output.at<cv::Vec3b>(i, j);
            //         pixel = colors[labels.at<int>(i, j)];
            //     }
            // }
            // ###

            // Check is pixel is within the boxes

            int label = labels.at<int>(i, j);
            cv::Vec3b &pixel = output.at<cv::Vec3b>(i, j);
            pixel = colors[label];
        }
    }

    // Filter out small regions
    // for (int i = 1; i < nLabels; i++)
    // {
    //     if (stats.at<int>(i, cv::CC_STAT_AREA) < MIN_AREA)
    //         continue;

    //     int left = stats.at<int>(i, cv::CC_STAT_LEFT);
    //     int top = stats.at<int>(i, cv::CC_STAT_TOP);
    //     int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
    //     int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    // }

    drawFeatures(output, boxes);

    if (trainingMode)
    {
        labelAndSaveFeatures(labels, 1, "features.csv");
    }

    // Display the result
    cv::imshow(windowSegmentName, output);
}

void thresholdDemo(int, void *)
{
    int currentTrackbarValue = cv::getTrackbarPos(trackbarValue, windowName);
    int currentTrackbarMorphKernel = cv::getTrackbarPos(morphKernel, windowName);

    cv::threshold(gray, dst, currentTrackbarValue, maxBinaryValue,
                  threshType); // Will need to make my own custom version eventually

    // Apply morphological opening to reduce noise
    // morphologicalFilter(dst, cv::MORPH_OPEN, currentTrackbarMorphKernel);

    // Apply morphological closing to fill in gaps
    morphologicalFilter(dst, cv::MORPH_CLOSE, currentTrackbarMorphKernel);

    segmentedRegions(dst, trainingMode);

    // Display the result
    // cv::imshow(windowDetectName, dst);
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
        if (key == 'q' || key == 'Q')
        {
            std::cout << "User terminated program" << std::endl;
            break;
        }
        if (key == 's' || key == 'S')
        {
            std::cout << "Saving frame" << std::endl;
            imwrite("frame.png", frame);
        }
        if (key == 'n' || key == 'N')
        {
            std::cout << "Training Mode entered..." << std::endl;
            trainingMode = !trainingMode;
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
