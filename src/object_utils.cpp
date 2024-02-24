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
bool classifyMode = false;

std::vector<std::pair<std::string, std::vector<float>>> trainingSet;
std::string dataset = "features.csv";
/////////

// Compare vector features to a set of features
// Returns the index of the closest match
int compareFeatures(const std::vector<float> &features,
                    const std::vector<std::pair<std::string, std::vector<float>>> &trainingSet)
{
    int closestMatch = -1;
    float minDistance = std::numeric_limits<float>::max();

    for (int i = 0; i < trainingSet.size(); i++)
    {
        float distance = 0;
        for (int j = 0; j < features.size(); j++)
        {
            distance += (features[j] - trainingSet[i].second[j]) * (features[j] - trainingSet[i].second[j]);
        }
        distance = std::sqrt(distance);
        if (distance < minDistance)
        {
            printf("Distance for %s: %f\n", trainingSet[i].first.c_str(), distance);
            minDistance = distance;
            closestMatch = i;
        }
    }

    if (minDistance > 1000)
    {
        printf("No match found\n");
        return -1;
    }

    return closestMatch;
}

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

        double largestArea = 0;
        for (const auto &contour : contours)
        {
            double area = cv::contourArea(contour);
            if (area > largestArea)
            {
                largestArea = area;
            }
        }
        features.push_back(largestArea);
    }
    else
    {
        features.push_back(0);
        features.push_back(0);
    }

    return features;
}

std::vector<std::pair<std::string, std::vector<float>>> loadTrainingSet(const std::string &filename)
{
    std::vector<std::pair<std::string, std::vector<float>>> featureVectors;
    std::ifstream file(filename);
    std::string line;
    printf("Loading training set from %s\n", filename.c_str());

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string item;
        std::vector<float> vector;
        std::string filename;
        std::getline(ss, filename, ',');

        while (std::getline(ss, item, ','))
        {
            vector.push_back(std::stof(item));
        }

        featureVectors.emplace_back(filename, vector);
    }
    file.close();
    printf("Loaded features: %s\n", featureVectors.at(0).first.c_str());
    printf("Loaded %lu feature vectors\n", featureVectors.size());

    return featureVectors;
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
    trainingSet = loadTrainingSet(dataset);
}

void drawFeatures(cv::Mat &src, const std::vector<cv::RotatedRect> &boxes, std::string result = "")
{
    for (auto box : boxes)
    {
        cv::Point2f vertices[4];
        box.points(vertices);
        for (int i = 0; i < 4; i++)
        {
            cv::line(src, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }

        float maxY = vertices[0].y;
        for (int i = 1; i < 4; i++)
        {
            if (vertices[i].y > maxY)
            {
                maxY = vertices[i].y;
            }
        }

        cv::putText(src, result, cv::Point(vertices[0].x, maxY + 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
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

void segmentedRegions(const cv::Mat &img, bool isTraining = false, bool isClassify = false)
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
            int label = labels.at<int>(i, j);
            cv::Vec3b &pixel = output.at<cv::Vec3b>(i, j);
            pixel = colors[label];
        }
    }

    // drawFeatures(output, boxes);

    if (isTraining)
    {
        labelAndSaveFeatures(labels, 1, "features.csv");
    }
    // if (isClassify)
    // {
    // printf("Classifying object ...\n");
    int closestMatch = compareFeatures(calcFeatures(labels, 1), trainingSet);
    std::string result;

    if (closestMatch == -1)
    {
        // std::cout << "No match found" << std::endl;
        result = "No match found";
    }
    else
    {
        result = trainingSet[closestMatch].first;
    }

    // std::cout << "Closest match: " << result << std::endl;

    // add label to the image
    // cv::putText(output, result, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    drawFeatures(output, boxes, result);

    // classifyMode = false;
    // }

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

    segmentedRegions(dst, trainingMode, classifyMode);

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

    // Load vectors from file
    trainingSet = loadTrainingSet(dataset);

    // Create the input and output windows
    namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    // namedWindow(windowDetectName, cv::WINDOW_AUTOSIZE);

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
        if (key == 'c' || key == 'C')
        {
            std::cout << "Attempting to classify object ..." << std::endl;
            classifyMode = !classifyMode;
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
