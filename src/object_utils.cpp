// Author: Kevin Heleodoro
// Date: February 18, 2024
// Purpose: A collection of utils used for object recognition

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "object_utils.h"

using namespace cv;

////////// Global variables
cv::Mat gray;
cv::Mat dst;

int threshValue = 0;
int threshType = 1; // Binary Inverted threshold
int const maxVal = 255;
int const maxBinaryValue = 255;
const char *trackbarValue = "Value";

int morphKernelSize = 1;
int maxMorphKernelSize = 25;
const char *morphKernel = "Morph Kernel";

int k = 6;
int MIN_AREA = 300;

const char *windowDetectName = "Object Detection";
const char *windowSegmentName = "Region Segmentation";
const char *windowName = "Capture";

bool trainingMode = false;
bool classifyMode = false;

std::vector<std::pair<std::string, std::vector<float>>> trainingSet;
std::string dataset = "features.csv";

/////////

/**
 * @brief Compares the features of a region to the features of the training set. This is done by calculating the
 * euclidean distance between the features of the region and the features of each training set entry.
 *
 * @param features The features of the region to compare
 * @param trainingSet The training set
 * @return int The index of the closest match in the training set
 */
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

/**
 * @brief Compares the features of a region to the features of the training set. This is done by calculating the
 * euclidean distance between the features of the region and the features of each training set entry. This version
 * uses the k-nearest neighbors algorithm to classify the region.
 *
 * @param features The features of the region to compare
 * @param trainingSet The training set
 * @param k The number of neighbors to consider
 * @return std::string The label of the closest match in the training set
 */
std::string compareKNearest(const std::vector<float> &features,
                            const std::vector<std::pair<std::string, std::vector<float>>> &trainingSet, int k = 6)
{
    std::vector<std::pair<float, int>> distances;
    for (int i = 0; i < trainingSet.size(); i++)
    {
        float distance = 0;
        for (int j = 0; j < features.size(); j++)
        {
            distance += (features[j] - trainingSet[i].second[j]) * (features[j] - trainingSet[i].second[j]);
        }
        distance = std::sqrt(distance);
        distances.push_back(std::make_pair(distance, i));
    }

    std::sort(distances.begin(), distances.end());
    std::map<std::string, int> labels;
    for (int i = 0; i < k; i++)
    {
        labels[trainingSet[distances[i].second].first]++;
    }

    int max = 0;
    std::string result;
    for (auto &label : labels)
    {
        if (label.second > max)
        {
            max = label.second;
            result = label.first;
        }
    }

    return result;
}

/**
 * @brief Calculates the features of a region. The features are the area, centroid, aspect ratio, and largest area
 * of the region.
 *
 * @param regionMap The region map
 * @param stats The statistics of the region map
 * @param regionId The id of the region
 * @return cv::RotatedRect The features of the region
 */
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

/**
 * @brief Calculates the features of a region. The features are the area, centroid, aspect ratio, and largest area
 * of the region.
 *
 * @param regionMap The region map
 * @param regionId The id of the region
 * @return std::vector<float> The features of the region
 */
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

        // Largest Area
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

/**
 * @brief Loads the training set from a file. The training set is a list of feature vectors and their respective labels.
 * The file is expected to be in the following format:
 * label1,feature1,feature2,feature3,...,featureN
 *
 * @param filename The name of the file to load the training set from
 * @return std::vector<std::pair<std::string, std::vector<float>>> The training set
 */
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

/**
 * @brief Saves the features of a region to a file. The features are the area, centroid, aspect ratio, and largest area
 * of the region.
 *
 * @param label The label of the region
 * @param features The features of the region
 * @param filename The name of the file to save the features to
 * @return void
 */
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

/**
 * @brief Used to calculate the features of a region and save them to a file. The user is prompted to enter the label
 * of the region.
 *
 * @param regionMap The region map
 * @param regionId The id of the region
 * @param filename The name of the file to save the features to
 */
void labelAndSaveFeatures(const cv::Mat &regionMap, int regionId, const std::string &filename)
{
    std::string label;
    std::cout << "Enter the label for the region: ";
    std::cin >> label;

    auto features = calcFeatures(regionMap, regionId);
    saveFeatures(label, features, filename);
    trainingSet = loadTrainingSet(dataset);
}

/**
 * @brief Draws the bounding boxes of the regions and the label of the region on the image.
 *
 * @param src The source image
 * @param boxes The bounding boxes of the regions
 * @param result The label of the region
 */
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
}

/**
 * @brief Applies a morphological operation to an image.
 *
 * @param img The image to apply the operation to
 * @param operation The morphological operation to apply
 * @param kernelSize The size of the kernel to use
 */
void morphologicalFilter(cv::Mat &img, int operation, int kernelSize)
{
    // Create the structuring element for the morphological operation
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                                                cv::Point(kernelSize, kernelSize));

    // Apply the morphological operation
    cv::morphologyEx(img, img, operation, element);
}

/**
 * @brief Segments the regions of an image. The regions are painted with a random color and the features of each region
 * are calculated and displayed on the image.
 *
 * @param img The image to segment
 * @param isTraining Whether the function is being used in training mode
 * @param isClassify Whether the function is being used in classification mode
 */
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

    if (isTraining)
    {
        labelAndSaveFeatures(labels, 1, "features.csv");
    }

    // Gets the label of the closest match in the training set
    std::string result = compareKNearest(calcFeatures(labels, 1), trainingSet);

    // add border and label to the image
    drawFeatures(output, boxes, result);

    // Display the result
    cv::imshow(windowSegmentName, output);
}

/**
 * @brief Callback function for the thresholding demo. This function is called whenever the trackbars are changed.
 *
 */
void thresholdDemo(int, void *)
{
    // Get thresh and morph kernel values
    int currentTrackbarValue = cv::getTrackbarPos(trackbarValue, windowName);
    int currentTrackbarMorphKernel = cv::getTrackbarPos(morphKernel, windowName);

    cv::threshold(gray, dst, currentTrackbarValue, maxBinaryValue,
                  threshType); // Will need to make my own custom version eventually

    // Apply morphological closing to fill in gaps
    morphologicalFilter(dst, cv::MORPH_CLOSE, currentTrackbarMorphKernel);

    // Segment, classify, and display the regions
    segmentedRegions(dst, trainingMode, classifyMode);
}

/**
 * @brief Applies thresholding to an image. The user can choose the threshold value and the morphological kernel size
 * using trackbars.
 *
 * @param imgPath The path to the image to apply thresholding to
 * @return int
 */
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

/**
 * @brief Applies thresholding to a video stream. The user can choose the threshold value and the morphological kernel
 * size using trackbars.
 *
 * @return int
 */
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

/**
 * @brief Applies thresholding to a video stream. The user passes in the threshold value.
 *
 * @param src The source image
 * @param dst The destination image
 * @param threshValue The threshold value
 */
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
