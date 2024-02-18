#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

// using namespace std;
// using namespace cv;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: obj_recog <image_path>" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Could not read the image: " << argv[1] << std::endl;
        return -1;
    }

    // Preprocessing
    cv::Mat gray;
    cv::Mat blurred;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // Dynamic thresholding (ISODATA algorithm)
    cv::Mat labels;
    cv::Mat centers;
    cv::Mat sample = blurred.reshape(1, blurred.total());
    sample.convertTo(sample, CV_32F);
    cv::kmeans(sample, 2, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3,
               cv::KMEANS_PP_CENTERS, centers);

    double threshValue = (centers.at<float>(0, 0) + centers.at<float>(1, 0)) / 2;
    cv::Mat thresh;
    cv::threshold(blurred, thresh, threshValue, 255, cv::THRESH_BINARY);

    // Display original and thresholded images
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Thresholded", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", img);
    cv::imshow("Thresholded", thresh);

    cv::waitKey(0);
    return 0;
}