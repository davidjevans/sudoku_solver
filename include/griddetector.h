#include <stdio.h>
#include <vector>
//#include <iostream>
//#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>

namespace gd
{
  cv::Mat findLargestBlob(cv::Mat outerBox);
  void mergeRelatedLines(std::vector<cv::Vec2f> *lines, cv::Mat &img);

  void drawLine(cv::Vec2f line, cv::Mat &img, cv::Scalar rgb);
  cv::Mat warpImage(cv::Mat &img, std::vector<cv::Point> edgePoints, int maxLength);
  std::vector<cv::Point> findCornerPoints(cv::Mat &outerBox, std::vector<cv::Vec2f> lines);


}
