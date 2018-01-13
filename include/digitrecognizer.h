#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ml.h>

#include <iostream>

using namespace cv;
#define MAX_NUM_IMAGES    60000
class DigitRecognizer
{
public:
    DigitRecognizer();

    ~DigitRecognizer();

    bool train(char* trainPath, char* labelsPath);

    int classify(Mat img);

private:
    Mat preprocessImage(Mat img);

    Mat importData(char* path);

    int readFlippedInteger(FILE *fp);
    

private:
    Ptr<ml::KNearest> knn;
    int numRows, numCols, numImages;

};
