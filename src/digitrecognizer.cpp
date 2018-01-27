#include "digitrecognizer.h"

DigitRecognizer::DigitRecognizer()
{
  knn = ml::KNearest::create();
  numRows = 50;
  numCols = 50;

}

DigitRecognizer::~DigitRecognizer()
{
  delete knn;
}

/*
  train: reads in the MNIST images and labels to train the OpenCV KNN classifier

  params:
    trainPath- pointer to training data
    labelsPath- pointer to training labels

  return:
    true if trained properly, false otherwise
*/

bool DigitRecognizer::train(char *trainPath, char *labelsPath)
{
  Mat trainingLabels = importData(labelsPath);
  Mat trainingImages = importData(trainPath);

  if(trainingLabels.empty() || trainingImages.empty())
  {
    std::cout<<"Training data not loaded correctly";
    return false;
  }   

  knn->train(trainingImages, ml::ROW_SAMPLE, trainingLabels);
  
 
  return true;

}

/*
  classify: Given an image, classify the image as an integer
 
  params;
    img- openCV matrix image to be classified

  returns:
    integer the matrix is classified as
*/
//
int DigitRecognizer::classify(cv::Mat img)
{
  Mat cloneImg;

  cloneImg = preprocessImage(img);

  if(!cloneImg.empty())
  {
    Mat matValue(0, 0, CV_32F);  

    knn->findNearest(cloneImg, 1, matValue);
 
    float value = (float)matValue.at<float>(0, 0);
    
    return value - 48;
  }
  return 0;
}





Mat DigitRecognizer::importData(char *path)
{
  Mat foi; 

  FileStorage fs(path, FileStorage::READ);
  
  if(fs.isOpened() == false)
  {
    std::cout<<"Error: Cannot open " << path;
    return foi;
  }
  fs["doi"] >> foi;
  fs.release();
  return foi;
}


/*
  preprocessImage: using KNN on raw image doesn't provide accurate classification results
                  Instead we will center the digit to be similar to the MNIST dataset

  params:
    img- openCV matrix image to be preprocessed before being classified

  returns:
    img- openCV matrix centered ready to be classified
*/

Mat DigitRecognizer::preprocessImage(Mat img)
{

  std::vector<std::vector<cv::Point> > ptContours;        // declare contours vector

  findContours(img, ptContours, RETR_LIST, CHAIN_APPROX_SIMPLE);           

  int imgArea = img.rows*img.cols;

  int MIN_CONTOUR_AREA = ceil(imgArea*.13);
  int MAX_CONTOUR_AREA = ceil(imgArea*.4);
  int MIN_WHITE_AREA = ceil(imgArea*.14);

  float cropPercent = .05;
  int tbBoundary = ceil(img.rows*cropPercent);
  int lrBoundary = ceil(img.cols*cropPercent);

  Rect cropRegion(lrBoundary/2, tbBoundary/2, img.cols-lrBoundary, img.rows-tbBoundary);
  
  Mat croppedImg = img(cropRegion);
  Moments m = moments(croppedImg, true);
  int whitePixelArea = m.m00;

//  std::cout << "Area:" << whitePixelArea << std::endl;
  if (whitePixelArea > MIN_WHITE_AREA){
    for (int i = 0; i < ptContours.size(); i++) {                           // for each contour
          
      Rect boundingRect = cv::boundingRect(ptContours[i]);
      
      int rectArea = boundingRect.width*boundingRect.height;

      if (rectArea > MIN_CONTOUR_AREA && rectArea < MAX_CONTOUR_AREA) 
      {
        
        Mat cloneImg;
        
        Mat boundedRect = img(boundingRect);

        resize(boundedRect, cloneImg, Size(numCols, numRows));
        
        namedWindow("Display window", WINDOW_AUTOSIZE);
        imshow("Display window", cloneImg);
        waitKey(0);

        Mat cloneImgFloat;
        cloneImg.convertTo(cloneImgFloat, CV_32FC1); 
              
        Mat cloneImgFloatFlat;
        cloneImgFloatFlat = cloneImgFloat.reshape(1, 1);
        return cloneImgFloatFlat;
      }
    }
  }
  
  Mat emptyImg;

  return emptyImg;
}
