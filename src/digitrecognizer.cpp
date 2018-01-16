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
//
///*
//  readFlippedInteger: Reads an integer in reverse bit order
//
//  *Intel processors read a different "endian" format from the MNIST dataset.
//    This means in order to read an integer in the MNIST dataset, you must flp the order of the bits  
//
//  params:
//    fp is the file with the integer to be flipped
//
//  return:
//    the integer value of the flipped bits
//*/
//int DigitRecognizer::readFlippedInteger(FILE *fp)
//{
//  int ret = 0;
//  
//  char *temp;
//
//  temp = (char*)(&ret);
//  
//  fread(&temp[3], sizeof(char), 1, fp);
//  fread(&temp[2], sizeof(char), 1, fp);
//  fread(&temp[1], sizeof(char), 1, fp);
//  fread(&temp[0], sizeof(char), 1, fp);
//
//  return ret;
//}

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

// resize(img, cloneImg, Size(100,100));
// cloneImg.convertTo(cloneImg, CV_32FC1);  
//
// cloneImg = cloneImg.reshape(1, 1);

 cloneImg = preprocessImage(img);

  if(!cloneImg.empty())
  {
    Mat matValue(0, 0, CV_32F);  

    std::cout<<cloneImg.rows << "," <<cloneImg.cols<<std::endl; 
    knn->findNearest(cloneImg, 1, matValue);
 
    float fltValue = (float)matValue.at<float>(0, 0);
    char value = char(fltValue);
    std::cout << value;
    return 0;
  }
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

  Moments m = cv::moments(img, true);
  int area = m.m00;
  int imgArea = img.rows*img.cols;

  int MIN_CONTOUR_AREA = ceil(imgArea*.13);
  int MAX_CONTOUR_AREA = ceil(imgArea*.4);


  for (int i = 0; i < ptContours.size(); i++) {                           // for each contour
        
    Rect boundingRect = cv::boundingRect(ptContours[i]);
    
    int rectArea = boundingRect.width*boundingRect.height;

    if (rectArea > MIN_CONTOUR_AREA && rectArea < MAX_CONTOUR_AREA) 
    {
      
      Mat cloneImg;
      
      Mat boundedRect = img(boundingRect);

      resize(boundedRect, cloneImg, Size(numCols, numRows));
      
      std::cout << imgArea <<" , " << rectArea << " , " << contourArea(ptContours[i]) << std::endl;      
      imshow("window", cloneImg);
      int intChar = cv::waitKey(0);           // get key press
  
      Mat cloneImgFloat;
      cloneImg.convertTo(cloneImgFloat, CV_32FC1); 
    
      Mat cloneImgFloatFlat;
      cloneImgFloatFlat = cloneImgFloat.reshape(1, 1);
//      return cloneImgFloatFlat;
     }
  }
  
  Mat emptyImg;

  return emptyImg;
}
