#include "digitrecognizer.h"

DigitRecognizer::DigitRecognizer()
{
  knn = ml::KNearest::create();


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
  //Mat cloneImg = preprocessImage(img);
  Mat cloneImg;
  resize(img, cloneImg, Size(100,100));

  Mat cloneImgFloat;
  cloneImg.convertTo(cloneImgFloat, CV_32FC1); 

  Mat cloneImgFloatFlat;
  cloneImgFloatFlat = cloneImgFloat.reshape(1, 1);

  Mat matValue(0, 0, CV_32F);  

  knn->findNearest(cloneImgFloatFlat, 1, matValue);
  
  float fltValue = (float)matValue.at<float>(0, 0);
  std::cout << char(int(fltValue));
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

//
///*
//  preprocessImage: using KNN on raw image doesn't provide accurate classification results
//                  Instead we will center the digit to be similar to the MNIST dataset
//
//  params:
//    img- openCV matrix image to be preprocessed before being classified
//
//  returns:
//    img- openCV matrix centered ready to be classified
//*/
//
//Mat DigitRecognizer::preprocessImage(Mat img)
//{
//  int rowTop = -1, rowBottom= -1, colLeft=-1, colRight=-1;
//
//  Mat temp;
//  int thresholdBottom = 50;
//  int thresholdTop = 50;
//  int thresholdLeft = 50;
//  int thresholdRight = 50;
//  int center = img.rows/2;
//
//  for(int i = center; i<img.rows; i++)
//  {
//    if(rowBottom==-1)
//    {
//      temp = img.row(i);
//      IplImage stub = temp;
//      if(cvSum(&stub).val[0] < thresholdBottom || i==img.rows-1)
//      {
//        rowBottom = i;
//      }
//    }
//    
//    if(rowTop==-1)
//    {
//      temp = img.row(img.rows-i);
//      IplImage stub = temp;
//      if(cvSum(&stub).val[0] < thresholdTop || i==img.rows-1)
//      {
//        rowTop = img.rows-i;
//      }
//    }
// 
//   if(colRight==-1)
//    {
//      temp = img.col(i);
//      IplImage stub = temp;
//      if(cvSum(&stub).val[0] < thresholdRight || i==img.cols-1)
//      {
//        colRight = i;
//      }
//    }
//
//   
//    if(colLeft==-1)
//    {
//      temp = img.col(img.cols-i);
//      IplImage stub = temp;
//      if(cvSum(&stub).val[0] < thresholdLeft || i==img.cols-1)
//      {
//        colLeft = img.cols-i;
//      }
//    }
//
//  }
//
//  Mat newImg;
//  
//  newImg = newImg.zeros(img.rows, img.cols, CV_8UC1);
//
//  int startAtX = (newImg.cols/2)-(colRight-colLeft)/2;
//  int startAtY = (newImg.rows/2)-(rowBottom-rowTop)/2;
//
//  for(int y=startAtY; y<(newImg.rows/2)+(rowBottom-rowTop)/2; y++)
//  {
//    
//    uchar *ptr = newImg.ptr<uchar>(y);
//
//    for(int x=startAtX; x<(newImg.cols/2)+(colRight-colLeft)/2; x++)
//    {
//      ptr[x] = img.at<uchar>(rowTop + (y-startAtY), colLeft+(x-startAtX));
//    }
//  }
//
//  Mat cloneImg = Mat(numRows, numCols, CV_8UC1);
//
//  resize(newImg, cloneImg, Size(numCols, numRows));
//  
//  for(int i=0; i<cloneImg.rows; i++)
//  {
//    floodFill(cloneImg, cvPoint(0, i), cvScalar(0,0,0));
//    floodFill(cloneImg, cvPoint(cloneImg.cols-1, i), cvScalar(0,0,0));
//
//    floodFill(cloneImg, cvPoint(i, 0), cvScalar(0));
//    floodFill(cloneImg, cvPoint(i, cloneImg.rows-1), cvScalar(0));
//  }
//  
//  cloneImg = cloneImg.reshape(1, 1);
//  return cloneImg;
//}
