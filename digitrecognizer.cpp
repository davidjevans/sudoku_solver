#include "digitrecognizer.h"

DigitRecognizer::DigitRecognizer()
{
  knn = new ::ml::KNearest();

}

DigitRecognizer::~DigitRecognizer()
{
  delete knn;
}

/*
  readFlippedInteger: Reads an integer in reverse bit order

  *Intel processors read a different "endian" format from the MNIST dataset.
    This means in order to read an integer in the MNIST dataset, you must flp the order of the bits  

  params:
    fp is the file with the integer to be flipped

  return:
    the integer value of the flipped bits
*/
int DigitRecognizer::readFlippedInteger(FILE *fp)
{
  int ret = 0;
  
  char *temp;

  temp = (char*)(&ret);
  
  fread(&temp[3], sizeof(char), 1, fp);
  fread(&temp[2], sizeof(char), 1, fp);
  fread(&temp[1], sizeof(char), 1, fp);
  fread(&temp[0], sizeof(char), 1, fp);

  return ret;
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
  //open the training data and training labels paths
  FILE *fp = fopen(trainPath, "rb");
  FILE *fp2 = fopen(labelsPath, "rb");

  //only advance if both paths opened
  if(!fp || !fp2)
    return false;

  //Read bytes in flipped order
  //order of MNIST dataset is 
  //  magic number (4 bytes)
  //  number of images (4 bytes)
  //  number of pixel rows in image (4 bytes)
  //  number of pizel columns in image (4 bytes)
  int magicNumber = readFlippedInteger(fp);
  numImages = readFlippedInteger(fp);
  numRows = readFlippedInteger(fp);
  numCols = readFlippedInteger(fp);

  fseek(fp2, 0x08, SEEK_SET);

  if(numImages > MAX_NUM_IMAGES) numImages = MAX_NUM_IMAGES;

  ///////////////////////////////////////////////////////////
  //Create a list of the training images and training labels to train on
  //traininingVectors: list of images
  //trainingLabels: list of labels
  int size = numRows*numCols;

  CvMat *trainingVectors = cvCreateMat(numImages, size, CV_32FC1);

  CvMat *trainingClasses = cvCreateMat(numImages, 1, CV_32FC1);

  memset(trainingClasses->data.ptr, 0, sizeof(float)*numImages);

  char *temp = new char[size];
  char tempClass = 0;

  for(int i = 0; i < numImages; i++)
  {
    fread((void*)temp, size, 1, fp);
    fread((void*)(&tempClass), sizeof(char), 1, fp2);

    trainingClasses->data.fl[i] = tempClass;

    for(int k=0; k<size; k++)
    {
       trainingVectors->data.fl[i*size+k] = temp[k];
    }

  }

  //train the openCV knn classifier with the images and labels
  knn->train(trainingVectors, trainingClasses);
  fclose(fp);
  fclose(fp2);

  return true;
}

/*
  classify: Given an image, classify the image as an integer
 
  params;
    img- openCV matrix image to be classified

  returns:
    integer the matrix is classified as
*/

int DigitRecognizer::classify(cv::Mat img)
{
  Mat cloneImg = preprocessImage(img);
  return knn->find_nearest(Mat_<float>(cloneImg), 1);
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
  int rowTop = -1, rowBottom= -1, colLeft=-1, colRight=-1;

  Mat temp;
  int thresholdBottom = 50;
  int thresholdTop = 50;
  int thresholdLeft = 50;
  int thresholdRight = 50;
  int center = img.rows/2;

  for(int i = center; i<img.rows; i++)
  {
    if(rowBottom==-1)
    {
      temp = img.row(i);
      IplImage stub = temp;
      if(cvSum(&stub).val[0] < thresholdBottom || i==img.rows-1)
      {
        rowBottom = i;
      }
    }
    
    if(rowTop==-1)
    {
      temp = img.row(img.rows-i);
      IplImage stub = temp;
      if(cvSum(&stub).val[0] < thresholdTop || i==img.rows-1)
      {
        rowTop = img.rows-i;
      }
    }
 
   if(colRight==-1)
    {
      temp = img.col(i);
      IplImage stub = temp;
      if(cvSum(&stub).val[0] < thresholdRight || i==img.cols-1)
      {
        colRight = i;
      }
    }

   
    if(colLeft==-1)
    {
      temp = img.col(img.cols-i);
      IplImage stub = temp;
      if(cvSum(&stub).val[0] < thresholdLeft || i==img.cols-1)
      {
        colLeft = img.cols-i;
      }
    }

  }

  Mat newImg;
  
  newImg = newImg.zeros(img.rows, img.cols, CV_8UC1);

  int startAtX = (newImg.cols/2)-(colRight-colLeft)/2;
  int startAtY = (newImg.rows/2)-(rowBottom-rowTop)/2;

  for(int y=startAtY; y<(newImg.rows/2)+(rowBottom-rowTop)/2; y++)
  {
    
    uchar *ptr = newImg.ptr<uchar>(y);

    for(int x=startAtX; x<(newImg.cols/2)+(colRight-colLeft)/2; x++)
    {
      ptr[x] = img.at<uchar>(rowTop + (y-startAtY), colLeft+(x-startAtX));
    }
  }

  Mat cloneImg = Mat(numRows, numCols, CV_8UC1);

  resize(newImg, cloneImg, Size(numCols, numRows));
  
  for(int i=0; i<cloneImg.rows; i++)
  {
    floodFill(cloneImg, cvPoint(0, i), cvScalar(0,0,0));
    floodFill(cloneImg, cvPoint(cloneImg.cols-1, i), cvScalar(0,0,0));

    floodFill(cloneImg, cvPoint(i, 0), cvScalar(0));
    floodFill(cloneImg, cvPoint(i, cloneImg.rows-1), cvScalar(0));
  }
  
  cloneImg = cloneImg.reshape(1, 1);
  return cloneImg;
}
