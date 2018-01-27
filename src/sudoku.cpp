#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <vector>

#include "digitrecognizer.h"
#include "solver.h"
#include "griddetector.h"

using namespace cv;
using namespace std;

int main()
{

	//read in the sudoku image
	Mat original = imread("../sudoku.jpg", 0);

	//create an outer box container the same size as the sudoku image
	Mat outerBox = Mat(original.size(), CV_8UC1);
	Mat sudoku = Mat(original.size(), CV_8UC1);

	//return error if sudoku image not found
	if (!original.data)
	{
		printf("No image data \n");
		return -1;
	}

	//PREPROCESSING//////////////////////////////////////////
	//Gaussian blur to smooth the image to make grid line extraction easier
	GaussianBlur(original, sudoku, Size(11,11), 0);

	//Adaptive threshold used to achieve more accurate threshold by setting threshold as mean level in 5x5 pixel window
	adaptiveThreshold(sudoku, outerBox, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);

	//invert the threshold image to set areas of interest to white
	bitwise_not(outerBox, outerBox);

	//Dilate the image to smooth any discontinuities on lines caused by thresholding operation
	Mat kernel = (Mat_<uchar>(3,3) <<0,1,0,1,1,1,0,1,0);
	dilate(outerBox, outerBox, kernel);

  //Find the largest object in the image
  outerBox = gd::findLargestBlob(outerBox);

  //Erode the image to counter the previous dilate operation
	erode(outerBox, outerBox, kernel);
	////////////////////////////////////////////////////////////////////////////



	//DETECTING LINES//////////////////////////////////////////////////////////
	
	std::vector<Vec2f> lines;

	//generate lines based off pixels
	HoughLines(outerBox, lines, 1, CV_PI/180, 200);

	//merge similar lines together
	gd::mergeRelatedLines(&lines, sudoku);

	for(int i=0; i<lines.size(); i++)
	{
		gd::drawLine(lines[i], outerBox, CV_RGB(0,0,128));
	}
  
  vector<Point> cornerPoints = gd::findCornerPoints(outerBox, lines);

  int maxLength = gd::findMaxLength(cornerPoints);
  Mat undistorted = Mat(Size(maxLength, maxLength), CV_8UC1);
    

  undistorted = gd::warpImage(original, cornerPoints, maxLength);
   
  Mat undistortedThreshed = undistorted.clone();
  adaptiveThreshold(undistorted, undistortedThreshed, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 101, 1);
  ////////////////////////////////////////////////////////////////////////////////



  //CLASSIFY DIGITS/////////////////////////////////////////////////////////////

  //initialize and train digit recognizer
  DigitRecognizer *dr = new DigitRecognizer();
  bool b = dr->train("../images.xml", "../classifications.xml");

  if(!b)
  {
    cout<<"failed to train";
  }

  //isolate sudoku spaces from image
  int dist = ceil((double)maxLength/9);
  Mat currentSpace = Mat(dist, dist, CV_8UC1);
  

  vector<vector<int>> puzzle(9, vector<int>(9, 0));

  for(int j=0; j<9; j++)
  {
    for(int i=0; i<9; i++)
    {
      for(int y=0; y<dist && j*dist+y<undistortedThreshed.cols; y++)
      {
        uchar* ptr = currentSpace.ptr(y);

        for(int x=0; x<dist && i*dist+x<undistortedThreshed.rows; x++)
        {
          ptr[x] = undistortedThreshed.at<uchar>(j*dist + y, i*dist + x);        
        }
      }

      //classify sudoku space
      int value = dr->classify(currentSpace);
      puzzle[i][j] = value;
    }

  }
  ////////////////////////////////////////////////////////////////////////////////



  //SOLVE PUZZLE///////////////////////////////////////////////////////////////// 
  bool solved  = ss::solve(puzzle);

  if(!solved)
  {
    printf("The puzzle could not be solved.  This is what was read in:");
  }
  else
  {
    printf("The complete solution:");
  }

  printf("\n");
  cout << solved << endl;
  for(int j=0; j<9; j++)
  {
    for(int i=0; i<9; i++)
    {
      printf("%d", puzzle[i][j]);
    }
    printf("\n");
  }
  
  //////////////////////////////////////////////////////////////////////////////

 	namedWindow("Display window", WINDOW_AUTOSIZE);
  imshow("Display window", original);


	waitKey(0);
	return 0;


}


