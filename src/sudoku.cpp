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

	//Set a threshold to differentiate between background and lines
	//Adaptive threshold used to achieve more accurate threshold by setting threshold as mean level in 5x5 pixel window
	adaptiveThreshold(sudoku, outerBox, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);
	//invert the threshold image to set areas of interest to white
	bitwise_not(outerBox, outerBox);

	//Dilate the image to smooth any discontinuities on lines caused by thresholding operation
	Mat kernel = (Mat_<uchar>(3,3) <<0,1,0,1,1,1,0,1,0);
	dilate(outerBox, outerBox, kernel);
	/////////////////////////////////////////////////////////////////////

	//FIND LARGEST BLOB////////////////////////////////////////////////////
  outerBox = gd::findLargestBlob(outerBox);

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
	/////////////////////////////////////////////////////////////////////////////
  vector<Point> points = gd::findCornerPoints(outerBox, lines);
  Point ptTopLeft = points[0];
  Point ptTopRight = points[1];
  Point ptBottomRight = points[2]; 
  Point ptBottomLeft = points[3];

  int maxLength = (ptBottomLeft.x-ptBottomRight.x)*(ptBottomLeft.x-ptBottomRight.x) + (ptBottomLeft.y-ptBottomRight.y)*(ptBottomLeft.y-ptBottomRight.y);
  int temp = (ptTopRight.x-ptBottomRight.x)*(ptTopRight.x-ptBottomRight.x) + (ptTopRight.y-ptBottomRight.y)*(ptTopRight.y-ptBottomRight.y);
  
  if(temp>maxLength)
  {
    maxLength = temp;
  }

  temp = (ptTopRight.x-ptTopLeft.x)*(ptTopRight.x-ptTopLeft.x)+(ptTopRight.y-ptTopLeft.y)*(ptTopRight.y-ptTopLeft.y);

	if(temp>maxLength)
    {
    	maxLength = temp;
    }

    temp = (ptBottomLeft.x-ptTopLeft.x)*(ptBottomLeft.x-ptTopLeft.x) + (ptBottomLeft.y-ptTopLeft.y)*(ptBottomLeft.y-ptTopLeft.y);

    if(temp>maxLength)
    {
    	maxLength = temp;
    }

    maxLength = sqrt((double)maxLength);

    Mat undistorted = Mat(Size(maxLength, maxLength), CV_8UC1);
    

    undistorted = gd::warpImage(original, points, maxLength);
   	namedWindow("Display window", WINDOW_AUTOSIZE);
	  imshow("Display window", undistorted);

   
    Mat undistortedThreshed = undistorted.clone();
    adaptiveThreshold(undistorted, undistortedThreshed, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 101, 1);

//initialize digit recognizer here
    DigitRecognizer *dr = new DigitRecognizer();
    bool b = dr->train("../images.xml", "../classifications.xml");

    if(!b)
    {
      cout<<"failed to train";
    }

  int dist = ceil((double)maxLength/9);
  Mat currentCell = Mat(dist, dist, CV_8UC1);
  

  vector<vector<int>> puzzle(9, vector<int>(9, 0));

  for(int j=0; j<9; j++)
  {
    for(int i=0; i<9; i++)
    {
      for(int y=0; y<dist && j*dist+y<undistortedThreshed.cols; y++)
      {
        uchar* ptr = currentCell.ptr(y);

        for(int x=0; x<dist && i*dist+x<undistortedThreshed.rows; x++)
        {
          ptr[x] = undistortedThreshed.at<uchar>(j*dist + y, i*dist + x);        
        }
      }


      int value = dr->classify(currentCell);
      puzzle[i][j] = value;//  printf("%d", value);
    }

   // printf("\n");
  }

	/////////////////////////////////////////////////////////////////////////////
//  vector<int> puzzle = {1, 3, 4, 5};
//  vector<vector<int>> puzzle = {{1, 2, 4, 3}, {4, 3, 2, 1}, {2, 1, 3, 4}, {3, 4, 1, 2}};

//  Solver *ss = new Solver();

//  bool test = ss->violation(puzzle, 6, 0, 2);
  
  bool solved  = ss::solve(puzzle);
  cout << solved << endl;
  for(int j=0; j<9; j++)
  {
    for(int i=0; i<9; i++)
    {
      printf("%d", puzzle[i][j]);
    }
    printf("\n");
  }

	waitKey(0);
	return 0;


}


