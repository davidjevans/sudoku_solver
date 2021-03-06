#include "griddetector.h"

namespace gd
{

int findMaxLength(std::vector<cv::Point> points){

  cv::Point ptTopLeft = points[0];
  cv::Point ptTopRight = points[1];
  cv::Point ptBottomRight = points[2];
  cv::Point ptBottomLeft = points[3];

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

    return maxLength;
}

/*
  findCornerPoints: finds the 4 corner points of the sudoku puzzle from the Hough transform lines
  
  params:
    outerBox- the image of the box outlining the sudoku grid
    lines- lines of the sudoku grid generated by the Hough transform

  return:
    points- vector of the 4 corner points 
*/
std::vector<cv::Point> findCornerPoints(cv::Mat &outerBox, std::vector<cv::Vec2f> lines)
{
  cv::Vec2f topEdge = cv::Vec2f(1000,1000); double topYIntercept = 100000, topXIntercept=0;
  cv::Vec2f bottomEdge = cv::Vec2f(-1000,-1000); double bottomYIntercept = 0, bottomXIntercept=0;
  cv::Vec2f leftEdge = cv::Vec2f(1000,1000); double leftXIntercept = 100000, leftYIntercept=0;
  cv::Vec2f rightEdge = cv::Vec2f(-1000,-1000); double rightXIntercept = 0, rightYIntercept=0;


  for(int i=0; i<lines.size();i++)
  {
    cv::Vec2f current = lines[i];

    float p = current[0];
    float theta = current[1];

    //if line is merged into other line, skip
    if(p==0 && theta==-100)
    {
      continue;
    }

    double xIntercept, yIntercept;

    //calculate x and y intercepts
    xIntercept = p/cos(theta);
    yIntercept = p/(cos(theta)*sin(theta));

    if(theta>CV_PI*80/180 && theta<CV_PI*100/180)
    {
      if(p<topEdge[0])
      {
        topEdge = current;

      }

      if(p>bottomEdge[0])
      {
        bottomEdge = current;
      }
    }
    else if(theta <CV_PI*10/180 || theta>CV_PI*170/180)
    {
      if(xIntercept>rightXIntercept)
      {
        rightEdge = current;
        rightXIntercept = xIntercept;
      }
      else if(xIntercept <= leftXIntercept)
      {
        leftEdge = current;
        leftXIntercept = xIntercept;
      }
    }
  }

//  gd::drawLine(topEdge, sudoku, CV_RGB(0,0,0));
//  gd::drawLine(bottomEdge, sudoku, CV_RGB(0,0,0));
//  gd::drawLine(leftEdge, sudoku, CV_RGB(0,0,0));
//  gd::drawLine(rightEdge, sudoku, CV_RGB(0,0,0));

  /////////////////////////////////////////////////////////////////////////////

  //CALCULATE INTERSECTIONS BETWEEN EDGES////////////////////////////////////



  cv::Point left1, left2, right1, right2, bottom1, bottom2, top1, top2;

  int height = outerBox.size().height;
  int width = outerBox.size().width;

  if(leftEdge[1]!=0)
  {
    left1.x = 0;
    left1.y = leftEdge[0]/sin(leftEdge[1]);

    left2.x = width;
    left2.y = -left2.x/tan(leftEdge[1]) + left1.y;

  }
  else
  {
    left1.y = 0;
    left1.x = leftEdge[0]/cos(leftEdge[1]);

    left2.y = height;
    left2.x = left1.x - height*tan(leftEdge[1]);
  }

  if(rightEdge[1]!=0)
  {
    right1.x = 0;
    right1.y = rightEdge[0]/sin(rightEdge[1]);

    right2.x = width;
    right2.y = -right2.x/tan(rightEdge[1]) + right1.y;

  }
  else
  {
    right1.y = 0;
    right1.x = rightEdge[0]/cos(rightEdge[1]);

    right2.y = height;
    right2.x = right1.x - height*tan(rightEdge[1]);
  }

  bottom1.x = 0;
  bottom1.y = bottomEdge[0]/sin(bottomEdge[1]);

  bottom2.x = width;
  bottom2.y = -bottom2.x/tan(bottomEdge[1]) + bottom1.y;

  top1.x = 0;
  top1.y = topEdge[0]/sin(topEdge[1]);

  top2.x = width;
  top2.y = -top2.x/tan(topEdge[1]) + top1.y;

 // Next, we find the intersection of  these four lines

  double leftA = left2.y - left1.y;
  double leftB = left1.x - left2.x;
  double leftC = leftA*left1.x + leftB*left1.y;

  double rightA = right2.y - right1.y;
  double rightB = right1.x - right2.x;
  double rightC = rightA*right1.x +rightB*right1.y;

  double topA = top2.y - top1.y;
  double topB = top1.x - top2.x;
  double topC = topA*top1.x + topB*top1.y;

  double bottomA = bottom2.y - bottom1.y;
  double bottomB = bottom1.x - bottom2.x;
  double bottomC = bottomA*bottom1.x + bottomB*bottom1.y;

  //Intersection of top and left
  double detTopLeft = leftA*topB - leftB*topA;

  CvPoint ptTopLeft = cvPoint((topB*leftC - leftB*topC)/detTopLeft, (leftA*topC - topA*leftC)/detTopLeft);

  // Intersection of top and right
  double detTopRight = rightA*topB - rightB*topA;

  CvPoint ptTopRight = cvPoint((topB*rightC-rightB*topC)/detTopRight, (rightA*topC-topA*rightC)/detTopRight);

  // Intersection of right and bottom
  double detBottomRight = rightA*bottomB - rightB*bottomA;
  CvPoint ptBottomRight = cvPoint((bottomB*rightC-rightB*bottomC)/detBottomRight, (rightA*bottomC-bottomA*rightC)/detBottomRight);// Intersection of bottom and left
  
  //Intersection of left and bottom
  double detBottomLeft = leftA*bottomB-leftB*bottomA;
  CvPoint ptBottomLeft = cvPoint((bottomB*leftC-leftB*bottomC)/detBottomLeft, (leftA*bottomC-bottomA*leftC)/detBottomLeft);

 
  std::vector<cv::Point> points;
  points.push_back(ptTopLeft);
  points.push_back(ptTopRight);
  points.push_back(ptBottomRight);
  points.push_back(ptBottomLeft);

 return points;
}

/*
  warpImage: warps the original image to make its 4 corners the 4 corners of the new image

  params:
    img- the original image to be warped
    cornerPoints- the 4 corner points of the puzzle
    maxLength- the longest side of the puzzle
*/
  cv::Mat warpImage(cv::Mat &img, std::vector<cv::Point> cornerPoints, int maxLength)
  {
    cv::Point2f src[4], dst[4];

    src[0] = cornerPoints[0]; dst[0] = cv::Point2f(0,0);
    src[1] = cornerPoints[1]; dst[1] = cv::Point2f(maxLength-1,0);
    src[2] = cornerPoints[2]; dst[2] = cv::Point2f(maxLength-1, maxLength-1);
    src[3] = cornerPoints[3]; dst[3] = cv::Point2f(0, maxLength-1);

    cv::Mat unwarped = cv::Mat(cv::Size(maxLength, maxLength), CV_8UC1);
    cv::warpPerspective(img, unwarped, cv::getPerspectiveTransform(src, dst), cv::Size(maxLength, maxLength));
    
    return unwarped;
  }


  /*
    findLargestBlob: find the largest area that represents the sudoku puzzle. Do this by floodfilling any contours white and the center of the spaces as black. The final result is a 9x9 grid on a black background.

    params:
      outerBox- is the image to find the blob in

    returns:
      outerBox- the image of the 9x9 white grid on a black background
  */
  cv::Mat findLargestBlob(cv::Mat outerBox)
  {
    int count = 0;
    int max = -1;

    cv::Point maxPt;

    for(int y = 0; y<outerBox.size().height;y++)
    {
      uchar *row = outerBox.ptr(y);
      for(int x = 0; x<outerBox.size().width;x++)
      {
        if(row[x]>=128)
        {
          int area = cv::floodFill(outerBox, cv::Point(x,y), CV_RGB(0,0,64));

          if(area>max)
          {
            maxPt = cv::Point(x,y);
            max = area;
          }
        }

      }

    }

    cv::floodFill(outerBox, maxPt, CV_RGB(255, 255, 255));

    for(int y = 0; y<outerBox.size().height;y++)
    {
      uchar *row = outerBox.ptr(y);
      for(int x = 0; x<outerBox.size().width; x++)
      {
        if(row[x] ==64 && x!=maxPt.x && y!=maxPt.y)
        {
          int area = cv::floodFill(outerBox, cv::Point(x,y), CV_RGB(0,0,0));
        }
      }
    }

    return outerBox;
  }

  


/*
  mergeRelatedLines: The Hough transform returns a lot of possible lines.  We want to reduce the number of lines by averaging similar lines together.

  Params:
    lines- a pointer to a vector of lines the Hough tranform created
    img- the address of the image the lines come from, necessary for knowing size and width of area

  Returns:
    void, but the lines vector will now have merged lines
*/
void mergeRelatedLines(std::vector<cv::Vec2f> *lines, cv::Mat &img)
{
  std::vector<cv::Vec2f>::iterator current;

  for(current=lines->begin();current!=lines->end();current++)
  {
    //if line has already been merged, skip it
    if((*current)[0]==0 && (*current)[1]==-100) continue;

    float p1 = (*current)[0];
    float theta1 = (*current)[1];

    cv::Point pt1current, pt2current;
    if(theta1>CV_PI*45/180 && theta1<CV_PI*135/180)
    {
      pt1current.x = 0;
      pt1current.y = p1/sin(theta1);

      pt2current.x = img.size().width;
      pt2current.y = -pt2current.x/tan(theta1) + p1/sin(theta1);
    }
    else
    {
      pt1current.y = 0;
      pt1current.x = p1/cos(theta1);
      
      pt2current.y = img.size().height;
      pt2current.x = -pt2current.y/tan(theta1) + p1/cos(theta1);
    }
    
    std::vector<cv::Vec2f>::iterator pos;
    for(pos=lines->begin();pos!=lines->end();pos++)
    {
      //if they are the same line, skip it
      if(*current == *pos) continue;

      if(fabs((*pos)[0]-(*current)[0])<20 && fabs((*pos)[1]-(*current)[1])<CV_PI*10/180)
      {
        float p =(*pos)[0];
        float theta = (*pos)[1];

        cv::Point pt1, pt2;
        if((*pos)[1]>CV_PI*45/180 && (*pos)[1]<CV_PI*135/180)
        {
          pt1.x = 0;
          pt1.y = p/sin(theta);

          pt2.x = img.size().width;
          pt2.y = -pt2.x/tan(theta) + p/sin(theta);
        }
        else
        {
          pt1.y = 0;
          pt1.x = p/cos(theta);

          pt2.y = img.size().height;
          pt2.x = -pt2.y/tan(theta) + p/cos(theta);
        }

        if(((double)(pt1.x-pt1current.x)*(pt1.x-pt1current.x) + (pt1.y-pt1current.y)*(pt1.y-pt1current.y)<64*64) &&
            ((double)(pt2.x-pt2current.x)*(pt2.x-pt2current.x) + (pt2.y-pt2current.y)*(pt2.y-pt2current.y)<64*64))
        {
          //merge the two lines
          (*current)[0] = ((*current)[0]+(*pos)[0])/2;
          (*current)[1] = ((*current)[1]+(*pos)[1])/2;

          (*pos)[0] = 0;
          (*pos)[1] = -100;

        }

      }
    }
  }

}

  /*
    drawLine(Vec2f line. Mat &img, Scalar rgb)
    Draws a line onto a given image with a given color

    Params:
      line- line to be drawn given in normal form
      img- img line is being drawn on
      rgb- color of line

    Return:
      void, but draws the lines on the given image
  */
  void drawLine(cv::Vec2f line, cv::Mat &img, cv::Scalar rgb = CV_RGB(0,0,255))
  {
    //If the line is not vertical, find the slope and y intercept and drawn the line
    //If the line is vertical, draw a vertical line
    if(line[1]!=0)
    {
      float m = -1/tan(line[1]);
      float c = line[0]/sin(line[1]);

      cv::line(img, cv::Point(0,c), cv::Point(img.size().width, m*img.size().width+c), rgb);
    }
    else
    {
      cv::line(img, cv::Point(line[0], 0), cv::Point(line[0], img.size().height),rgb);
    }
  }
}
