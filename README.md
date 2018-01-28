# Computer Vision Sudoku Solver
I wanted the ability to solve a sudoku puzzle just by taking a picture of it.  This project accomplishes that goal by using computer vision techniques to extract the puzzle from the image, and then using an algorithm to find th solution to the puzzle.

## How it works
Here's a rundown of how the program identifies the puzzle, extract the digits, and eventually solves the puzzle.
1. The program starts with a image of a sudoku puzzle.

<img src="https://github.com/davidjevans/sudoku_solver/blob/master/explanation_images/original.png" width="400">

2. The image is slightly blurred to smooth out noise in the image.
<img src="https://github.com/davidjevans/sudoku_solver/blob/master/explanation_images/blurred.png" width="400">
3. The image is now put through an adaptive threshold to form a binary image (either white or black) of the puzzle.
<img src="https://github.com/davidjevans/sudoku_solver/blob/master/explanation_images/threshed.png" width="400">
4. A Hough tranform is used to identify the major lines in the puzzle.  From here, the grid can be extracted from the puzzle.
<img src="https://github.com/davidjevans/sudoku_solver/blob/master/explanation_images/grid.png" width="400">
5. With the grid extracted, we can use a contour finding library to extract the numbers out of the spaces.
<img src="https://github.com/davidjevans/sudoku_solver/blob/master/explanation_images/number.png" width="200">
6. To recognize the digits from the images, I used a k-Nearest Neighbors(kNN) algorithm.  If you've never seen kNN before check out this [explanation](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor)


7. With each digit classified, the puzzle is known.  From here, a recursive algorithm is used to solve the puzzle.  The result is the solved sudoku puzzle:
<img src="https://github.com/davidjevans/sudoku_solver/blob/master/explanation_images/solution.png" width="200">

## How to make it work for you
### Prequisites
### Deployment
### Assumptions about puzzle image

## Acknowledgements

