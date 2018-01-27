#include "solver.h"


namespace ss
{
/*
  solve: takes in an unsolved puzzle and attempts to recursively solve the puzzle.

  params:
    puzzle: the address of the unsolved puzzle

  return:
    true if the puzzle was solved
    false if the puzzle was not solved
*/
bool solve(std::vector<std::vector<int>> &puzzle)
{

  if(complete(puzzle))
  {
    return true;
  }

  //Iterate over puzzle
  for(int i = 0; i < 9; i++)
  {
    for(int j = 0; j < 9; j++)
    {
      //Find a space that is unassigned (value of zero)
      if(puzzle[j][i] == 0)
      {
        //Pick a value to try for the space
        for(int num = 1; num <=9; num++)
        {
          //Determine if that value violates any rules
          if(!violation(puzzle, num, j, i))
          {
            //Set the space to that value, and solve the new puzzle
            puzzle[j][i] = num;
        
            if(solve(puzzle))
            {
              return true;
            }

            //reset space if failed
            puzzle[j][i] = 0;
          }
        }
          
        return false;
      }
    }
  }
}

/*
  complete: checks to see if the given puzzle has any unsolved spaces left

  params: 
    puzzle: the puzzle to be checked
  return:
    true if puzzle is complete
    false if puzzle still has unsolved spaces
*/
bool complete(std::vector<std::vector<int>> puzzle)
{
  for(int i = 0; i < 9; i++)
  {
    for(int j = 0; j < 9; j++)
    {
      if(puzzle[i][j] == 0)
      {
        return false;
      }
    }
  }

  return true;
}

/*
  violation: given a puzzle, and a guess of a space, check to see if there are any violations
    according to the sudoku rules: horizontal, vertical, and box violations.

  params:
    puzzle- the puzzle that you are checking the violation in
    newValue- the value of the space that you are checking
    locX- the x location in the puzzle of the space you are checking
    locY- the y location in the puzzle of the space you are checking

  return:
    true if there is a violation
    false if there is no violation
*/
bool violation(std::vector<std::vector<int>> puzzle, int newValue, int locX, int locY)
{

  for(int i = 0; i < puzzle[0].size(); i++)
  {
    //Check if there are any horizontal violations
    if(i != locX)
    {
      if(newValue == puzzle[i][locY])
      {
        return true;
      } 
    }

    //Check if there are any vertical violations
    if(i != locY)
    {
      if(newValue == puzzle[locX][i])
      {
        return true;
      }
    }
    
  }

  //Check if there are any box violations
  for(int x = 3*(locX/3); x < 3*(locX/3) + 3; x++)
  {
    for(int y = 3*(locY/3); y < 3*(locY/3) + 3; y++)
    {
      if(x != locX && y != locY)
      {
        if(newValue == puzzle[x][y])
        {
          return true;
        }
      }
    }
  }

  return false;
}
}
