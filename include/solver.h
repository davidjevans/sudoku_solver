#include <stdio.h>
#include <vector>
#include <iostream>

class Solver
{
  public:
    Solver();
    ~Solver();
    
    bool solve(std::vector<std::vector<int>> &puzzle);

//  private:    
    bool violation(std::vector<std::vector<int>>, int newValue, int i, int j);
    bool complete(std::vector<std::vector<int>> puzzle);
    void printPuzzle(std::vector<std::vector<int>> puzzle);
    int depth;
//  private:
};


