#include <stdio.h>
#include <vector>
#include <iostream>

namespace ss
{
  bool solve(std::vector<std::vector<int>> &puzzle);
  bool violation(std::vector<std::vector<int>>, int newValue, int i, int j);
  bool complete(std::vector<std::vector<int>> puzzle);
  void printPuzzle(std::vector<std::vector<int>> puzzle);
}
