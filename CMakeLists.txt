cmake_minimum_required(VERSION 3.5.1)

project( sudoku )
find_package( OpenCV REQUIRED)

include_directories(include)
set(SOURCES src/sudoku.cpp src/digitrecognizer.cpp )

add_executable(sudoku ${SOURCES})

target_link_libraries( sudoku ${OpenCV_LIBS})
