# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zeumer/Documents/fun/opencv/sudoku

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zeumer/Documents/fun/opencv/sudoku/build

# Include any dependencies generated for this target.
include CMakeFiles/sudoku.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sudoku.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sudoku.dir/flags.make

CMakeFiles/sudoku.dir/src/sudoku.cpp.o: CMakeFiles/sudoku.dir/flags.make
CMakeFiles/sudoku.dir/src/sudoku.cpp.o: ../src/sudoku.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zeumer/Documents/fun/opencv/sudoku/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sudoku.dir/src/sudoku.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sudoku.dir/src/sudoku.cpp.o -c /home/zeumer/Documents/fun/opencv/sudoku/src/sudoku.cpp

CMakeFiles/sudoku.dir/src/sudoku.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sudoku.dir/src/sudoku.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zeumer/Documents/fun/opencv/sudoku/src/sudoku.cpp > CMakeFiles/sudoku.dir/src/sudoku.cpp.i

CMakeFiles/sudoku.dir/src/sudoku.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sudoku.dir/src/sudoku.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zeumer/Documents/fun/opencv/sudoku/src/sudoku.cpp -o CMakeFiles/sudoku.dir/src/sudoku.cpp.s

CMakeFiles/sudoku.dir/src/sudoku.cpp.o.requires:

.PHONY : CMakeFiles/sudoku.dir/src/sudoku.cpp.o.requires

CMakeFiles/sudoku.dir/src/sudoku.cpp.o.provides: CMakeFiles/sudoku.dir/src/sudoku.cpp.o.requires
	$(MAKE) -f CMakeFiles/sudoku.dir/build.make CMakeFiles/sudoku.dir/src/sudoku.cpp.o.provides.build
.PHONY : CMakeFiles/sudoku.dir/src/sudoku.cpp.o.provides

CMakeFiles/sudoku.dir/src/sudoku.cpp.o.provides.build: CMakeFiles/sudoku.dir/src/sudoku.cpp.o


CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o: CMakeFiles/sudoku.dir/flags.make
CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o: ../src/digitrecognizer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zeumer/Documents/fun/opencv/sudoku/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o -c /home/zeumer/Documents/fun/opencv/sudoku/src/digitrecognizer.cpp

CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zeumer/Documents/fun/opencv/sudoku/src/digitrecognizer.cpp > CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.i

CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zeumer/Documents/fun/opencv/sudoku/src/digitrecognizer.cpp -o CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.s

CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o.requires:

.PHONY : CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o.requires

CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o.provides: CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o.requires
	$(MAKE) -f CMakeFiles/sudoku.dir/build.make CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o.provides.build
.PHONY : CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o.provides

CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o.provides.build: CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o


CMakeFiles/sudoku.dir/src/solver.cpp.o: CMakeFiles/sudoku.dir/flags.make
CMakeFiles/sudoku.dir/src/solver.cpp.o: ../src/solver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zeumer/Documents/fun/opencv/sudoku/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/sudoku.dir/src/solver.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sudoku.dir/src/solver.cpp.o -c /home/zeumer/Documents/fun/opencv/sudoku/src/solver.cpp

CMakeFiles/sudoku.dir/src/solver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sudoku.dir/src/solver.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zeumer/Documents/fun/opencv/sudoku/src/solver.cpp > CMakeFiles/sudoku.dir/src/solver.cpp.i

CMakeFiles/sudoku.dir/src/solver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sudoku.dir/src/solver.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zeumer/Documents/fun/opencv/sudoku/src/solver.cpp -o CMakeFiles/sudoku.dir/src/solver.cpp.s

CMakeFiles/sudoku.dir/src/solver.cpp.o.requires:

.PHONY : CMakeFiles/sudoku.dir/src/solver.cpp.o.requires

CMakeFiles/sudoku.dir/src/solver.cpp.o.provides: CMakeFiles/sudoku.dir/src/solver.cpp.o.requires
	$(MAKE) -f CMakeFiles/sudoku.dir/build.make CMakeFiles/sudoku.dir/src/solver.cpp.o.provides.build
.PHONY : CMakeFiles/sudoku.dir/src/solver.cpp.o.provides

CMakeFiles/sudoku.dir/src/solver.cpp.o.provides.build: CMakeFiles/sudoku.dir/src/solver.cpp.o


CMakeFiles/sudoku.dir/src/griddetector.cpp.o: CMakeFiles/sudoku.dir/flags.make
CMakeFiles/sudoku.dir/src/griddetector.cpp.o: ../src/griddetector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zeumer/Documents/fun/opencv/sudoku/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/sudoku.dir/src/griddetector.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sudoku.dir/src/griddetector.cpp.o -c /home/zeumer/Documents/fun/opencv/sudoku/src/griddetector.cpp

CMakeFiles/sudoku.dir/src/griddetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sudoku.dir/src/griddetector.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zeumer/Documents/fun/opencv/sudoku/src/griddetector.cpp > CMakeFiles/sudoku.dir/src/griddetector.cpp.i

CMakeFiles/sudoku.dir/src/griddetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sudoku.dir/src/griddetector.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zeumer/Documents/fun/opencv/sudoku/src/griddetector.cpp -o CMakeFiles/sudoku.dir/src/griddetector.cpp.s

CMakeFiles/sudoku.dir/src/griddetector.cpp.o.requires:

.PHONY : CMakeFiles/sudoku.dir/src/griddetector.cpp.o.requires

CMakeFiles/sudoku.dir/src/griddetector.cpp.o.provides: CMakeFiles/sudoku.dir/src/griddetector.cpp.o.requires
	$(MAKE) -f CMakeFiles/sudoku.dir/build.make CMakeFiles/sudoku.dir/src/griddetector.cpp.o.provides.build
.PHONY : CMakeFiles/sudoku.dir/src/griddetector.cpp.o.provides

CMakeFiles/sudoku.dir/src/griddetector.cpp.o.provides.build: CMakeFiles/sudoku.dir/src/griddetector.cpp.o


# Object files for target sudoku
sudoku_OBJECTS = \
"CMakeFiles/sudoku.dir/src/sudoku.cpp.o" \
"CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o" \
"CMakeFiles/sudoku.dir/src/solver.cpp.o" \
"CMakeFiles/sudoku.dir/src/griddetector.cpp.o"

# External object files for target sudoku
sudoku_EXTERNAL_OBJECTS =

sudoku: CMakeFiles/sudoku.dir/src/sudoku.cpp.o
sudoku: CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o
sudoku: CMakeFiles/sudoku.dir/src/solver.cpp.o
sudoku: CMakeFiles/sudoku.dir/src/griddetector.cpp.o
sudoku: CMakeFiles/sudoku.dir/build.make
sudoku: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_superres3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_face3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_plot3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_reg3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_text3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_shape3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_video3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_viz3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_flann3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_ml3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_photo3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.2.0
sudoku: /opt/ros/kinetic/lib/libopencv_core3.so.3.2.0
sudoku: CMakeFiles/sudoku.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zeumer/Documents/fun/opencv/sudoku/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable sudoku"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sudoku.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sudoku.dir/build: sudoku

.PHONY : CMakeFiles/sudoku.dir/build

CMakeFiles/sudoku.dir/requires: CMakeFiles/sudoku.dir/src/sudoku.cpp.o.requires
CMakeFiles/sudoku.dir/requires: CMakeFiles/sudoku.dir/src/digitrecognizer.cpp.o.requires
CMakeFiles/sudoku.dir/requires: CMakeFiles/sudoku.dir/src/solver.cpp.o.requires
CMakeFiles/sudoku.dir/requires: CMakeFiles/sudoku.dir/src/griddetector.cpp.o.requires

.PHONY : CMakeFiles/sudoku.dir/requires

CMakeFiles/sudoku.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sudoku.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sudoku.dir/clean

CMakeFiles/sudoku.dir/depend:
	cd /home/zeumer/Documents/fun/opencv/sudoku/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zeumer/Documents/fun/opencv/sudoku /home/zeumer/Documents/fun/opencv/sudoku /home/zeumer/Documents/fun/opencv/sudoku/build /home/zeumer/Documents/fun/opencv/sudoku/build /home/zeumer/Documents/fun/opencv/sudoku/build/CMakeFiles/sudoku.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sudoku.dir/depend

