# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/labs-source

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/build-dir

# Include any dependencies generated for this target.
include CMakeFiles/Stencil_DatasetGenerator.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Stencil_DatasetGenerator.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Stencil_DatasetGenerator.dir/flags.make

CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o: CMakeFiles/Stencil_DatasetGenerator.dir/flags.make
CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o: /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/labs-source/Module8/Stencil/dataset_generator.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/build-dir/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o -c /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/labs-source/Module8/Stencil/dataset_generator.cpp

CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/labs-source/Module8/Stencil/dataset_generator.cpp > CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.i

CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/labs-source/Module8/Stencil/dataset_generator.cpp -o CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.s

CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o.requires:
.PHONY : CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o.requires

CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o.provides: CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o.requires
	$(MAKE) -f CMakeFiles/Stencil_DatasetGenerator.dir/build.make CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o.provides.build
.PHONY : CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o.provides

CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o.provides.build: CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o

# Object files for target Stencil_DatasetGenerator
Stencil_DatasetGenerator_OBJECTS = \
"CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o"

# External object files for target Stencil_DatasetGenerator
Stencil_DatasetGenerator_EXTERNAL_OBJECTS =

Stencil_DatasetGenerator: CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o
Stencil_DatasetGenerator: libwb.a
Stencil_DatasetGenerator: /usr/local/cuda-8.0/lib64/libcudart.so
Stencil_DatasetGenerator: /usr/lib/x86_64-linux-gnu/libcuda.so
Stencil_DatasetGenerator: CMakeFiles/Stencil_DatasetGenerator.dir/build.make
Stencil_DatasetGenerator: CMakeFiles/Stencil_DatasetGenerator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable Stencil_DatasetGenerator"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Stencil_DatasetGenerator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Stencil_DatasetGenerator.dir/build: Stencil_DatasetGenerator
.PHONY : CMakeFiles/Stencil_DatasetGenerator.dir/build

CMakeFiles/Stencil_DatasetGenerator.dir/requires: CMakeFiles/Stencil_DatasetGenerator.dir/Module8/Stencil/dataset_generator.cpp.o.requires
.PHONY : CMakeFiles/Stencil_DatasetGenerator.dir/requires

CMakeFiles/Stencil_DatasetGenerator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Stencil_DatasetGenerator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Stencil_DatasetGenerator.dir/clean

CMakeFiles/Stencil_DatasetGenerator.dir/depend:
	cd /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/build-dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/labs-source /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/labs-source /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/build-dir /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/build-dir /home/lambda/work/hhb/lab/cuda_learn/Source_Code/labs-source/build-dir/CMakeFiles/Stencil_DatasetGenerator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Stencil_DatasetGenerator.dir/depend

