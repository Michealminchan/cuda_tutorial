# GPU Teaching Kit -- Accelerated Computing Labs

## Software Requirements

_Caution: **You must have an [NVIDIA CUDA Capable GPU](https://developer.nvidia.com/cuda-gpus)
to use the compiled binaries.**_

The labs in the teaching kit require a CUDA supported operating system,
C compiler, and a recent CUDA Toolkit. The CUDA Toolkit can be downloaded
from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page.
Instructions on how to install the CUDA Toolkit are available in the
[Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).
Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html),
[Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and
[OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are
also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

Aside from a C compiler and the CUDA Toolkit, [CMake](https://cmake.org/) 2.8 or later is required
to generate build scripts for your target IDE and compiler. The next section describes
the process of compiling and running a lab.

## Compiling and Running Labs

In this section we describe how to setup your machine to compile the labs.
First, regardless of the platform compiling the labs the
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and
[CMake](https://cmake.org/) must be installed.

Now, checkout the the GPU Teaching Kit -- Accelerated Computing Labs from the
[Bitbucket repository](https://bitbucket.org/hwuligans/gputeachingkit-labs)

Since, the project depends on an external [libwb](https://github.com/abduld/libwb) repository [![Build Status](https://travis-ci.org/abduld/libwb.svg?branch=master)](https://travis-ci.org/abduld/libwb)
 we must perform a recursive clone (to also checkout the `libwb` repository).

~~~{.bash}
git clone --recursive git@bitbucket.org:hwuligans/gputeachingkit-labs.git
~~~

In the next section we will show how to compile and run the labs on Linux, OSX,
and Windows.

### Linux and Mac OSX

We will show how to compile the labs on both Linux and OSX using Makefiles.
First, create the target build directory

~~~
mkdir build-dir
cd build-dir
~~~

We will use `ccmake`

~~~
ccmake /path/to/gpu-kit-git-checkout
~~~

You will see the following screen

![ccmake](https://s3.amazonaws.com/gpuedx/resources/screenshots/Screenshot+2015-10-23+11.58.27.png)

Pressing `c` would configure the build to your system (in the process detecting
  the compiler, the CUDA Toolkit location, etc...).

![ccmake-config](https://s3.amazonaws.com/gpuedx/resources/screenshots/Screenshot+2015-10-23+12.03.26.png)

Note the options available to you, specifically:

~~~
BUILD_DESCRIPTION               *OFF
BUILD_GENERATOR                 *ON
BUILD_LIBWB_LIBRARY             *ON
BUILD_SOLUTION                  *ON
BUILD_TEMPLATE                  *OFF
~~~

* `BUILD_DESCRIPTION` -- option toggles whether to regenerate
`pdf` and `docx` lab output (this requires a python, latex, and pandoc installation)
* `BUILD_GENERATOR` -- option toggles whether to build the dataset
generator scripts as part of the build process
* `BUILD_LIBWB_LIBRARY` -- option toggles whether to build the `libwb` (the support library)
as part of the build process
* `BUILD_TEMPLATE` -- option toggles whether to build the code templates
as part of the build process (the templates are missing critical code that
makes them uncompilable).

Templates are meant to be used as starting
code for students whereas the solution is meant for instructor use.

If you have modified the above, then you should type `g` to regenerate the Makefile and then `q` to quit out of `ccmake`.
You can then use the `make` command to build the labs.

![make](https://s3.amazonaws.com/gpuedx/resources/screenshots/Screenshot+2015-10-23+12.11.15.png)

The `make` scripts builds the executables which can be run using the command template
provided in the lab's description. Here we run the `DeviceQuery` lab.

![device-query-osx](https://s3.amazonaws.com/gpuedx/resources/screenshots/Screenshot+2015-10-23+12.12.28.png)

### Windows

The usage of CMake on windows is the same as that on linux, except for windows we will using the GUI version (one can still use the command line version however).

First, launch the CMake gui application and set your source directory (the checkout directory) and the build directory (where you want the labs to be built).


![cmake-gui1](https://s3.amazonaws.com/gpuedx/resources/screenshots/1.PNG)

Clicking configure gives you an option to select which compiler to use to compile the labs. The compiler must be installed on the system and support by the CUDA toolkit

![compiler-select](https://s3.amazonaws.com/gpuedx/resources/screenshots/2.PNG)

The CMake system then searches the system and populates the proper options in your configuration. As a user you can override these options if needed

![compiler-options](https://s3.amazonaws.com/gpuedx/resources/screenshots/3.PNG)

Clicking `Generate` button, the CMake system creates the build scripts in the previously specified build directory. Since we selected Visual Studio, a Visual Studio solution is generated.

![vs-dir](https://s3.amazonaws.com/gpuedx/resources/screenshots/4.PNG)

Opening the Visual Studio solution, you can edit and compile all the labs  

![vs-view](https://s3.amazonaws.com/gpuedx/resources/screenshots/5.PNG)

The labs are built like any Visual Studio project using the build button

![vs-build](https://s3.amazonaws.com/gpuedx/resources/screenshots/6.PNG)

Once the lab is built, it can be run. Here we run the device query lab

![dev-query](https://s3.amazonaws.com/gpuedx/resources/screenshots/7.PNG)
