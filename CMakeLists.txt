############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files
cmake_minimum_required(VERSION 3.11)
project(ai LANGUAGES CXX CUDA)

find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)


include_directories( ${OpenCV_INCLUDE_DIRS} 
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
include
)
include_directories()
enable_language(CUDA)
link_libraries( ${OpenCV_LIBS} )



add_executable(main 
    src/HW3.cu
    src/main.cpp
    src/student_func.cu
    src/compare.cpp
    src/reference_calc.cpp
    src/loadSaveImage.cpp
)

set_property(TARGET main PROPERTY CUDA_ARCHITECTURES "35;50;72")

# CUDA_ADD_EXECUTABLE(HW1 ${HW1_files} ${hdr} ${cu})
