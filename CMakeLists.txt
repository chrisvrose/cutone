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


include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(/opt/cuda/include)
enable_language(CUDA)
link_libraries( ${OpenCV_LIBS} )



add_executable(main HW3.cu main.cpp student_func.cu compare.cpp reference_calc.cpp loadSaveImage.cpp )

set_property(TARGET main PROPERTY CUDA_ARCHITECTURES "35;50;72")

# CUDA_ADD_EXECUTABLE(HW1 ${HW1_files} ${hdr} ${cu})
