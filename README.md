# cutone

Use CMAKE to build the solution


## Building

```sh
# Create and move into a build directory
mkdir -p build
cd build

# Ask cmake to configure the project that's just outside the build directory
cmake ..
# Cmake creates a nice makefile in the build directory, make it
make
```



| file                     | desc                      |
| ------------------------ | :------------------------ |
| `src/compare.cpp`        | comparing 2 imgs          |
| `src/loadSaveImage.cpp`  | loading and saving images |
| `src/reference_calc.cpp` | Reference calculation     |
| `src/student_func.cu`    | Parallel calculation      |


