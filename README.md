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

## Executing

```sh
# Be in the build
cd build
# Copy this nice image
cp ../memorial.exr ./
# run for this
./main ../memorial.exr
```

## Development 
| file                     | desc                      |
| ------------------------ | :------------------------ |
| `src/compare.cpp`        | comparing 2 imgs          |
| `src/loadSaveImage.cpp`  | loading and saving images |
| `src/reference_calc.cpp` | Reference calculation     |
| `src/histogram.cu`    | Parallel calculation      |


## Steps

1. Load image
2. (*PARALELLIZABLE*) Covert to Luminance
3. Make bins