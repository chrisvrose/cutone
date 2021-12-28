#include <stdio.h>
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int v = prop.major * 10 + prop.minor;
    printf("-gencode arch=compute_%d,code=sm_%d\n", v, v);
}