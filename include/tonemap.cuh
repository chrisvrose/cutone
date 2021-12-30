void preProcess(float **d_luminance, unsigned int **d_cdf,
                size_t *numRows, size_t *numCols, unsigned int *numBins,
                const std::string& filename);

void postProcess(const std::string& output_file, size_t numRows, size_t numCols,
                 float min_logLum, float max_logLum);

void cleanupGlobalMemory(void);