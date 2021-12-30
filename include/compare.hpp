#ifndef tonemap_H__
#define tonemap_H__

void compareImages(std::string reference_filename, std::string test_filename, bool useEpsCheck,
				   double perPixelError, double globalError);

#endif
