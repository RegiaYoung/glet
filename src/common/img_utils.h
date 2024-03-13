#ifndef OPENCVUTILS_H // To make sure you don't declare the function more than once by including the header multiple times.
#define OPENCVUTILS_H

#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <pthread.h>
#include <queue>
#include <condition_variable>
#include "torchutils.h"

torch::Tensor serialPreprocess(std::vector<cv::Mat> input, int imagenet_row, int imagenet_col);

#else
#endif
