#ifndef __INPUT_
#define __INPUT_
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>
#include "router.h"


using namespace std;
#define IMG_POOL 592 //the maximum number of image you can read

torch::Tensor getRandLatVec(int batch_size);
torch::Tensor getRandImgTensor();
int readImgData(const char *path_to_txt, int num_of_img);
int configAppSpec(const char* ConfigJSON,SysMonitor &SysState, string res_dir);
int readAppJSONFile(const char* AppJSON, AppSpec &App, string res_dir);
// used in backend and proxy, provide proper configJSON and InputDimMapping, mapFiletoID will be filled 
#else
#endif 
