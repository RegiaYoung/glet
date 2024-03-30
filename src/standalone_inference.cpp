#include <torch/script.h> // One-stop header.
#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <memory>
#include <sys/time.h>
#include <pthread.h>
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <queue>
#include <condition_variable>
#include <cuda_profiler_api.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "json/json.h"

#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include "socket.h"
#include "torchutils.h"
#include "common_utils.h" //printTimeStamp moved to here
#include <torch/csrc/jit/runtime/graph_executor.h>
#define IMAGENET_ROW 224
#define IMAGENET_COL 224

namespace po = boost::program_options; 
using namespace cv;

// logging related variables
uint64_t *arrStartTime;
uint64_t *arrEndTime;

bool WARMUP = true; // flag for indicating to do warmup

double gMean;
int gNumRequests;
int gBatchSize;
int gSocketFD;
std::string gTask;
std::string gTaskFile;

void computeRequest();

typedef struct _InputInfo {
   std::vector<std::vector<int>*> InputDims;
   std::vector<std::string> InputTypes;
} InputInfo;

std::map<std::string,InputInfo*> gNametoInputInfo;

int readInputJSONFile(const char* input_config_file, std::map<std::string, InputInfo*> &mapping){
#ifdef DEBUG
    printf("Reading App JSON File: %s \n", input_config_file);
#endif
    Json::Value root;
    std::ifstream ifs;
    ifs.open(input_config_file);

    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    if (!parseFromStream(builder, ifs, &root, &errs)) {
        std::cout << errs << std::endl;
        ifs.close();
        return EXIT_FAILURE;
    }
    for(unsigned int i=0; i < root["ModelInfoSpecs"].size(); i++){
        std::string model_name = root["ModelInfoSpecs"][i]["ModelName"].asString();
        mapping[model_name]=new InputInfo();
        for(unsigned int j=0; j< root["ModelInfoSpecs"][i]["Inputs"].size(); j++){
            mapping[model_name]->InputDims.push_back(new std::vector<int>());
            for(unsigned int k=0; k<root["ModelInfoSpecs"][i]["Inputs"][j]["InputDim"].size(); k++){
                mapping[model_name]->InputDims[j]->push_back(root["ModelInfoSpecs"][i]["Inputs"][j]["InputDim"][k].asInt());
            }
            mapping[model_name]->InputTypes.push_back(root["ModelInfoSpecs"][i]["Inputs"][j]["InputType"].asString());
        }
    }
    ifs.close();
    return EXIT_SUCCESS;
}

po::variables_map parse_opts(int ac, char** av) {
        po::options_description desc("Allowed options");
        desc.add_options()("help,h", "Produce help message")
                ("task,t", po::value<std::string>()->default_value("resnet50"), "name of model")
                ("taskfile",po::value<std::string>()->default_value("resnet50.pt"), "dir/to/model.pt")
                ("batch,b", po::value<int>()->default_value(1),"size of batch to send") 
                ("requests,r",po::value<int>()->default_value(1),"how many requests are going to be issued to the server" ) 
                ("mean,m,",po::value<double>()->default_value(0.3),"how long is the average time between each request(in seconds)")
                ("input,i",po::value<std::string>()->default_value("input.txt"),"txt file that contains list of inputs")
                ("input_config_json", po::value<std::string>()->default_value("input_config.json"), "json file for input dimensions");
        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm); 
        po::notify(vm); 
        if (vm.count("help")) {
                std::cout << desc << "\n"; exit(1);   
        } 
        return vm;
}

torch::Tensor getRandomNLPInput(std::vector<int> &input_dims, std::string &input_type){
        std::vector<int64_t> dims;
        // read input dimensions
        dims.push_back(gBatchSize);
        int cnt=1;  // skip the first element
        for(auto dim : input_dims){
                if(cnt  !=0){
                        cnt = cnt -1;
                        continue;
                }
                dims.push_back(dim);
        }
        torch::TensorOptions options;
        if (input_type == "INT64")
                options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA,0).requires_grad(false);
        else if(input_type == "FP32")
                options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA,0).requires_grad(false);
        else{
             printf("unsupported type: %s \n", input_type.c_str());
             exit(1);   
        }
        torch::Tensor input=torch::ones(dims,options);
        return input;
}

torch::Tensor getRandomImgInput(std::vector<int> &input_dims, std::string &input_type, int batch_size){
        std::vector<int64_t> dims;
        // read input dimensions
        dims.push_back(batch_size);
        for(auto dim : input_dims)
                dims.push_back(dim);
        torch::TensorOptions options;
        if (input_type == "INT64")
         options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA,0).requires_grad(false);
       else if(input_type == "FP32")
          options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA,0).requires_grad(false);
        else{
             printf("unsupported type: %s \n", input_type.c_str());
             exit(1);   
        }
        torch::Tensor input=torch::ones(dims,options);
        return input;
}
void getInputs(const char* netname, std::vector<torch::jit::IValue> &inputs, int batch_size){
        torch::Tensor input;
        torch::Device gpu_dev(torch::kCUDA,0);
#ifdef DEBUG
        printf("get input for %s \n", netname);
#endif 
        std::string STR_BERT = "bert";
        std::string STR_GPT2 = "gpt2";
        std::string str_name = std::string(netname);        
        // assume this model is for profiling random model
        if(gNametoInputInfo.find(str_name) == gNametoInputInfo.end()){
                std::string TYPE = "FP32";
                std::vector<int> DIMS = {3,224,224};
                input = getRandomImgInput(DIMS,TYPE,batch_size);
        }
        else{
                for(unsigned int i=0; i < gNametoInputInfo[str_name]->InputDims.size(); i++){
                if(str_name.find(STR_BERT) != std::string::npos || str_name.find(STR_GPT2) != std::string::npos)
                        input = getRandomNLPInput(*(gNametoInputInfo[str_name]->InputDims[i]), gNametoInputInfo[str_name]->InputTypes[i]);
                else
                        input = getRandomImgInput(*(gNametoInputInfo[str_name]->InputDims[i]), gNametoInputInfo[str_name]->InputTypes[i], batch_size);
                }
        }
        input = input.to(gpu_dev);
        inputs.push_back(input);
        return;

}
void setupGlobalVars(po::variables_map &vm){
        gTask = vm["task"].as<std::string>();
        gMean = vm["mean"].as<double>();
        gNumRequests=vm["requests"].as<int>();
        gBatchSize= vm["batch"].as<int>();
        assert(gBatchSize!=0);
        gTaskFile = vm["taskfile"].as<std::string>();
        if(readInputJSONFile(vm["input_config_json"].as<std::string>().c_str(), gNametoInputInfo))
        {
            printf("Failed reading json file: %s \n", vm["input_config_json"].as<std::string>().c_str());
            exit(1);
        }
        return;
}

void PyTorchInit(){
    uint64_t total_end, total_start;
    std::vector<torch::jit::IValue> inputs;
    std::vector<int64_t> sizes={1};
    torch::TensorOptions options;
    options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA,0).requires_grad(false);
    total_start = getCurNs();
    torch::Tensor dummy1 = at::empty(sizes,options);
    torch::Tensor dummy2 = at::empty(sizes,options);
    torch::Tensor dummy3 = dummy1 + dummy2;
    cudaDeviceSynchronize();
    total_end = getCurNs();
    std::cout << double(total_end - total_start)/1000000 << " PyTorchInit total ms "<<std::endl;
    return;
}

int main(int argc, char** argv) {
        torch::jit::getBailoutDepth() = 0;
        torch::jit::getProfilingMode() = false;
        // torch::jit::setGraphExecutorOptimize(false);
        // torch::jit::FusionStrategy static0 = { {torch::jit::FusionBehavior::STATIC, 0} }; 
        // torch::jit::setFusionStrategy(static0); 
        /*get parameters for this program*/
        po::variables_map vm = parse_opts(argc, argv);
        setupGlobalVars(vm);
        printTimeStamp("START PROGRAM");
        computeRequest();   
        printTimeStamp("END PROGRAM");
        return 0;
}

void computeRequest(){
#ifdef DEBUG
        std::cout<<"started copmuting thread"<<std::endl;
#endif
        cpu_set_t cpuset;
        torch::Tensor input;
        std::vector<torch::jit::IValue> inputs;
        torch::Device gpu_dev(torch::kCUDA,0);
        uint64_t total_end, total_start;
        const char *netname = gTask.c_str();
        int i;
        PyTorchInit();
        std::cout<< "waiting for 3 seconds after PyTorchInit" << std::endl;
        usleep(3*1000*1000);
        c10::InferenceMode guard;
        uint64_t t1,t2,t3,t4;
        t1 = getCurNs();
        std::shared_ptr<torch::jit::script::Module> module = std::make_shared<torch::jit::script::Module>(torch::jit::load(gTaskFile.c_str(),gpu_dev));
        t2 = getCurNs();
        module->to(gpu_dev);
        module->eval();
        cudaDeviceSynchronize();
        t3= getCurNs();       
        if(WARMUP){
                for(int batch_size = 32; batch_size >=1; batch_size--){
                        getInputs(netname, inputs,batch_size);
                        module->forward(inputs);
                        cudaDeviceSynchronize();
                        inputs.clear();
                }
        }
        t4 = getCurNs();
        if(WARMUP){
                std::cout<< "waiting for 3 seconds after warmup" << std::endl;
                usleep(3*1000*1000);
        }

        std::cout << "main jit-load: " << double(t2-t1)/1000000 <<\
                "warmup: " << double(t3-t4)/1000000 << \
                std::endl;
          
        for (int i =0; i < gNumRequests; i++){
                usleep(gMean * 1000* 1000);
                getInputs(netname,inputs,gBatchSize);
                printTimeStampWithName(netname, "START EXEC");
#ifdef DEBUG
                if(i==0) total_start=getCurNs();
#endif
                uint64_t start,end;
                uint64_t start_check= getCurNs();
                cudaProfilerStart();
                start = getCurNs();
                torch::IValue output = module->forward(inputs);
                if(output.isTuple()){
                        torch::Tensor t = output.toTuple()->elements()[0].toTensor(); // 1st output;
                        t = t.to(torch::Device(torch::kCPU));
                }
                else torch::Tensor t = output.toTensor().to(torch::Device(torch::kCPU));
                cudaDeviceSynchronize();
                end=getCurNs(); 
                cudaProfilerStop();
                printTimeStampWithName(netname, "END EXEC");
                printf("latency: %lf\n", double(end-start)/1000000);

#ifdef DEBUG
                if(i==gNumRequests-1) total_end = getCurNs();
                uint64_t now=getCurNs();
                printf("throughput: %lf\n", (i+1)*(gBatchSize)/(double(now-total_start)/1000000000) );
#endif 
                inputs.clear();
        }
#ifdef DEBUG
        printf("total latency: %lf \n", double(total_end-total_start)/1000000);
        printf("total throughput: %lf\n", gNumRequests*(gBatchSize)/(double(total_end-total_start)/1000000000) );
#endif
        
}
