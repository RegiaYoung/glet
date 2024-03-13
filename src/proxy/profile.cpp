#include <stdio.h>
#include <iostream>
#include <assert.h>
#include "profile.h"

using namespace std;

ReqProfile::ReqProfile(int req_id, int job_id){
    _id = req_id;
    _jid=job_id;
}

ReqProfile::~ReqProfile(){
}

void ReqProfile::setInputStart(uint64_t time){
    input_start=time;
}

void ReqProfile::setInputEnd(uint64_t time){
    input_end = time;
}

void ReqProfile::setCPUCompStart(uint64_t time){
    cpu_comp_start = time;
}

void ReqProfile::setGPUStart(uint64_t time){
    //added for debugging this functiion
    #ifdef PROFILE
    std::cout << __func__ << ": called for req id: " << _id << std::endl;
    #endif
    gpu_start = time;
}

void ReqProfile::setGPUEnd(uint64_t time){
    #ifdef PROFILE
    std::cout << __func__ << ": called for req id: " << _id << std::endl;
    #endif
    gpu_end = time;
}

void ReqProfile::setCPUPostEnd(uint64_t time){
    cpu_post_end=time;
}

void ReqProfile::setOutputStart(uint64_t time){
    output_start=time;
}

void ReqProfile::setOutputEnd(uint64_t time){
    output_end = time;
}

void ReqProfile::setBatchSize(int time){
    _batch_size = time;
} 

void ReqProfile::printTimes(){
    printf("[PROFILE] printing execution time info of req_id: %d \n",_id);
    assert(input_start);
    assert(gpu_end);
    assert(gpu_start);
    assert(output_end);
    double total = double(output_end - input_start)/1000000;
    double input_cpu = double(input_end - input_start)/1000000;
    double input_delay = double(cpu_comp_start - input_end)/1000000;
    double preprocess = double(gpu_start - cpu_comp_start)/1000000;
    double total_gpu = double(gpu_end - gpu_start)/1000000;
    double postprocess = double(cpu_post_end - gpu_end)/1000000;
    double output_delay=double(output_start - cpu_post_end)/1000000;
    double output_cpu = double(output_end - output_start)/1000000;
    double total_cpu = total-total_gpu;
    printf("[PROFILE] %d id: %d batch_size: %d input_cpu: %lf input_delay: %lf preprocess: %lf total_gpu: %lf postprocess: %lf output_delay: %lf output_cpu: %lf total_cpu: %lf \n" ,\
                    _jid,_id,_batch_size, \
                    input_cpu,\
                    input_delay,\
                    preprocess,\
                    total_gpu,\
                    postprocess,\
                    output_delay,\
                    output_cpu,\
                    total_cpu);
}
