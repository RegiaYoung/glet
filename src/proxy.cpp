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
#include <mutex>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "socket.h"
#include "json/json.h"
#include "torchutils.h"
#include "common_utils.h"
#include "custom_ops.h"
#include "config.h"
#include "profile.h"
#include "proxy_ctrl.h"
#include "shmem_ctrl.h"
#include "gpu_proxy.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <signal.h>
#include <execinfo.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#define INIT_MALLOC_SIZE 32*3*300*300
#define MAX_BATCH 32
using namespace std;

unordered_map<int,float*> per_jobid_input_mem;

namespace po=boost::program_options;

mutex gIqueueMtx;
condition_variable gIqueueCV;
queue<struct queue_elem*> gInputQueue;
mutex gOqueueMtx;
condition_variable gOqueueCV;
queue<struct queue_elem*> gOutputQueue;
mutex gCqueueMtx;
condition_variable gCqueueCV;
queue<struct queue_elem*> gControlQueue;
ProxyCtrl *gProxyCtrl;
proxy_info *gpPInfo;

// below are flags for controlling proxy 
bool exitFlag = false;
bool computing = false;
bool receiving = false;
bool readyFlag_input=false;
bool readyFlag_output=false;
bool readyFlag_compute=false;
bool waiting_input_conn=false;
bool waiting_output_conn=true;
bool input_closed=true;

enum action {LOAD=0, RUN=1, UNLOAD=2}; 

struct queue_elem{
    int reqid;
    int jobid;
    int batch_size;
    std::vector<int64_t> dims;
    float* float32_indata;
    int64_t* int64_indata;

    torch::Tensor outVal;
    action act;
    ReqProfile *pReqProf;
};



int gDevID;
int gThreadCap;
int gDedup;
std::string gCommonDir;
int gPartIdx;
torch::Device gpu_dev(torch::kCUDA,0);


const char *SEND="SEND";
const char *RECV="RECV";
const char *COMP="COMP";
const char *LISTENING="LISTENING";
const char *ACCEPTED="ACCEPTED";
const char *DONE="INIT DONE";
const char *START="INIT START";

vector<int> gLoadedModelIDs; 
map<std::string, int> gMapping_name_to_id;
std::unordered_map<std::string, int> gMapping_file_to_id;
map<int, shared_ptr<torch::jit::script::Module>> gModelTable;
std::unordered_map<int, std::vector<uint64_t>> InputDimMapping;
std::unordered_map<int,string> gMapping_id_to_InputDataType;
std::unordered_map<int,string> gMapping_id_to_OutputDataType;


void freeMemory(queue_elem* q){
    free(q->pReqProf);
    q->outVal.reset();
    free(q);
}
torch::Tensor getRandInput(int id, int batch_size){
    std::vector<int64_t> dimension;
    dimension.push_back(batch_size); //push batchsize first
    for(auto iter = InputDimMapping[id].begin(); iter != InputDimMapping[id].end(); iter++){
        dimension.push_back(*iter);
    }
    torch::Tensor input; 
    torch::TensorOptions options;
    if(gMapping_id_to_InputDataType[id] == "FP32"){
        torch::TensorOptions options;
        options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA,0).requires_grad(false);
    }
    else if(gMapping_id_to_InputDataType[id] == "INT64"){
        options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA,0).requires_grad(false);
    }
    input=torch::ones(torch::IntList(dimension),options);
    return input;
}
string getDirToFile(int model_id){
    string filename;
    for(auto pair : gMapping_file_to_id){
        if(pair.second == model_id){
            filename = pair.first;
            break;
        }
    }
    if(filename.empty()){
        printf("File for id %d NOT FOUND!! \n", model_id);
        return "FILE_NOT_FOUND";
    }
    return gCommonDir + filename;
}


void warmupModel(int id, int max_batch_size, torch::Device &gpu_dev)
{
    
    // Found out pytorch tends to use less memory when given a larger batch size for the first time
#ifdef PROXY_LOG
    cout << __func__ << ": called with max batch size: " << max_batch_size << endl;
#endif 

    	int batch=max_batch_size;

#ifdef PROXY_LOG
        uint64_t start,end;
        start=getCurNs();
#endif 

        torch::Tensor input = getRandInput(id, batch);
#ifdef PROXY_LOG
        cout << __func__ << ": type of random tensor: " << input.dtype() 
        << endl;
        cout << __func__ << ": size of tensor: " << input.sizes()
        << endl;
        cout << __func__ << ": device of tensor(before): " << input.device()
        << endl;
#
#endif 
        input=input.to(gpu_dev);

        std::vector<torch::IValue> inputs;
        inputs.push_back(input);
#ifdef PROXY_LOG
        cout << __func__ << ": device of tensor(after): " << input.device()
        << endl;
#endif

        gModelTable[id]->forward(inputs);
        cudaDeviceSynchronize();


#ifdef PROXY_LOG
        end=getCurNs();
        printf("[warmupModel] latency: %lf ms \n", double(end-start)/1000000);
#endif
        inputs.clear();
}

po::variables_map parse_opts(int ac, char** av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")
    ("common,com", po::value<std::string>()->default_value("../../pytorch-common/"),"Directory with configs and weights")
    ("devid,d", po::value<int>()->default_value(-1),"Device ID")
    ("threadcap,tc", po::value<int>()->default_value(100),"thread cap(used for designation)")
    ("dedup,dn", po::value<int>()->default_value(0),"identifier between same device and cap")
    ("config,cj", po::value<string>()->default_value("../proxy_config.json"),"file for configuring protocol")
    ("model_list", po::value<string>()->default_value("../ModelList.txt"),"file of a list of models to load")
    ("partition", po::value<int>()->default_value(0),"index of proxy in GPU")
    ("ngpu", po::value<int>()->default_value(2),"the total number of GPUs in this node(used in shared memory)")
    ("npart", po::value<int>()->default_value(7),"the total number of possible partitions(used in shared memory)");
  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return vm;
}

void initPyTorch(){
    cout << __func__ << ": called"<<endl;

    uint64_t total_end, total_start;
    total_start = getCurNs();
   
    std::vector<torch::jit::IValue> inputs;
 
    vector<int64_t> sizes={1};
    torch::TensorOptions options;
    options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA,0).requires_grad(false);
    
    torch::Tensor dummy1 = at::empty(sizes,options);
   
    cudaDeviceSynchronize();
    total_end = getCurNs();
    cout << double(total_end - total_start)/1000000 << " PyTorchInit total ms "<<endl;
    return;
}

void unloadModel(int id){
    #ifdef PROXY_LOG
    printf("[unloadModel] called for partition [%d,%d,%d] \n", gDevID, gThreadCap, gDedup);
    #endif 
    vector<int>::iterator fiter = find(gLoadedModelIDs.begin(), gLoadedModelIDs.end(),id);
    if(fiter == gLoadedModelIDs.end()){
        printf("model id : %d is NOT LOADED in memory. Skipping! \n", id);
        return;
    }

#ifdef PROXY_LOG
    printf("[unloadModel](%s) unloading jobid :%d \n", timeStamp(), id);
    uint64_t start,end;
    start=getCurNs();
#endif 

    gModelTable[id].reset();

    c10::cuda::CUDACachingAllocator::emptyCache();

    cudaDeviceSynchronize();
    gLoadedModelIDs.erase(fiter);
#ifdef PROXY_LOG
    end=getCurNs();
    printf("[unloadModel] latency: %lf ms \n" ,double(end-start)/1000000);
#endif
}

// you can use this after you called unload too, just like "reload"
void loadModel(queue_elem*  q, torch::Device gpu_dev,bool warmup=false){ 
    int id = q->jobid;
    int max_batch_size=q->batch_size;
    vector<int>::iterator fiter = find(gLoadedModelIDs.begin(), gLoadedModelIDs.end(),id);
    if(fiter != gLoadedModelIDs.end()){
        printf("model id : %d already loaded! skipping loading \n", id);
        return;
    }
    string temp_str = getDirToFile(id).c_str();

    const char* dir_to_file = temp_str.c_str();
#ifdef PROXY_LOG
    printf("[loadModel](%s) loading jobid :%d \n",timeStamp(), id);
    uint64_t start,end;
    start=getCurNs();
    uint64_t p1,p2,p3;
#endif 
    gModelTable[id] = make_shared<torch::jit::script::Module>(torch::jit::load(dir_to_file));
    #ifdef PROXY_LOG
    p1 = getCurNs();
    #endif 
    gModelTable[id]->to(gpu_dev);
    gModelTable[id]->eval();
    cudaDeviceSynchronize();
    cout << "finished loadiing" << endl;
    #ifdef PROXY_LOG
    p2 = getCurNs();
    #endif 
    if(warmup){
        warmupModel(id, max_batch_size, gpu_dev);
    }
    gLoadedModelIDs.push_back(id);

#ifdef PROXY_LOG
    end=getCurNs();
    cout <<"[loadModel] latency: " << double(end-start)/1000000 << endl;
    cout <<"[loadModel] jit::load  latency : " << double(p1-start)/1000000 << endl;
    cout <<"[loadModel] load_to_gpu latency : " << double(p2-p1)/1000000 << endl;
    cout <<"[loadModel] warmup latency : " << double(end-p2)/1000000 << endl;
#endif 

    return;
 }


void pushToQueue(queue_elem* input_elem, queue<struct queue_elem*> &queue, mutex &mtx, condition_variable  &cv){
    mtx.lock();
    queue.push(input_elem);
    mtx.unlock();
    cv.notify_all();
}


void* recv_input(void* vp){

    printTimeStampWithName(RECV, START);
    int server_sock, rc;
    socklen_t len;
    int i;
    int bytes_rec = 0;
    struct sockaddr_un server_sockaddr;
    struct sockaddr_un client_sockaddr;
    int cur_read;
    int backlog = 10;
    struct queue_elem* q;
            
    memset(&server_sockaddr, 0, sizeof(struct sockaddr_un));
    memset(&client_sockaddr, 0, sizeof(struct sockaddr_un));
            
    stringstream sockname;
    sockname<<"/tmp/gpusock_input_"<<gDevID<<"_"<<gThreadCap<<"_"<<gDedup;
     
    server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_sock == -1){
        printf("SOCKET ERROR: %s\n", strerror(errno));
        exit(1);
    }   
    server_sockaddr.sun_family = AF_UNIX;
    strcpy(server_sockaddr.sun_path, sockname.str().c_str());
    len=sizeof(server_sockaddr);
    setsockopt(server_sock, SOL_SOCKET,SO_REUSEADDR, NULL, 1);
    unlink(sockname.str().c_str());
    rc = bind(server_sock, (struct sockaddr *) &server_sockaddr, len);
    if (rc == -1){
        printf("BIND ERROR: %s\n", strerror(errno));
        close(server_sock);
        exit(1);
    }

    rc = listen(server_sock, backlog);
    if (rc == -1){ 
        printf("LISTEN ERROR: %s\n", strerror(errno));
        close(server_sock);
        exit(1);
    }
    printTimeStampWithName(server_sockaddr.sun_path, LISTENING);
    readyFlag_input=true;
    while (1){
    waiting_input_conn=true;
    int input_client_sock = accept(server_sock, (struct sockaddr *) &client_sockaddr, &len);
    if (input_client_sock == -1){
        printf("ACCEPT ERROR: %s\n", strerror(errno));
        close(server_sock);
        close(input_client_sock);
        exit(1);
    }
    waiting_input_conn=false;
    input_closed=false;
    printTimeStampWithName( server_sockaddr.sun_path, ACCEPTED);
    
    for(unordered_map<string, int>::iterator it = gMapping_file_to_id.begin(); it != gMapping_file_to_id.end(); it++){
        // allocate 40MB, which is more than enough to accomadate 32 * 3 * 300 * 300 32FP data
        per_jobid_input_mem[it->second]=(float*)malloc(INIT_MALLOC_SIZE*sizeof(float));

       #ifdef PROXY_LOG
       printf("input buffer for task %d initiated \n",it->second);
       #endif
    }
    
    while(1){
        int ret;
        int dimlen=0;
        int buf=0;
        int datalen=0;
        if (ret=read(input_client_sock,&dimlen, sizeof(int)) <=0){

            printf("client Disconnected  OR timed out\n");
            //printTimeStamp("CLOSED INPUT SOCKET");
            std::cout <<"CLOSED PROXY SOCKET AT: " << timeStamp() << std::endl;
            break;
        }
        if(dimlen == CLOSE_SOCKET){
            printf("received close socket from client \n");
            break;
        }
        else if(dimlen == LOAD_MODEL){
            int model_id;
            int batch_size;
            read(input_client_sock,&model_id, sizeof(int));
            read(input_client_sock,&batch_size, sizeof(int));
            printf("received load model %d from client \n", model_id);
            q=new queue_elem();
            q->act=LOAD;
            q->jobid=model_id;
            q->batch_size=batch_size;
            pushToQueue(q, gControlQueue, gCqueueMtx, gCqueueCV);
            continue;
        }
        else if(dimlen ==UNLOAD_MODEL){
            int model_id;
            read(input_client_sock,&model_id, sizeof(int));
            printf("received unload model %d from client \n", model_id);
             q=new queue_elem();
            q->act=UNLOAD;
            q->jobid=model_id;
            pushToQueue(q, gControlQueue, gCqueueMtx, gCqueueCV);
            continue;
        }
        else{
            printf("received dimlen: %d \n", dimlen);
        }
        receiving=true;
        if(dimlen > 4){
            printf("STRANGE dimlen: %d \n", dimlen);
            printf("continuing execution! \n");
            continue;
        }
         uint64_t start = getCurNs();
     
        if(dimlen!=0){
            q=new queue_elem();
            q->act=RUN;
            if (ret=read(input_client_sock, &buf, sizeof(int)) >0){
                q->jobid=buf;
            }
            

            for(int i =0; i <dimlen; i++){
                if ((ret=read(input_client_sock,&buf,sizeof(int))) > 0){
                    q->dims.push_back(buf);
                }
            }
            buf=0;
            if (ret=read(input_client_sock, &buf, sizeof(int)) >0){
                q->reqid=buf;
            }
            buf=0;
#ifdef PROFILE
            q->pReqProf=new ReqProfile(q->reqid, q->jobid);
            q->pReqProf->setInputStart(getCurNs());
#endif

            uint64_t start2 = getCurNs();           
            #ifndef NO_NET
            ret=read(input_client_sock,&datalen,sizeof(int));
            if(gMapping_id_to_InputDataType[q->jobid] == "FP32"){
                q->float32_indata=(float*)malloc(sizeof(float)*datalen);
                if (ret=SOCKET_receive(input_client_sock, (char*)q->float32_indata, datalen*sizeof(float), false) <=0){
                    printf("ERROR in receiving input data\n ");
                }
            }
            else if(gMapping_id_to_InputDataType[q->jobid] == "INT64"){
                q->int64_indata=(int64_t*)malloc(sizeof(int64_t)*datalen);
                if (ret=SOCKET_receive(input_client_sock, (char*)q->int64_indata, datalen*sizeof(int64_t), false) <=0){
                    printf("ERROR in receiving input data\n ");
                }
            }
            #else
            #endif
#ifdef PROXY_LOG
            cout <<__func__ <<": " <<timeStamp() << " received input as following: " << endl;
            printf("reqid: %d, jobid: %d \n", q->reqid, q->jobid);
#endif 
        
            uint64_t end2 = getCurNs();
            #ifdef PROXY_LOG
            cout << __func__ <<": finished receiving input data" << endl; 
            #endif
#ifdef PROFILE
            q->pReqProf->setInputEnd(getCurNs());
#endif 
            receiving=false;
            uint64_t end = getCurNs();

            pushToQueue(q, gInputQueue, gIqueueMtx, gIqueueCV);
            }
            else{
                printf("read returned 0. stop reading \n");
                break;
            }
        }// inner loop
        SOCKET_close(input_client_sock,true);
        input_closed=true;
        gOqueueCV.notify_all();
        gIqueueCV.notify_all();
        gCqueueCV.notify_all();

        if(exitFlag) break;

    }//outer loop
    cout << "exiting input thread" << endl;  
}

void* load_control(void *args){
    while(1){
        unique_lock<mutex> lock(gCqueueMtx);
        gCqueueCV.wait(lock, []{return gControlQueue.size() || exitFlag;});
        if(exitFlag){
            break;
        }
        queue_elem *q=gControlQueue.front();
        gControlQueue.pop();
        if(q->act == LOAD) loadModel(q,gpu_dev,true);
        else if(q->act == UNLOAD) unloadModel(q->jobid);
        // should not happen
        else{ 
            printf("ERROR! CHECK your code for load_control \n");
        }
        if(gControlQueue.size() == 0 && (gProxyCtrl->getProxyState(gpPInfo) != EXITING) )
        {
            //print time stamp for debugging purposes
            #ifdef PROXY_LOG
            cout << __func__ << ": marking proxy as RUNNING at " << timeStamp()
            <<endl; 
            #endif
            gProxyCtrl->markProxy(gpPInfo, RUNNING);
        }
        freeMemory(q);
    }
    cout << "exiting load_control thread" << endl;

}


pthread_t init_load_control_thread(){
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8*1024 * 1024);
    pthread_t tid;

    if(pthread_create(&tid, &attr, load_control, NULL)!=0){
        printf("init_input_thread: Error\n");
    }
    return tid;
}

pthread_t init_input_thread(){
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8*1024 * 1024);
    pthread_t tid;

    if(pthread_create(&tid, &attr, recv_input, NULL)!=0){
        printf("init_input_thread: Error\n");
    }
    return tid;
}

void* send_output(void* vp){
    printTimeStampWithName(SEND, START);
    int server_sock,rc;
    socklen_t len;
    int i;
    int bytes_rec = 0;
    struct sockaddr_un server_sockaddr; 
    struct sockaddr_un client_sockaddr;
    int cur_read;
    int backlog = 10;
            
    memset(&server_sockaddr, 0, sizeof(struct sockaddr_un));
    memset(&client_sockaddr, 0, sizeof(struct sockaddr_un));
            
    stringstream sockname;
    
    sockname<<"/tmp/gpusock_output_"<<gDevID<<"_"<<gThreadCap<<"_"<<gDedup;
           
                
    server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_sock == -1){
        printf("SOCKET ERROR: %s\n", strerror(errno));
        exit(1);
    }   
    server_sockaddr.sun_family = AF_UNIX;
    strcpy(server_sockaddr.sun_path, sockname.str().c_str());
    len=sizeof(server_sockaddr);
        
    unlink(sockname.str().c_str());
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR,NULL,1) ;
    rc = bind(server_sock, (struct sockaddr *) &server_sockaddr, len);
    if (rc == -1){
        printf("BIND ERROR: %s\n", strerror(errno));
        close(server_sock);
        exit(1);
    }

    rc = listen(server_sock, backlog);
    if (rc == -1){ 
        printf("LISTEN ERROR: %s\n", strerror(errno));
        close(server_sock);
        exit(1);
    }
    printTimeStampWithName(server_sockaddr.sun_path, LISTENING);
    readyFlag_output=true;
    while(1){
        waiting_output_conn=true;
        int output_client_sock = accept(server_sock, (struct sockaddr *) &client_sockaddr, &len);
        if (output_client_sock == -1){
                printf("ACCEPT ERROR: %s\n", strerror(errno));
                close(server_sock);
                close(output_client_sock);
                exit(1);
        }
        waiting_output_conn=false;
        printTimeStampWithName(server_sockaddr.sun_path, ACCEPTED);
        while(1){
                unique_lock<mutex> lock(gOqueueMtx);
                gOqueueCV.wait(lock, []{return gOutputQueue.size() || exitFlag;});
                std::cout << "receiving: " <<  receiving << " computing: "<< computing <<   " gInputQueue: " << gInputQueue.size() 
                << " gOutputQueue.size: " << gOutputQueue.size() << " exitFlag: " << exitFlag 
                <<endl;
                if(!receiving && gInputQueue.size() ==0 && gOutputQueue.size() ==0 && exitFlag){
                    break;
                }
                if(gOutputQueue.empty()){
                    lock.unlock();
                    continue;
                } 
                struct queue_elem* q =gOutputQueue.front();
#ifdef PROFILE
                q->pReqProf->setOutputStart(getCurNs());
#endif 
                gOutputQueue.pop();
                lock.unlock();
               	torch::Tensor otensor = q->outVal;
				int dim = otensor.dim();

			    int len=1;
				write(output_client_sock, (void *)&dim,sizeof(int));
				for(int i=0; i<dim; i++){
						int size = otensor.size(i);
						write(output_client_sock, (void *)&size,sizeof(int));
						len*=size;
				}
#ifndef NO_NET
				float *raw_data=(float*)(q->outVal).data_ptr();
				SOCKET_send(output_client_sock, (char*)raw_data, len*sizeof(float), false); 
#else
#endif
#ifdef PROFILE
                q->pReqProf->setOutputEnd(getCurNs());
                q->pReqProf->printTimes();
#endif 

                #ifdef PROXY_LOG
                cout << __func__ <<": output sent for req_id: " << q->reqid <<" at: " << timeStamp() << endl;
                #endif
                freeMemory(q);

            } // inner infinite loop
            SOCKET_close(output_client_sock, true);
            if(exitFlag) break;
    }// outer infinit loop
    cout << "exiting output thread" << endl;
}


pthread_t init_output_thread(){
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8*1024 * 1024);
    pthread_t tid;

    if(pthread_create(&tid, &attr, send_output, NULL)!=0){
        printf("init_output_thread: Error\n");
    }
    return tid;
}

void* compute(void* vp){
    c10::InferenceMode guard;
    uint64_t com_start, com_end;
    com_start=getCurNs();
    printTimeStampWithName(COMP, START);
    
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor input;
    torch::Tensor t;
    initPyTorch();

    readyFlag_compute=true;
    while(!readyFlag_input || !readyFlag_output || !readyFlag_compute){}
    printTimeStampWithName(COMP, DONE);
    gProxyCtrl->markProxy(gpPInfo, RUNNING);
    com_end = getCurNs();
    #ifdef PROXY_LOG
    cout << "computing thread took " << double(com_end - com_start)/1000000 << " ms to initiate" << endl; 
    #endif

    while(1){
        //compute here
        unique_lock<mutex> lock(gIqueueMtx);
        gIqueueCV.wait(lock, []{return gInputQueue.size() || exitFlag;});
        if (gInputQueue.size() ==0 && !receiving && exitFlag ){
            lock.unlock();
            break;
        }
        computing=true;
        struct queue_elem* q=gInputQueue.front();
        gInputQueue.pop();
        lock.unlock();
#ifdef PROFILE
        q->pReqProf->setCPUCompStart(getCurNs());
#endif 
 #ifdef PROXY_LOG
        printf("started to execute request ID: %d \n", q->reqid);
#endif 

        #ifndef NO_NET
        if(gMapping_id_to_InputDataType[q->jobid] == "FP32"){
            auto options(torch::kFloat32);
            t=convert_rawdata_to_tensor(q->float32_indata, q->dims, options);
        }
        else if(gMapping_id_to_InputDataType[q->jobid] == "INT64"){
            auto options(torch::kInt64);
            t=convert_rawdata_to_tensor(q->int64_indata, q->dims, options);
        }
        #else 
         t=getRandInput(q->jobid,q->dims[0]);   
        #endif


#ifdef PROXY_LOG
        cout << __func__ <<": type of tensor " << t.dtype()
                <<endl;
        cout << __func__ <<": sizes " << t.sizes()
                <<endl;
#endif 
        t=t.to(gpu_dev);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(t);
        {
        int input_batch_size = t.size(0);
#ifdef PROFILE
        printf("setting gpu_start of req_id: %d \n", q->reqid);
        q->pReqProf->setGPUStart(getCurNs());
#endif

        torch::IValue temp_output= gModelTable[q->jobid]->forward(inputs);
        cudaDeviceSynchronize();
#ifdef PROFILE
        printf("setting gpu_end of req_id: %d \n", q->reqid);
        q->pReqProf->setGPUEnd(getCurNs());
        q->pReqProf->setBatchSize(input_batch_size);
#endif

        torch::Tensor temp;
        if (temp_output.isTensor()){
            temp= temp_output.toTensor();
        }
        else if(temp_output.isTuple() && (q->jobid == gMapping_file_to_id["ssd-mobilenetv1.pt"])){
            temp=custom::getBoxforTraffic(temp_output.toTuple()->elements().at(0).toTensor(), 
                            temp_output.toTuple()->elements().at(1).toTensor(),
                            t);
        }
        else if(temp_output.isTuple() && (q->jobid == gMapping_file_to_id["bert.pt"])){
            // pick the only tuple for output
            temp = temp_output.toTuple()->elements().at(0).toTensor();
        }
        else{ // not supposed to happen
            printf("LOGIC ERROR \n ");
            exit(1);
        }
        temp = temp.to(torch::kCPU);
        temp.detach();
        q->outVal = temp;

        int output_batch_size = temp.size(0);
        assert(input_batch_size == output_batch_size);
#ifdef PROFILE
        q->pReqProf->setCPUPostEnd(getCurNs());
#endif
       //send output
        pushToQueue(q,gOutputQueue, gOqueueMtx,gOqueueCV);
        computing=false;

        }// dummy scope for lowering shared_ptr use_counts 
 
    }//infinite loop
    if(exitFlag){
        gOqueueCV.notify_one();
    }
    cout << "exiting compute thread" << endl;
}

pthread_t init_compute_thread(){
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8*1024 * 1024);
    pthread_t tid;

    if(pthread_create(&tid, &attr, compute, NULL)!=0){
        printf("init_compute_thread: Error\n");
    }
    return tid;
}
void* control(void *args){
    #ifdef PROXY_LOG
    cout << "[" << gDevID << "," << gPartIdx << "]" << "starting control thread" << endl;
    #endif 
    while(true){
        proxy_state curr_state;
        curr_state=gProxyCtrl->getProxyState(gpPInfo);
        while(curr_state != EXITING){
            cout << "STATE: " << curr_state << " at "<< timeStamp() <<endl;
            usleep(1*1000*1000);
            curr_state=gProxyCtrl->getProxyState(gpPInfo);
        }
        exitFlag=true;
        gIqueueCV.notify_one();
        gOqueueCV.notify_one();
        gCqueueCV.notify_one();
        break;
     }
    cout << "exiting control thread" << endl;
    return (void*)0;
}
pthread_t init_control_thread(){
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8*1024 * 1024);
    pthread_t tid;
    if(pthread_create(&tid, &attr, control, NULL)!=0){
        printf("init_control_thread: Error\n");
    }
    return tid;
}


int main(int argc, char** argv){    
    torch::jit::getBailoutDepth() = 0;
    torch::jit::getProfilingMode() = false;
    uint64_t main_start, main_end;
    main_start=getCurNs();
    pthread_t compute,control, send, recv, load_control;
    po::variables_map vm=parse_opts(argc, argv);
    gDevID=vm["devid"].as<int>();
    gDedup=vm["dedup"].as<int>();
    gThreadCap=vm["threadcap"].as<int>();
    gCommonDir=vm["common"].as<string>() + "models/";
    gPartIdx = vm["partition"].as<int>();
    gpPInfo = new proxy_info();
    gpPInfo->dev_id=gDevID;
    gpPInfo->partition_num = gPartIdx;
    
    gProxyCtrl= new ProxyCtrl(/*clear_memory=*/false);
    gProxyCtrl->markProxy(gpPInfo, BOOTING);
    
    readInputDimsJsonFile(vm["config"].as<string>().c_str(), gMapping_name_to_id, InputDimMapping);
    readInputTypesJSONFile(vm["config"].as<string>().c_str(), gMapping_id_to_InputDataType, gMapping_id_to_OutputDataType);
    
    for(auto pair : gMapping_name_to_id ){
        string file_name = pair.first + ".pt";
        gMapping_file_to_id[file_name] = pair.second;
    }
    stringstream ss;
    ss<<"/tmp/nvidia-mps";
    if(gDevID<4){
        setenv("CUDA_MPS_PIPE_DIRECTORY", ss.str().c_str(), 1);
    }
    recv=init_output_thread();
    send=init_input_thread();
    pthread_detach(send);
    main_end=getCurNs();
    #ifdef PROXY_LOG
        cout << "main to init_compute_thread took: " << double(main_end-main_start)/1000000 <<" ms" <<endl;
    #endif

    compute=init_compute_thread();
    load_control = init_load_control_thread(); 
    pthread_detach(load_control);
    control=init_control_thread();
    pthread_join(control, NULL);    
    pthread_join(compute, NULL);
    if(!waiting_output_conn) pthread_join(send, NULL);
    gProxyCtrl->markProxy(gpPInfo,FLUSHED);
    while(!input_closed){usleep(1*1000);}
    gProxyCtrl->markProxy(gpPInfo, COLD);
    cout << "MAIN thread reached return" << endl;
    return 0;
}
