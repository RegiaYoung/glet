#ifndef _ROUTER_H__
#define _ROUTER_H__

#include <deque>
#include <vector>
#include <queue>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <map>
#include <boost/lockfree/queue.hpp>
#include <thread>
#include "batched_request.h"
#include "concurrentqueue.h"
#include "gpu_proxy.h"
#include "interference_model.h"
#include "latency_model.h"
#include "scheduler_incremental.h"
#include "EWMA.h"
#include "proxy_ctrl.h"
#include "gpu_utils.h"
#include "self_tuning.h" 
#include "load_balancer.h"
#include "backend_proxy_ctrl.h"
class AppSpec;
using namespace std;
enum scheduler {MPS_STATIC, ORACLE};

typedef struct _TaskSpec{
    int device_id;
    string ReqName;
    int BatchSize;
    int dedup_num;
    int CapSize; // maximum amount of cap for this task, used in MPS environment
    proxy_info* proxy;
} TaskSpec; // mostly used for scheduling

//struct which stores system state related info, used in stateful scheduling
typedef struct _SysMonitor{
	map<string, queue<shared_ptr<request>>> ReqListHashTable; // per request type queue
    moodycamel::ConcurrentQueue<shared_ptr<request>> *cmpQ;
    map<proxy_info*, deque<shared_ptr<TaskSpec>>*> PerProxyBatchList; // list of tasks to batch for each [dev,cap] pair
    map<proxy_info*, vector<pair<string,int>>> PerProxyTaskList; // used in MPS_STATIC,list of available tasks in proxy, stores [{modelname, batch_size}]
    map<int, vector<proxy_info*>> PerDevMPSInfo;
    map<string,int> PerModeltoIDMapping;
    int nGPUs; // number of GPUS
    int nProxyPerGPU; // number of proxys per gpu
    double DROP_RATIO;
    bool IS_LOCAL;
    bool EXP_DIST;
    vector<AppSpec> AppSpecVec;
    bool TRACK_LATENCY;
    bool TRACK_INTERVAL;
    bool TRACK_TRPT;
    bool SYS_FLUSH;
    // used in backend, the proxy_dir, full path
    string FULL_PROXY_DIR;
    // used in backend, the name of proxy script for booting proxys
    string PROXY_SCRIPT;
    // used in backend, index conversion table from frontend view of ID, to actual GPU ID in host
    map<int,int> FrontDevIDToHostDevID;
    int nHostGPUs;
       // number of emulated backend node, 0 by default
    int nEmulBackendNodes;
    // used for trackign per model rate 
    map<string, uint64_t> PerModelCnt;
    // used for tracking per model trpt
    map<string,int> PerModelFinCnt;
     // used for tracking per model avg latency
    map<string,float> PerModelLatestLat;   
    map<string, bool> PerModelFlushed;

    _SysMonitor(){
        // for performance issues we explicitely set how much the queue should hold 
        cmpQ = new moodycamel::ConcurrentQueue<shared_ptr<request>>(1000);
        nEmulBackendNodes=0;
    }

    proxy_info* findProxy(int dev_id, int partition_num){
        if(FrontDevIDToHostDevID.empty()){
            assert(dev_id < nGPUs);
        }
        else{
            int first_idx= FrontDevIDToHostDevID[0];    
            assert(dev_id-first_idx < nGPUs);
        }
        #ifdef BACKEND_DEBUG
        cout << __func__ << ": finding dev_id: " << dev_id << ", partition_num: " << partition_num
        << endl;
        #endif
        proxy_info* ret = NULL;
        for(auto pPInfo : PerDevMPSInfo[dev_id]){
            if(pPInfo->partition_num == partition_num)  
            {   
                ret=pPInfo;
                break;
            }   
        }   
        assert(ret != NULL);
        return ret;
    }

    proxy_info* findProxy(int dev_id, int resource_pntg, int dedup_num){
        if(FrontDevIDToHostDevID.empty()){
            assert(dev_id < nGPUs);
        }
        else{
            int first_idx= FrontDevIDToHostDevID[0];    
            assert(dev_id-first_idx < nGPUs);
        }
        proxy_info* ret = NULL;
        for(auto pPInfo : PerDevMPSInfo[dev_id]){
            if(pPInfo->dev_id == dev_id && pPInfo->cap == resource_pntg && pPInfo->dedup_num == dedup_num)  
            {   
                ret=pPInfo;
                break;
            }   
        }   
        assert(ret != NULL);
        return ret;
    }

    
} SysMonitor;

typedef struct _load_args{
    proxy_info *pPInfo;
    vector<pair<int,int>> model_ids_batches;
} load_args;

typedef struct _reroute_args{
    SysMonitor *SysState;
    GPUPtr gpu_ptr;
} reroute_args;

class GlobalScheduler
{

public: 
    GlobalScheduler();
    ~GlobalScheduler();
    //below are methods related to scheduling
    vector<shared_ptr<TaskSpec>> executeScheduler(SysMonitor *SysState);
    void doMPSScheduling(SysMonitor* SysState);
    // below are schedulers called according to _mode
    vector<shared_ptr<TaskSpec>> staticMPSScheduler(SysMonitor *SysState);
    vector<shared_ptr<TaskSpec>> OracleScheduler(SysMonitor *SysState);
    vector<shared_ptr<TaskSpec>> STScheduler(SysMonitor *SysState);
    /*methods for setting up*/
    int setSchedulingMode(string mode, SysMonitor *SysState);
    void setLoadBalancingMode(string mode);
    // mps_static specific load balance setup, called in 
    void setupLoadBalancer(map<proxy_info*, vector<pair<int,double>>> &mapping_trps);
    void setupScheduler(SysMonitor *SysState, map<string,string> &param_file_list, string scheduler, string res_dir);
    void setupSBPScheduler(SysMonitor *SysState, map<string,string> &param_file_list, string res_dir);
    void setupProxys(SysMonitor *SysState, string model_list_file);
    void setupInitTgtRate(string rate_file);
    void setupLatencyModel(string latency_file);
    void setupEstModels();
    void setupMPSInfo(SysMonitor *SysState, string capfile);
    void createProxyInfos(SysMonitor *SysState, vector<int> parts, int node_id, int ngpus, string type);
    void setupProxyTaskList(string list_file, SysMonitor *SysState);
    void setupPerModelLatencyVec(SysMonitor *SysState);
    void setupModelSLO(string name, int SLO);
    void bootProxys(SysMonitor *SysState);
    proxy_info* getProxyInfo(int dev_id, int thread_cap, int dedup_num, SysMonitor *SysState);
    vector<Task> getListofTasks(SysMonitor *SysState);
    vector<int> getListofParts(SysMonitor *SysState);
    vector<vector<int>> getParts(SysMonitor *SysState);
    void loadAllProxys(SysMonitor *SysState);
    void loadModelstoProxy(proxy_info* pPInfo, SysMonitor *SysState);
    void unloadModelsfromProxy(proxy_info* pPInfo, SysMonitor *SysState);   
    void setMaxBatch(string name, string type, int max_batch);
    void setMaxDelay(string name, string type, int max_delay);  
    float getTaskTgtRate(string model_name);  
    /*methods for updating*/
    void initReqRate(string ReqName, int monitor_interval_ms, int query_interval_ms, float init_rate);
    void updateAvgReqRate(string ReqName, uint64_t timestamp);
    float getTaskArivRate(string ReqName);
    void addtoNetNames(string name);
    void removeMaxRates();  
    // conduct scheduling
    bool doScheduling(vector<Task> &task_list, SysMonitor *SysState,SimState &curr_decision, SimState &new_decision);
    // recovers fron scheduliong failure
    void recoverFromScheduleFailure(SysMonitor *SysState, map<string,float> &backup_rate, SimState &backup);
    //setup only tasks that need reschediling
    void setupTasks(SysMonitor *SysState, bool include_downsize, vector<Task> &output_vec);
    //setup all tasks for scehduling
    void setupAllTasks(SysMonitor *SysState, vector<Task> &output_vec);
    // fill task_list with current configured rate
    void refillTaskList(SysMonitor *SysState, vector<Task> &task_list);
    //updates history of rates and returns maximum rate
    float updateAndReturnMaxHistRate(string model_name, const float rate);
    void applyReorganization(SysMonitor *SysState, const SimState &output);
    void RevertTgtRates(SimState &backup); 
    /*methods for getting numbers */
    float getMaxDelay(proxy_info *pPInfo);
    vector<string>* getNetNames();
    int getSLO(string name);
    float getEstLatency(string task, int batch, int cap, string type);
    float getBeaconInterval();
    int getModelID(string name);
    bool detectRepart(SysMonitor *SysState, GPUPtr &gpu_ptr);
    /*methods for updating latency related info*/
    float getTailLatency(proxy_info* pPInfo, string modelname);
    float getAvgLatency(proxy_info* pPInfo, string modelname);
    void clearPerModelLatency(proxy_info *pPInfo, string model);
    void addLatency(proxy_info *pPInfo, string modelname, float latency);
    void initPerModelLatEWMA(SysMonitor *SysState, int monitor_interval_ms, int query_interval_ms);
    float getAvgLatency(string modelname);
    void clearPerModelLatency(string model);
    void addLatency(string modelname, float latency);
    void initPerModelTrptEWMA(SysMonitor *SysState, int monitor_interval_ms, int query_interval_ms);
    float getAvgTrpt(string modelname);
    void clearPerModelTrpt(string model);
    void addTrpt(string modelname, float tprt);
    void UpdateBatchSize(SysMonitor *SysState, string model_name, int new_batch_size);
    void initPerModelPartSum(SysMonitor *SysState, unordered_map<int,int> &sum_of_parts);
    
    void additionalLoad(GPUPtr &gpu_ptr, SysMonitor *SysState);
    void unloadModels(GPUPtr &gpu_ptr, SysMonitor *SysState);
    void unloadModels(SimState &prev_state, SimState &curr_state, SysMonitor *SysState);
    bool checkIfLoaded(proxy_info *pPInfo, string modelname);
    bool needToUnload(NodePtr &node_ptr, _proxy_info* pPInfo);
    bool needToLoad(NodePtr &node_ptr, _proxy_info* pPInfo);
    void updateGPUMemState(SimState &input, SysMonitor *SysState);
    void updateGPUMemState(SimState &input, SysMonitor *SysState, int num_of_gpus_to_update);
    bool checkLoad(int model_id, proxy_info* pPInfo);
    //below are scale related functions!!
    // checks and reduce the number of schedulable GPUs, returns true if reduced, false it was not reduced
    bool checkandReduceGPU(SimState &decision, SysMonitor *SysState);
    // totally flushes and updates memory configuration
    void flushandUpdateDevs(SimState &input, const int num_of_gpus);
    // functions used for retrieving memory information of GPU on remote node
    int getTotalMemory_backend(int gpu_id, int node_id);
    int getUsedMemory_backend(proxy_info *pPInfo);
    // usually called with update updateGPUMemState
    void updateGPUModelLoadState(SimState &input, SysMonitor *SysState);
    void* loadModel_sync_wrapper(void *args);
    void loadModel_sync(proxy_info *pPInfo, vector<pair<int,int>> &model_ids);
    void* unloadModel_async(void *args);
    void applyRouting_sync(SimState &output, SysMonitor *SysState);
    void bootProxy_sync(proxy_info* pPInfo);
    void shutdownProxy_sync(proxy_info* pPInfo);
    int FullRepartition(SimState &input, SimState &output, SysMonitor *SysState);
    int NonRepartition(SimState &input, SimState &output, vector<Task> &session, SysMonitor *SysState);
    int oneshotScheduling(SysMonitor *SysState);
    int readAndBoot(SysMonitor *SysState, string model_list_file);
    void setupBackendProxyCtrls();
    

private: 
    scheduler mSchedulerMode;
    //repartition_policy mRepartitionMode;
    LoadBalancer mLoadBalancer;
    vector<string> mNetNames; // vector containing names of network that can run on the server
    map<tuple<string, string>, int> mMaxBatchTable;//max batch table 
    map<tuple<string, string>, int> mMaxDelayTable; //max delay table
    // used in self-tuning scheduler 
    map<string, int> mPerModelLastBatch; 
    map<string, deque<float>> mPerTaskAvgArivInterval;  // used in get/set arriving intervals
    map<string, shared_ptr<EWMA::EWMA>> mPerTaskEWMARate; // EWMA value of rate
    map<string, shared_ptr<EWMA::EWMA>> mPerTaskEWMALat; // EWMA value of avg latency
    map<string, shared_ptr<EWMA::EWMA>> mPerTaskEWMATrpt; // EWMA value of avg throughput
    map<string, deque<float>> mPerTaskRateHist;  // used in get/set arriving intervals
    map<string, int> mPerTaskDownCnt;  // use
    map<std::string, mutex*> PerTaskArrivUpdateMtx; 
    map<std::string, mutex*> PerTaskLatUpdateMtx; 
    map<std::string, mutex*> PerTaskTrptUpdateMtx; 
    map<string,uint64_t> mPerTaskLastTimestamp; 
    map<string, float> mPerTaskTgtRate;  //  used in scheduling
    /*tables used for STScheduler only */
    map<string, int> mPerTaskSLOSuccess; 
    unordered_map<int, int> mPerTaskSumofPart;
    
    interference_modeling::interference_model mInterferenceModel;   
    LatencyModel mLatencyModel;
    map<string,int> mPerModelSLO;
    // the state currently scheduled
    SimState mCurrState;
    // store state that needs to be done before the start of the next epoch
    SimState mNextState;
    //store previously loaded state, which gives information of unneeded models
    SimState mPrevState;
    
    bool mNeedUpdate=false;
    squishy_bin_packing::IncrementalScheduler SBPScheduler;
    SelfTuning *pSTScheduler;
    BackendProxyCtrl* mProxyCtrl;
    const unsigned int MA_WINDOW=10; //used for tracking moving average, this is the number of records to keep when getting average
    const float SLO_GUARD=0.05;
    const int SKIP_CNT=100; // used for skipping tasks, especially useful for preventing resource osciliation
    const float LATENCY_ROOM=1.05;
    uint64_t BEACON_INTERVAL; 
    uint64_t EPOCH_INTERVAL;
    uint64_t REORG_INTERVAL;
    float HIST_LEN; // used with mPerTaskRateHist, initiated in EpochScheduler limits the number of past history the deque will hold
    float DOWN_COOL_TIME; // used and setted in SetupTasks, based on BEACON_INTERVAL
    bool ALLOW_REPART=false;
    bool SYS_FLUSH=false;
    
};

#endif 
