#include "proxy_ctrl.h"
#include "thread.h"
#include "common_utils.h"
#include "interference_model.h"
#include "config.h"
#include "router.h"
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <assert.h>
#include "scheduler_utils.h"
#include "backend_delegate.h"

extern SysMonitor ServerState;
extern map<int,BackendDelegate*> NodeIDtoBackendDelegate;

int GlobalScheduler::setSchedulingMode(string mode, SysMonitor *SysState){
	// setup tracking flags
	SysState->TRACK_LATENCY=false;
	SysState->TRACK_INTERVAL=true;
	SysState->TRACK_TRPT=false;

	if(mode == "mps_static"){
		mSchedulerMode=MPS_STATIC;
	}
	else if(mode == "oracle"){
		mSchedulerMode=ORACLE;
		ALLOW_REPART=true;
	}
	else
		return EXIT_FAILURE;
	return EXIT_SUCCESS;
}

void GlobalScheduler::setLoadBalancingMode(string mode){
	mLoadBalancer.setType(mode);
}

//added for incremental scheduling
vector<Task> GlobalScheduler::getListofTasks(SysMonitor *SysState){
	vector<string> models;
	vector<Task> ret_task;
	for(auto name : mNetNames){
		vector<string>::iterator it = find(models.begin(), models.end(),name);
		if(it == models.end()){	
			models.push_back(name);
		}
	}	
	for(auto name : models){
		Task new_task;
		new_task.id=SBPScheduler.GetModelID(name);
		new_task.SLO=getSLO(name);
		new_task.request_rate=0; // this is OK since, there is no need for request rate to be present when getting saturated trp
		ret_task.push_back(new_task);
		cout << "[getListofTasks] task: " << new_task.id << ", SLO: "<< new_task.SLO << endl;
	}
	return ret_task;

}

vector<int> GlobalScheduler::getListofParts(SysMonitor *SysState){
	vector<int> ret_vec;
	for(int i=0; i<SysState->nGPUs; i++){
		for(auto proxy_info : SysState->PerDevMPSInfo[i]){
		int part = proxy_info->cap;
        if(part==0) continue;
		vector<int>::iterator it = find(ret_vec.begin(), ret_vec.end(),part);
		if(it == ret_vec.end()) ret_vec.push_back(part);
		}
	}
	assert(!ret_vec.empty());
	return ret_vec;
}

vector<vector<int>> GlobalScheduler::getParts(SysMonitor *SysState){
	vector<vector<int>> ret_vec;
	for(int i=0; i<SysState->nGPUs; i++){
		vector<int> tmp;
		for(auto proxy_info : SysState->PerDevMPSInfo[i]){
			int part = proxy_info->cap;
            if(part==0) continue;
			tmp.push_back(part);
		}
		ret_vec.push_back(tmp);
	}
	assert(!ret_vec.empty());
	for(auto vec: ret_vec) assert(!vec.empty());
	return ret_vec;
}

proxy_info* createNewProxy(int dev_id, int cap, int dedup_num, int part_index, string type){
	proxy_info* pPInfo = new proxy_info();
	pPInfo->dev_id = dev_id;
	pPInfo->cap = cap;
	pPInfo->dedup_num=dedup_num;
	pPInfo->duty_cycle=0; // THIS WILL BE OVERWRITTEN IN setupProxyTaskList
	pPInfo->partition_num = part_index;
	pPInfo->isBooted=false;
	pPInfo->isConnected=false;
	pPInfo->isSetup=false;
	pPInfo->type=type;
	#ifdef DEBUG
	printf("[createNewProxy] create new proxy: [%d,%d,%d][%d] type: %s \n",dev_id, cap, dedup_num, part_index, type.c_str());
	#endif
	return pPInfo;
}

void GlobalScheduler::setupSBPScheduler(SysMonitor *SysState, map<string,string> &param_file_list, string res_dir){
	setupPerModelLatencyVec(SysState);
	//for the moment we hardcode the name of files required
	vector<string> files = {"1_28_28.txt", "3_224_224.txt", "3_300_300.txt"};
	bool success = SBPScheduler.InitializeScheduler(param_file_list["sim_config"], \
									param_file_list["mem_config"], \
									param_file_list["device_config"], \
									res_dir,\
									files);
	if(!success){
		cout<< __func__ <<": CRITICAL : failed to initialize scheduler!"
		<<endl;
		exit(1); 
	}

	SBPScheduler.SetMaxGPUs(SysState->nGPUs);

	//make proxy_info for every possible partitoin on every GPU
	vector<Task> list_of_tasks = getListofTasks(SysState);
	for (auto task : list_of_tasks){
		SBPScheduler.InitSaturateTrp(task);
	}
}

void GlobalScheduler::setupScheduler(SysMonitor* SysState, map<string,string> &param_file_list, string scheduler, string res_dir ){
  	if(setSchedulingMode(scheduler, SysState)){
    	printf("exiting server due to undefined scheduling mode\n");
    	exit(1);
  	}	
	#ifdef ChECK_NETBW
	SBPScheduler.setupNetworkChecker(res_dir+"/"+param_file_list["proxy_json"]);
	#endif
	switch(mSchedulerMode){
		case MPS_STATIC:
			setupSBPScheduler(SysState,param_file_list,res_dir);
			setupMPSInfo(SysState, param_file_list["model_list"]);
			break;
		case ORACLE:
			setupSBPScheduler(SysState,param_file_list, res_dir);
			EPOCH_INTERVAL= 20000; // interval of checking rates, in ms,
			BEACON_INTERVAL=EPOCH_INTERVAL; // required in initReqRate
			HIST_LEN=30 / float(EPOCH_INTERVAL / 1000.0);
			break;
		default: // should not happen, conditions are already checked during initialization
			break;
	}
}

// create proxy_infos for 'ngpus' on node 'node_id' with configurable 'parts'
void GlobalScheduler::createProxyInfos(SysMonitor *SysState, vector<int> parts, int node_id, int ngpus, string type){
	//make proxy_info for every possible partitoin on every GPU
	map<int,int> perdev_part_idx;
	for(int i =0; i < SysState->nGPUs+ngpus; i++) perdev_part_idx[i]=0;
	for(int devindex = SysState->nGPUs; devindex < SysState->nGPUs+ngpus; devindex++ ){
		for(int part: parts){
			int dedup_num=0;
			proxy_info* pPInfo = createNewProxy(devindex,part,dedup_num,perdev_part_idx[devindex], type);
			pPInfo->node_id=node_id;
			SysState->PerDevMPSInfo[devindex].push_back(pPInfo);			
			perdev_part_idx[devindex]++;
			// if partition is 50, make another proxy, with dedup_num=1
			if(part == 50){ 
				dedup_num++;
				proxy_info* pPInfo = createNewProxy(devindex,part,dedup_num,perdev_part_idx[devindex], type);
				pPInfo->node_id=node_id;
				SysState->PerDevMPSInfo[devindex].push_back(pPInfo);			
				perdev_part_idx[devindex]++;
			}
		}
	}
	SysState->nGPUs+=ngpus;
}

void GlobalScheduler::setupInitTgtRate(string rate_file){
  cout << __func__ <<": reading " << rate_file
	<< endl;
  string str_buf;
  fstream fs;
  fs.open(rate_file, ios::in);
  if(!fs){
	  cout << __func__ << ": file " << rate_file << " not found"
	  <<endl;
	  exit(1);
  }
  
  getline(fs, str_buf);
  int num_of_rates = stoi(str_buf);
  for(int j=0;j<num_of_rates;j++)
  { 
    getline(fs,str_buf,',');
    int id=stoi(str_buf);
    
    getline(fs,str_buf,',');
    int request_rate=stoi(str_buf);
    
    getline(fs,str_buf,',');
    int SLO=stoi(str_buf);
	mPerTaskTgtRate[SBPScheduler.GetModelName(id)]=request_rate;
    
  }  
 fs.close();
#ifdef EPOCH_DEBUG
	cout << "Rate setup ------------------------------------------------" << endl; 
	cout << "task name : [request_rate]" << endl;
	for(auto info_pair : mPerTaskTgtRate)
	{ 
		cout << info_pair.first << " : [" << info_pair.second << "]" << endl;
	}
	cout << endl;
	cout << "------------------------------------------------" << endl; 
#endif

}

void GlobalScheduler::setupLatencyModel(string latency_file){
	mLatencyModel.setupTable(latency_file);
}



void GlobalScheduler::setupMPSInfo(SysMonitor *SysState, string capfile){
		ifstream infile(capfile);
		string line;
		string token;
		string first_task;
		map<int, int> perdev_part_idx;
		for(int i =0; i < SysState->nGPUs; i++) perdev_part_idx[i]=0;
		while(getline(infile,line)){
			istringstream ss(line);
			getline(ss,token,',');
			int devindex=stoi(token);
			getline(ss,token,',');
			int cap=stoi(token);
			getline(ss,token,',');
			int dedup_num=stoi(token);
			getline(ss,token,',');
			proxy_info *pPInfo = getProxyInfo(devindex,cap,dedup_num,SysState);		
			bool found = false;
			for(vector<proxy_info*>::iterator it = SysState->PerDevMPSInfo[devindex].begin(); 
				it != SysState->PerDevMPSInfo[devindex].end(); it++)
			{
				if((*it)->cap == cap && (*it)->dedup_num == dedup_num){
					found= true;
					break;
				}
			}
			if(!found){
				proxy_info* pPInfo = createNewProxy(devindex,cap,dedup_num,perdev_part_idx[devindex],pPInfo->type);
				perdev_part_idx[devindex]++;
				SysState->PerDevMPSInfo[devindex].push_back(pPInfo);
			}
		}
		
}

// following is a one-time intiation code
void GlobalScheduler::loadAllProxys(SysMonitor *SysState){
	for(int i=0; i < SysState->nGPUs; i++){
		for(auto pPInfo : SysState->PerDevMPSInfo[i]){
			// following is an async call
			loadModelstoProxy(pPInfo,SysState);
			// wait for 0.1 seconds(enough time to send load instruction to backend node)
			usleep(100*1000);
		}
	}
	for(int i=0; i < SysState->nGPUs; i++){
		for(auto pPInfo : SysState->PerDevMPSInfo[i]){
			if(!SysState->PerProxyTaskList[pPInfo].empty()) mProxyCtrl->waitProxy(pPInfo,RUNNING);
		}
	}
}

void GlobalScheduler::setupProxyTaskList(string specfile, SysMonitor *SysState){
#ifdef FRONTEND_DEBUG
    printf("[setup] setupProxyTaskList called for file: %s \n", specfile.c_str());
#endif 
		//throughput info to deliver to load balancer
		map<proxy_info*, vector<pair<int,double>>> mapping_trps;
		ProxyPartitionReader ppr = ProxyPartitionReader(specfile, SysState->nGPUs);
		vector<task_config> list = ppr.getAllTaskConfig();
		for(vector<task_config>::iterator it = list.begin(); it != list.end(); it++){
            proxy_info* ptemp=getProxyInfo(it->node_id,it->thread_cap,it->dedup_num,SysState);
			#ifdef DEBUG
				cout <<"[setup] name: " << it->name << ", batch_size: " << it->batch_size << endl;
			#endif
            if(it->name=="reserved"){
				ptemp->isReserved=true;
				continue;
			}
			ptemp->isSchedulable=true;
			pair<string, int> t(it->name,it->batch_size);
			SysState->PerProxyTaskList[ptemp].push_back(t);
			ptemp->duty_cycle = it->duty_cycle;      
			// for load balancer, we also get throughput
			// 对于复杂, 计算duty_cycle, 初始状态下是不计算inf的, 因为只有一个模型
			double duty_cycle = max(it->duty_cycle,getEstLatency(it->name,it->batch_size,it->thread_cap,ptemp->type));
			double trpt = it->batch_size * (1000 / it->duty_cycle);
			pair<int, double> t2(getModelID(it->name),trpt);
			mapping_trps[ptemp].push_back(t2);
		}
		setupLoadBalancer(mapping_trps);

#ifdef FRONTEND_DEBUG
    printf("[setup] perProxyTaskList configures as following: \n");
    for(int j=0; j<SysState->nGPUs; j++){
        for(unsigned int k=0; k<SysState->PerDevMPSInfo[j].size(); k++){
            proxy_info *pPInfo = SysState->PerDevMPSInfo[j][k];
            printf("proxy[%d,%d,%d]: \n", pPInfo->dev_id, pPInfo->cap,pPInfo->dedup_num);
            for(vector<pair<string,int>>::iterator it = SysState->PerProxyTaskList[pPInfo].begin(); it != SysState->PerProxyTaskList[pPInfo].end(); it++){
                printf("task: %s, batch: %d, duty_cycle: %lf \n", (*it).first.c_str(), (*it).second, pPInfo->duty_cycle);
            }
        }
    }
#endif
}
void GlobalScheduler::setupPerModelLatencyVec(SysMonitor *SysState){ // this initiates mutex locks and deques for proxy/model latency vectors
	#ifdef DEBUG
	printf("[SETUP] setting up per proxy/model mutex \n");
	#endif
	const int INIT_SKIP_CNT = 100;
	for(uint64_t i=0; i<SysState->nGPUs; i++){
		for(vector<proxy_info*>::iterator it = SysState->PerDevMPSInfo[i].begin(); it != SysState->PerDevMPSInfo[i].end(); it++){
			proxy_info *pPInfo = (*it);
			pPInfo->PerTaskLatencyVecMtx=new map<string,mutex*>();
			pPInfo->PerTaskLatencyVec=new map<string,deque<float>*>();
			pPInfo->PerTaskSkipCnt = new map<string, int>();
			for(vector<string>::iterator it = mNetNames.begin(); it != mNetNames.end(); it++){
				pPInfo->PerTaskLatencyVec->operator[](*it) = new deque<float>();
				pPInfo->PerTaskLatencyVecMtx->operator[](*it) = new mutex();
				pPInfo->PerTaskSkipCnt->operator[](*it) = INIT_SKIP_CNT;
			}
		}
	}
}

void GlobalScheduler::setupModelSLO(string name, int SLO)
{
        mPerModelSLO[name]=SLO;
#ifdef DEBUG
        printf("[SETUP] mPerModelSLO[%s]: %d \n", name.c_str(), SLO);
#endif
}

void GlobalScheduler::setMaxBatch(string name, string type, int max_batch)
{
    tuple<string, string> query = make_tuple(name, type);
    mMaxBatchTable[query]=max_batch;
}

void GlobalScheduler::setMaxDelay(string name, string type, int max_delay){
    tuple<string, string> query = make_tuple(name, type);
    mMaxDelayTable[query]=max_delay;

}


void GlobalScheduler::initPerModelLatEWMA(SysMonitor *SysState, int monitor_interval_ms, int query_interval_ms){
	for(auto modelname: mNetNames){	
		PerTaskLatUpdateMtx[modelname]=new mutex();
		SysState->PerModelLatestLat[modelname]=0;
		mPerTaskEWMALat[modelname]=make_shared<EWMA::EWMA>(monitor_interval_ms, query_interval_ms);
	}
}
void GlobalScheduler::initPerModelTrptEWMA(SysMonitor *SysState, int monitor_interval_ms, int query_interval_ms){
	for(auto modelname: mNetNames){
		PerTaskTrptUpdateMtx[modelname]=new mutex();
		SysState->PerModelFinCnt[modelname]=0;
		mPerTaskEWMATrpt[modelname]=make_shared<EWMA::EWMA>(monitor_interval_ms, query_interval_ms);
	}

}

void GlobalScheduler::initPerModelPartSum(SysMonitor *SysState, unordered_map<int,int> &sum_of_parts){
	for(auto gpu_ptr : mCurrState.vGPUList){
		for(auto node_ptr : gpu_ptr->vNodeList){
			for(auto task_ptr : node_ptr->vTaskList){
				int model_id = task_ptr->id;
				if(sum_of_parts.find(model_id) == sum_of_parts.end()){
					sum_of_parts[model_id]=node_ptr->resource_pntg;
				}
				else{
					sum_of_parts[model_id]+=node_ptr->resource_pntg;
				}
			}
		}
	}
	#ifdef ST_DEBUG
	cout << __func__ << ": all model's sum of part initated as following : "
	<<endl;
	for(unordered_map<int,int>::iterator it = sum_of_parts.begin(); it != sum_of_parts.end(); it++){
		cout <<"model id: " << it->first << " part:  " << it->second
		<<endl;
	}
	#endif

}

void GlobalScheduler::setupBackendProxyCtrls(){
    int i=0;
	bool first=true;
    for(auto pair :  NodeIDtoBackendDelegate){
		if(first){
        	mProxyCtrl = new BackendProxyCtrl();
			first=false;
		}
		mProxyCtrl->addBackendDelegate(pair.first, pair.second);
        #ifdef FRONTEND_DEBUG
        cout << "intiated BackendProxyCtrl for node : " << pair.first << endl;
        #endif
    }
}


void GlobalScheduler::setupLoadBalancer(map<proxy_info*, vector<pair<int,double>>> &mapping_trps){
	#ifdef LB_DEBUG
	for(auto pair_info : mapping_trps){
		for(auto item : pair_info.second){
			cout << __func__ <<": pushing <" << item.first << ", " << item.second << "> to " << proxy_info_to_string(pair_info.first)
			<< endl;
		}
	}
	#endif 


	if(mLoadBalancer.updateTable(mapping_trps)){
		cout << __func__ << ": ERROR when update load balancer table, exiting program"
		<<endl;
		exit(1);
	}
}