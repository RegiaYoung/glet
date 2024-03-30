#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <map>
#include <glog/logging.h>
#include <time.h>
#include <queue>
#include <vector>
#include <netinet/tcp.h>

#include "router.h"
#include "boost/program_options.hpp"
#include "socket.h"
#include "thread.h"
#include "request.h"
#include "batched_request.h"
#include "common_utils.h"
#include "input.h"
#include "proxy_ctrl.h"
#include "gpu_utils.h"

#define THREAD_PER_TASK 2
#define THREAD_PER_RESERVE 3 * THREAD_PER_TASK

using namespace std;
namespace po = boost::program_options;
SysMonitor ServerState; 
const int TOTAL_SENDTHREAD=8;
const int APP_THREAD=1;
extern bool READY_TO_EXEC;

FILE* pLogFile;
FILE* pAppLogFile;
ProxyCtrl *pProxyCtrl;
GPUUtil *gpGPUUtil;

bool EXIT;
bool finishedMPSProxyInit;
unordered_map<int,vector<uint64_t>> gInputDimMapping;
unordered_map<int,string> gInputTypeMapping;

po::variables_map ParseOpts(int argCount, char** argVar){
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")
    ("resource_dir, rd", po::value<string>()->default_value("../resource"), "directory which hold resource files" )
    ("backend_control_portno,bcp", po::value<int>()->default_value(8080),"Port to open backend control channel")
    ("backend_data_portno,bdp", po::value<int>()->default_value(8080),"Port to open backend data channel")
    ("gpu_idxs,gi", po::value<string>()->default_value("0-3"),"gpu index write as 'start idx'-'end idx'")
    ("ngpus,ng", po::value<int>()->default_value(4),"The number of gpus that are in the host")
//    ("ngpu,ng", po::value<int>()->default_value(4),"number of gpus in system to run service")
    ("nproxy,np", po::value<int>()->default_value(2),"number of proxys per gpu in system to run service")
    ("config_json, cj", po::value<string>()->default_value("config.json"), "json file which holds configuration for experiment" )
    ("proxy_json, pj", po::value<string>()->default_value("proxy_config.json"), "json file which holds configuration for models on proxy server" )
    ("proxy_script, ps", po::value<string>()->default_value("startProxy.sh"), "script name for starting proxy (default: startProxy.sh)" )
    ("full_proxy_dir, fpd", po::value<string>()->default_value("/home/scc/glet/resource/proxy"), "/FULL/PATH/TO/resource/proxy" );
    //("model_list", po::value<string>()->default_value("ModelList.txt"), "list of model and configuration of proxys" );
    po::variables_map vm;
  po::store(po::parse_command_line(argCount, argVar, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
          cout << desc << "\n";
          exit(1);
  }
  return vm;
}

// parse and return the total number of gpus
int parseGPUIdxs(map<int,int> &table_to_update, int &ngpus_to_update, string idxs_string){
  // parse gpu idxs
  std::stringstream test(idxs_string);
  std::string segment;
  std::vector<int> idxs;
  
  while(std::getline(test, segment, '-'))
  {
    idxs.push_back(stoi(segment));
  }
  assert(idxs.size()==2);
  assert(idxs[0] <= idxs[1]);
  ngpus_to_update=(idxs[1]-idxs[0]+1);
  int frontend_idx=0;
  for(int i = idxs[0]; i<=idxs[1]; i++){
    table_to_update[frontend_idx++]=i;
  }
   #ifdef BACKEND_DEBUG
   for (auto idx_pair : table_to_update){
    std::cout << "front index : " << idx_pair.first << " mapped to backend host index: " << idx_pair.second
    <<endl;
  }
  #endif
  
  return EXIT_SUCCESS;
}

void configureServerHW(po::variables_map &vm, SysMonitor &state){
  // setup some global & systemwide variables
  // if setting up gpu idx fails, deffinitely a bug in script or code
  if(parseGPUIdxs(state.FrontDevIDToHostDevID,state.nGPUs,vm["gpu_idxs"].as<string>() ) ) {
    exit(1);
  }
  state.nProxyPerGPU  = vm["nproxy"].as<int>();
  state.nHostGPUs = vm["ngpus"].as<int>();
  assert(state.nGPUs >= 0 && state.nProxyPerGPU>0);
}


int configureAppSpec(po::variables_map &vm, SysMonitor &state){
  configAppSpec(vm["config_json"].as<string>().c_str(), state, vm["resource_dir"].as<string>() );
  return EXIT_SUCCESS;
}


int receiveFrontendConnections(int socketfd){
  listen(socketfd, 128);
  LOG(ERROR) << "Backend is listening for connections on " << socketfd <<endl;
  //socket option to DISABLE delay
  const int iOptval = 1;
  while (true){
    pthread_t new_thread_id;
    int client_sock=accept(socketfd, (sockaddr*)0, (unsigned int*)0);
    if( client_sock == -1) {
      LOG(ERROR) << "Failed to accept.\n";
      continue;
    }              
    if(setsockopt(client_sock, IPPROTO_TCP, TCP_NODELAY, (char *) &iOptval, sizeof(int))==-1)
    {
        LOG(ERROR) << "set socket option failed" ;
        continue;       
    }
    new_thread_id = initBackendDataThread(client_sock);
  }
  return 0;

}

void intitiatePerformanceLogs(){
  pLogFile=fopen("log.txt","w");
  fprintf(pLogFile,"timestamp,task,TaskID,ReqID,deviceID,QUEUEING_DELAY,WAIT_EXEC,BATCH_LAT,EXEC,TO CMPQ,CMPQ_WRITE\n");
  fflush(pLogFile);
  pAppLogFile=fopen("Applog.txt","w");
  fprintf(pAppLogFile,"timestamp,task,TaskID,REQ_TO_EXEC,EXEC\n");
  fflush(pAppLogFile);
}

proxy_info* createNewProxy(int dev_id, int cap, int dedup_num, int part_index){
	proxy_info* pPInfo = new proxy_info();
	pPInfo->dev_id = dev_id;
	pPInfo->cap = cap;
	pPInfo->dedup_num=dedup_num;
	pPInfo->duty_cycle=0; // THIS WILL BE OVERWRITTEN IN setupProxyTaskList
	pPInfo->partition_num = part_index;
	pPInfo->isBooted=false;
	pPInfo->isConnected=false;
	pPInfo->isSetup=false;
	#ifdef DEBUG
	printf("[createNewProxy] create new proxy: [%d,%d,%d][%d] \n",dev_id, cap, dedup_num, part_index );
	#endif
	return pPInfo;
}

void createProxyInfos(SysMonitor *SysState, vector<int> parts, int node_id=0){
	//make proxy_info for every possible partitoin on every GPU
	map<int,int> perdev_part_idx;
	for(int i =0; i < SysState->nGPUs; i++) perdev_part_idx[i]=0;
	for(int devindex =0; devindex < SysState->nGPUs; devindex++ ){
		for(int part: parts){
			int dedup_num=0;
      int backend_idx = SysState->FrontDevIDToHostDevID[devindex];
			proxy_info* pPInfo = createNewProxy(backend_idx,part,dedup_num,perdev_part_idx[devindex]);
			pPInfo->node_id=node_id;
			SysState->PerDevMPSInfo[backend_idx].push_back(pPInfo);			
			perdev_part_idx[devindex]++;
			// if partition is 50, make another proxy, with dedup_num=1
			if(part == 50){ 
				dedup_num++;
				proxy_info* pPInfo = createNewProxy(backend_idx,part,dedup_num,perdev_part_idx[devindex]);
			  pPInfo->node_id=node_id;
				SysState->PerDevMPSInfo[backend_idx].push_back(pPInfo);			
				perdev_part_idx[devindex]++;
			}
		}
	}
}


int main(int argc, char* argv[]) {
  // Main thread for the server
  po::variables_map vm = ParseOpts(argc, argv);
  configureServerHW(vm,ServerState);
  cout << "Backend connected to " << ServerState.nGPUs << endl;
  // for now we hard code the available sizes
  vector<int> AVAIL_SIZES={20,40,50,60,80,100};
  pProxyCtrl = new ProxyCtrl(/*clear_memory=*/true);

  pProxyCtrl->setProxyConstants(vm["full_proxy_dir"].as<string>(), vm["proxy_script"].as<string>());
  gpGPUUtil = new GPUUtil();
  createProxyInfos(&ServerState,AVAIL_SIZES,/*dummy node_id=*/0);
  readInputDimsJsonFile(vm["proxy_json"].as<string>().c_str(), ServerState.PerModeltoIDMapping,gInputDimMapping);
  unordered_map<int,string> dummy;
  readInputTypesJSONFile(vm["proxy_json"].as<string>().c_str(), gInputTypeMapping,dummy);
  configureAppSpec(vm,ServerState);
  // ----------backend code from server.cpp -----------
  #ifdef BACKEND_DEBUG
  std::cout << "initaiting control port" << std::endl;
  #endif
  initBackendControlThread(SERVER_init(vm["backend_control_portno"].as<int>()));
  #ifdef BACKEND_DEBUG
  std::cout << "completed contorl port" << std::endl;
  std::cout << "initaiting data port "<< std::endl;
  #endif

  int ret= receiveFrontendConnections(SERVER_init(vm["backend_data_portno"].as<int>()));
  #ifdef BACKEND_DEBUG
  std::cout << "completed data port " << std::endl;
  std::cout << "[receiveFrontendConnections] ret val : " << ret << std::endl; 
  #endif
  return EXIT_SUCCESS;
}
