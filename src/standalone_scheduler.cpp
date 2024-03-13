#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <assert.h>
#include "config.h"
#include "boost/program_options.hpp"
#include "scheduler_incremental.h"
#include "scheduler_utils.h"

using namespace std;
namespace po = boost::program_options;

vector<int> AVAIL_PARTS;

po::variables_map parse_opts(int ac, char** av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")
  ("resource_dir,rd", po::value<string>()->default_value("../resource"),"directory which hold resource files")
  ("task_config,tc", po::value<string>()->default_value("tasks.csv"),"csv file with each task's SLO and ")
  ("sim_config", po::value<string>()->default_value("sim_config.json"),"json file which hold simulation configurations")
  ("output", po::value<string>()->default_value("ModelList.txt"),"txt file which hold simulation results")
  ("mem_config", po::value<string>()->default_value("mem-config.json"),"json file which holds the amount of memory each model+input uses")
  ("full_search", po::value<bool>()->default_value(false),"flag: conduct full search or not")
  ("proxy_config", po::value<string>()->default_value("proxy_config.json"),"json file which holds info input data")
  ("device_config", po::value<string>()->default_value("device-config.json"),"json file which holds per device type data");

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
          cout << desc << "\n";
          exit(1);
  }
  return vm;
}

//this function returns
void ReturnNodeList(vector<vector<Node>> *comb, vector<int> &availpart,vector<Node> temp ,int gpuid){
    if (gpuid ==0){
        comb->push_back(temp);
        return;
    }    
    for(vector<int>::iterator it1 = availpart.begin(); it1 != availpart.end(); it1++){
        int a1 = *it1;
        int b1 = 100 -a1;          
        Node t1,t2;
        t1.resource_pntg=a1;
        t2.resource_pntg=b1;
        t1.id = gpuid;
        t2.id = gpuid;
        t1.dedup_num=0;
        if(t1.resource_pntg == t2.resource_pntg) t2.dedup_num=1;
        else t2.dedup_num=0;
        if(t1.resource_pntg != 0 ) temp.push_back(t1);
        if(t2.resource_pntg != 0 ) temp.push_back(t2);            
        ReturnNodeList(comb,availpart,temp,gpuid-1);
        if(t1.resource_pntg != 0 ) temp.pop_back();
        if(t2.resource_pntg != 0 ) temp.pop_back();
    }    
}

void fillPossibleCases2(vector<vector<Node>> *pVec, int ngpu){
    vector<Node> temp;
    ReturnNodeList(pVec,AVAIL_PARTS,temp,ngpu);
}

static uint64_t getCurNs(){
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        uint64_t t = ts.tv_sec * 1000 * 1000 * 1000 + ts.tv_nsec;
        return t;
}

void writeSchedulingResults(string filename, SimState *simulator, squishy_bin_packing::BaseScheduler &sched){
    ProxyPartitionWriter ppw = ProxyPartitionWriter(filename, simulator->vGPUList.size());
    for(unsigned int i =0; i< simulator->vGPUList.size(); i++){
        for(auto it : simulator->vGPUList[i]->vNodeList){

            if(it->vTaskList.empty()){
              task_config new_config;
              new_config.node_id = it->id;
              new_config.thread_cap = it->resource_pntg;
              new_config.dedup_num = it->dedup_num;
              new_config.duty_cycle= 0;
              new_config.name="reserved";
              new_config.batch_size=sched.GetMaxBatchSize();
              ppw.addResults(new_config);
            }
            else {
            for(auto it2 : it->vTaskList){
              task_config new_config;
              new_config.node_id = it->id;
              new_config.thread_cap = it->resource_pntg;
              new_config.dedup_num = it->dedup_num;
              new_config.duty_cycle= it->duty_cycle;
              new_config.name=sched.GetModelName(it2->id);
              new_config.batch_size=it2->batch_size;
              ppw.addResults(new_config);
            }
        }
      }

  }
    ppw.writeResults();

}

int main(int argc, char* argv[])
{

    po::variables_map vm = parse_opts(argc, argv);
    vector<string> files={"1_28_28.txt", "3_224_224.txt", "3_300_300.txt"};
    squishy_bin_packing::IncrementalScheduler  SBP;
    bool success=  SBP.InitializeScheduler(vm["sim_config"].as<string>(),\
                 vm["mem_config"].as<string>(),
                 vm["device_config"].as<string>(),
                 vm["resource_dir"].as<string>(),
                 files);
    if(!success){
      cout << "Failed to initialize scheduler" << endl;
      exit(1);
    }
    #ifdef CHECK_NETBW
    SBP.setupNetworkChecker(vm["resource_dir"].as<string>()+"/"+vm["proxy_config"].as<string>());
    #endif
     vector<Task> task_list;
    if(SBP.GetUseParts()) AVAIL_PARTS={50,60,80,100}; // other part will be 100-20, 100-40, 100-50 and so on
    //if(SBP.GetUseParts()) AVAIL_PARTS={50,}; // other part will be 100-20, 100-40, 100-50 and so on


    else AVAIL_PARTS={100}; // other part will be 100-20, 100-40, 100-50 and so on
    SimState simulator;
    SimState final_output;
    SBP.SetupAvailParts(AVAIL_PARTS);

    string output_file = vm["output"].as<string>();
    
    vector<SimState> sim_list;
    SimState best_sim;
    SBP.SetupTasks(vm["task_config"].as<string>(), &task_list);
    vector<int> per_dev_mem;
    // device memory is initiated by SBP
    SBP.InitiateDevs(simulator,SBP.GetMaxGPUs());
    SBP.InitDevMems(simulator);

    // below is for debugging memory usage and related functions
    //vector<int> per_dev_used_mem = {5219, 6909,7877,9193};
    //vector<int> per_dev_used_mem = {5222};
    //SBP.UpdateDevMemUsage(per_dev_used_mem, simulator);

    for (auto task : task_list)
    {
      SBP.InitSaturateTrp(task);
    }
    uint64_t sbp_start = getCurNs();
    if(!vm["full_search"].as<bool>()){
      if (!SBP.SquishyBinPacking(&task_list, simulator, final_output,true))
      {
        sim_list.push_back(final_output);
        cout << "[main] Success" << endl;
      }
      else{
            cout <<"[main] failed "<< endl;
      }
 
      if(sim_list.empty()){
        printf("Received EMPTY list \n");
        FILE *r = fopen(output_file.c_str(), "w");
        fprintf(r,"EMPTY");
          return 1;
      }
    }
    else{
      //exhaustively search for best case
      vector<vector<Node>> possible_cases;
      fillPossibleCases2(&possible_cases, SBP.GetMaxGPUs()); 
      
      for(auto vec : possible_cases){
          SimState input,output;
          task_list.clear();
          SBP.SetupTasks(vm["task_config"].as<string>(),&task_list);
          SBP.ResetScheduler(input,vec);
          SBP.InitDevMems(input);
#ifdef SCHED_DEBUG
	  printf("input to be scheduled : \n");
	  PrintResults(input);
#endif
          if(!SBP.SquishyBinPacking(&task_list, input,output,false)){
              cout<< "[main] Success" << endl;
              sim_list.push_back(output);
              PrintResults(output);
              break;
          }
          else{
              cout <<"[main] failed "<< endl;
          }
 
        } // for each possible case   
   }
   uint64_t sbp_end=getCurNs();
   printf("computation time(ms): %lf \n",double(sbp_end - sbp_start)/1000000);
   if(sim_list.empty()){
        printf("Received EMPTY list \n");
        FILE *r = fopen(output_file.c_str(), "w");
        fprintf(r,"EMPTY");
        return 0;
   }
   best_sim = sim_list[0];
   PrintResults(best_sim);
   
   //filters out scheduling attempts that requires too much bandwidth, defined in network_limit.h
   #ifdef CHECK_NETBW
   /*
    if(!SBP.inspectNetworkBW(best_sim)){
     printf("Failed Network Capacity Test \n");
     FILE *r = fopen(output_file.c_str(), "w");
     fprintf(r,"EMPTY");
    return 0;
   }
   */
   #endif
   
   writeSchedulingResults(output_file,&best_sim,SBP);
   return 0;
}
