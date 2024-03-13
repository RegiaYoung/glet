#include "scheduler_base.h"
#include "scheduler_incremental.h"
#include "scheduler_utils.h"
#include "network_limit.h"
#include <assert.h>
#include <glog/logging.h>
#include <algorithm>
#include <iostream>
#include <math.h>
#include "config.h"


namespace squishy_bin_packing{
const int TRP_SLACK=15;
IncrementalScheduler::IncrementalScheduler(){}
IncrementalScheduler::~IncrementalScheduler(){}

bool IncrementalScheduler::SquishyBinPacking(vector<Task> *task_list, SimState &prev_output, SimState &new_output, bool allow_repart){
    if(allow_repart) _REPARTITION=true;
    else _REPARTITION=false;
    if(IncrementalScehduling(*task_list, prev_output)){
      return EXIT_FAILURE;
    }
    CopyToOutput(prev_output,new_output);
    return EXIT_SUCCESS;
}

float min(const float a, const float b){
    return (a<=b) ? a : b;
}

bool CheckContain(const int model_num, const NodePtr &node_ptr){
  for(auto task_ptr : node_ptr->vTaskList){
      if(model_num == task_ptr->id) return 1;
  }
  return 0;
}

bool cmp_nodeptr_occu_dsc(const NodePtr &a, const NodePtr &b){
  return a->occupancy > b->occupancy;
}

// Time shares nodes, if ts_node_list is not empty this function will only consider nodes that are in the vector
bool IncrementalScheduler::AllocateTimeShare(Task &task, SimState &sim, vector<NodePtr> &ts_node_list){
 #ifdef SCHED_DEBUG
  printf("[AllocateTimeShare] called for task id: %d , rate: %d \n",task.id, task.request_rate );
 #endif
  bool checkfirst=true;
  if(ts_node_list.empty()) checkfirst=false;
  vector<NodePtr> candidates;
    for(auto gpu_ptr : sim.vGPUList){
      //check whether there is room for task in GPU first
      for(auto node_ptr : gpu_ptr->vNodeList)
      {
      #ifdef CHECK_MEM
        if(!DoesFitMemLimit(gpu_ptr,task.id,node_ptr)) continue;
      #endif
        if(checkfirst){ // search for node in ts_node_list and add pointer if and only if the node is in list
          bool found = false;
          for(auto node_ptr2 : ts_node_list){
            if(node_ptr->id == node_ptr2->id && node_ptr->dedup_num == node_ptr2->dedup_num && node_ptr->resource_pntg == node_ptr2->resource_pntg){
              found=true;
              break;
            }
          }
          if(found) candidates.push_back(node_ptr);
        }
        else{
            if(node_ptr->occupancy < 1 && node_ptr->vTaskList.size() < _MODEL_LIMIT) candidates.push_back(node_ptr);
        }
      }
    }
  
  sort(candidates.begin(), candidates.end(), cmp_nodeptr_occu_dsc);
  while(task.request_rate + task.additional_rate> TRP_SLACK){
    bool found = false;
     for(auto node_ptr : candidates)
    {
        if(CheckContain(task.id, node_ptr)) continue;
        if(node_ptr->occupancy >= 1.0) continue;
        #ifdef SCHED_DEBUG
          printf("[AllocateTimeShare] occupancy: %lf \n" , node_ptr->occupancy);
        #endif
         int max_batch = int((node_ptr->duty_cycle) * (task.request_rate / 1000.0));
        #ifdef SCHED_DEBUG
          printf("[AllocateTimeShare] max batch: %d for duty cycle: %lf, rate: %d \n", max_batch,node_ptr->duty_cycle, task.request_rate);
        #endif 
        if(max_batch==0) continue;
         
        int local_batch_size;
        int duty_cycle;
        float latency;
     //get possible maximum batch size, (which leads to maximum througput)
        if(max_batch > _MAX_BATCH) max_batch=_MAX_BATCH;
        for(local_batch_size = max_batch; local_batch_size > 0; --local_batch_size){
            duty_cycle = local_batch_size * 1000.0  / (task.request_rate);
            latency = GetLatency(node_ptr->type,task.id,local_batch_size,node_ptr, sim);
            if(duty_cycle + latency < task.SLO && (latency / duty_cycle) < (1.0 - node_ptr->occupancy)){
              break;
            }
            
         }
         if(duty_cycle > node_ptr->duty_cycle) continue;
         if(local_batch_size==0) continue;

        #ifdef SCHED_DEBUG
         printf("[AllocateTimeShare] duty cycle before: %lf, after : %d, batch_size: %d \n", node_ptr->duty_cycle, duty_cycle, local_batch_size);
         printf("[AllocateTineShare] latency: %lf ms , occupancy: %lf \n", latency,latency/duty_cycle);
        #endif

          vector<int> SLOs;
          float latency_sum=0;
          bool skip=false;
          //first gather latency and SLO of new task
          latency_sum += latency;
          SLOs.push_back(task.SLO);
        #ifdef SCHED_DEBUG
         printf("[AllocateTimeShare] latency_sum: %lf, duty_cycle : %d, batch size: %d \n",latency_sum,duty_cycle, local_batch_size);
        #endif
          // then gather latency and SLO of old tasks 
          for(auto task_ptr : node_ptr->vTaskList){
              int batch_size = int(duty_cycle / (1000.0 / task_ptr->request_rate));
              if(batch_size==0) batch_size=1;
              latency_sum += GetLatency(node_ptr->type,task_ptr->id, batch_size, node_ptr, sim);
              SLOs.push_back(task_ptr->SLO);
        #ifdef SCHED_DEBUG
         printf("[AllocateTimeShare] latency_sum: %lf, duty_cycle : %d, batch_size: %d \n",latency_sum,duty_cycle, batch_size);
        #endif
          }
          if(latency_sum / duty_cycle > 1.0) continue;
           if(_CHECK_LATENCY){
               latency_sum+=duty_cycle;
              for(auto slo : SLOs)
              {
                if(latency_sum > slo) {
                  skip=true;
                  #ifdef SCHED_DEBUG
                    printf("[AllocateTineShare] latency sum %lf exceeded slo: %d \n", latency_sum, slo);
                  #endif
                }
              }
          }
          if(skip) continue;
        //update node
         max_batch=local_batch_size;
         float local_throughput = max_batch * (1000.0 / duty_cycle);
         node_ptr->vTaskList.push_back(CreateNewTaskPtr(task.id,task.request_rate,task.SLO,max_batch,local_throughput));
        // update remaining request rate left for task
         task.request_rate =  (task.request_rate - local_throughput >0)  ? task.request_rate - local_throughput : 0;
         #ifdef CHECK_MEM
         AddGPUMemUsage(sim.vGPUList[node_ptr->id],task.id,node_ptr);
         #endif
         float new_occupancy=0.0;
         for(auto task_ptr : node_ptr->vTaskList){
           if(task_ptr->id == task.id){
              new_occupancy += GetLatency(node_ptr->type,task_ptr->id,task_ptr->batch_size,node_ptr,sim) / duty_cycle;
              #ifdef SCHED_DEBUG
              printf("[AllocateTimeShare] new_occupancy: %lf , after adding %d \n",new_occupancy, task_ptr->id); 
              #endif
           }
           else{
            task_ptr->batch_size = int(duty_cycle / (1000.0 / task_ptr->request_rate));
            if(task_ptr->batch_size==0) task_ptr->batch_size=1;
            if(task_ptr->batch_size>_MAX_BATCH) task_ptr->batch_size=_MAX_BATCH;
            task_ptr->throughput = (task_ptr->batch_size * 1000.0) / duty_cycle;
            new_occupancy += GetLatency(node_ptr->type,task_ptr->id, task_ptr->batch_size, node_ptr,sim) / duty_cycle;
              #ifdef SCHED_DEBUG
              printf("[AllocateTimeShare] new_occupancy: %lf , after adding %d \n",new_occupancy, task_ptr->id); 
              #endif
           }
         }
         node_ptr->duty_cycle=duty_cycle;
         node_ptr->occupancy=new_occupancy;
         found=true;
        #ifdef SCHED_DEBUG
         node_ptr->id;
          printf("[AllocateTineShare] SUCCESS task : %d allocated to [%d,%d,%d] with trp: %lf, ending occupancy: %lf \n", task.id, \
          node_ptr->id, node_ptr->resource_pntg, node_ptr->dedup_num,\
          local_throughput,\
          node_ptr->occupancy);
        #endif
         break;
    } // for: candidates
    if(!found) return EXIT_FAILURE; // faied to allocate with time sharing 
    #ifdef SCHED_DEBUG
    printf("[AllocateTimeShare] remaining rate: %d \n",task.request_rate);
    #endif
  } // tsak.request_rate  
  return EXIT_SUCCESS;
}


bool cmp_task_dsc(const Task &a, const Task &b){
    return a.request_rate * a.SLO < b.request_rate * b.SLO || 
        ((a.request_rate * a.SLO == b.request_rate *b.SLO)  && (a.id> b.id));
}

bool IncrementalScheduler::ElasticPartitioning(vector<Task> &session, SimState &decision){
   sort(session.begin(), session.end(),cmp_task_dsc);
   for(auto task : session){
       // check if we have an available saturate table for task
      if (_per_model_sat_table.find(task.id) == _per_model_sat_table.end())
      {
        //if not, setup a table
        InitSaturateTrp(task);
      }
      if(AddModeltoSchedule(task, decision)){
          #ifdef SCHED_DEBUG
          cout << "[IncrementalScheduling] adding model failed!" << endl;
          #endif
          return EXIT_FAILURE;
      }
      
    #ifdef SCHED_DEBUG
      PrintResults(decision);
    #endif
  }
 
}

bool IncrementalScheduler::IncrementalScehduling(vector<Task> &session, SimState &decision){
  if(decision.vGPUList.empty()){
    InitiateDevs(decision,_MAX_GPU);
  }
  map<int,float> task_to_org_rate_mapping;
  map<int,float> task_to_intf_retry_flag;
  if(_USE_INTERFERENCE){
    for(auto task : session){
      task_to_intf_retry_flag[task.id]=false;
    }
  }
  bool good_to_go=false;
  while(!good_to_go){
    task_to_org_rate_mapping.clear();
    for(auto task : session){
      task_to_org_rate_mapping[task.id]=task.request_rate;
    }
    if(ElasticPartitioning(session,decision)){
      return EXIT_FAILURE;
    }
    good_to_go=true;
    //check how much leftover requests were  stored after adjusting batches
    
    if(_USE_INTERFERENCE){
        // 1. addup all throughputs for all tasks
        map<int,float> task_to_trpt;
        for(auto gpu_ptr : decision.vGPUList){
            for(auto node_ptr : gpu_ptr->vNodeList){
              for(auto task_ptr : node_ptr->vTaskList){
                task_to_trpt[task_ptr->id]+=task_ptr->throughput;
              }
            }
        }
        vector<Task> new_session;
        for(auto task:  session){
          const float INTF_THRESHOLD=0.93;
          #ifdef SCHED_DEBUG
          cout << "[IncrementalScheduling]: task_id: " << task.id << " task_trpt: "<<  task_to_trpt[task.id] << " task_org_rate: " << task_to_org_rate_mapping[task.id]
          <<endl;
          #endif
          if(task_to_trpt[task.id] < task_to_org_rate_mapping[task.id] * INTF_THRESHOLD){
              if(task_to_intf_retry_flag[task.id]) return EXIT_FAILURE;
              #ifdef SCHED_DEBUG
               cout << "[IncrementalScheduling]: task_id: " << task.id << " remaining rate: " << task_to_org_rate_mapping[task.id] - task_to_trpt[task.id]
              <<endl;    
              #endif
              task_to_intf_retry_flag[task.id]=true;
              good_to_go=false;
              // create new task in new task vector
              Task new_task;
              new_task.SLO=task.SLO;
              new_task.id=task.id;
              //new_task.request_rate=task_to_org_rate_mapping[task.id]*INTF_THRESHOLD;
              new_task.request_rate=task_to_org_rate_mapping[task.id];
              new_session.push_back(new_task);
          }
        }
        session.clear();
        CopySession(new_session,session);
    }// _USE_INTERFERENCE
    
  }// good_to_go
   
   // try scheduling without tightening batch size
   // uncomment this if tightening deems neccessary 
  /*  
  for(auto task : session){
      ResidueTightening(task, decision);
   }
   */
  return EXIT_SUCCESS;
}

int IncrementalScheduler::GetMinPart(string device, Task task, const NodePtr node_ptr, int &residue_rate ,int &result_batch){
  int max_part = 200;
  int given_pntg;
  int given_id;
  SimState dummy_sim;
  if(node_ptr == NULL){
      given_id=0;
      given_pntg=100;
  }
  else{
      given_id=node_ptr->id;
      given_pntg=node_ptr->resource_pntg;
  }
  #ifdef SCHED_DEBUG
  cout << "[GetMinPart] model_id: " << task.id << " given_pntg: " << given_pntg <<endl;
  #endif
  int local_max_part = min(_AVAIL_PARTS.front(),given_pntg);
  #ifdef SCHED_DEBUG
    cout << "[GetMinPart]  local_max_part: "  << local_max_part <<endl;
  #endif

  Node temp_node;
  NodePtr temp_node_ptr=make_shared<Node>(temp_node);
  temp_node_ptr->id=given_id;
  temp_node_ptr->resource_pntg=local_max_part;
  temp_node_ptr->type = node_ptr->type;
  
  int max_batch = GetMaxBatch(task,temp_node_ptr,dummy_sim,residue_rate,true,false);
  #ifdef SCHED_DEBUG
    cout << "[GetMinPart]  max_batch "  << max_batch <<endl;
  #endif
  if(!max_batch) return 0;
  
  assert(_MIN_BATCH <= max_batch && max_batch <= _MAX_BATCH);
  int prev_part=local_max_part;
  for(auto part : _AVAIL_PARTS){ // starting from highest partition
        if(part > local_max_part) continue;
        float latency = GetLatency(device, task.id,max_batch,part);
        float duty_cycle = max_batch * 1000.0 / residue_rate;
        #ifdef SCHED_DEBUG
        cout << "[GetMinPart] part: " << part << "latency: " << latency << " duty_cycle: " << duty_cycle << " SLO: "<< task.SLO << endl;
        #endif
        if(duty_cycle < latency){
        
          if(part == _AVAIL_PARTS[0]){
            prev_part=part;
          }
          break; 
        }
        if(task.SLO < latency + duty_cycle){
          break;
        } 
        prev_part=part;
        #ifdef SCHED_DEBUG
        cout << "prev_part: " << prev_part << endl;
        #endif
    
  }
  result_batch=max_batch;
  max_part=prev_part;
  #ifdef SCHED_DEBUG
  printf("[GetMinPart] min_part: %d, for task id: %d \n", max_part,task.id);
  #endif
  return max_part;
}

// receives task, and resource percentage the task will run as input and stores the maximum batch size it can support
// returns maximum throughput for that throughput
float IncrementalScheduler::MaxSaturateTrp(const Task &task, int &output_batch, const int resource_pntg, string type){

  bool found=false;
  float trp;
  for(auto entry : *_per_model_sat_table[task.id]){
      if(resource_pntg == entry.part && type == entry.type)
      {
        found=true;
        trp=entry.sat_trp;
        output_batch=entry.max_batch;
      }
  }
  assert(found); // if this is not found we have a problem
  return trp;
}

// stores value for saturating throughput of each available partition
void IncrementalScheduler::InitSaturateTrp(Task &task){
  #ifdef SCHED_DEBUG
    cout << "[InitSaturateTrp] Called for task  " << task.id << endl;
  #endif
    vector<SatTrpEntry> *new_table = new vector<SatTrpEntry>();
    Node temp_node;
    SimState dummy_sim;
    NodePtr temp_node_ptr=make_shared<Node>(temp_node);
    for(auto type_num_pair : _type_to_num_of_type_table){
      string type= type_num_pair.first;
      for(auto part : _AVAIL_PARTS){
        temp_node_ptr->resource_pntg=part;
        temp_node_ptr->type=type;
        int batch_size = GetMaxBatch(task,temp_node_ptr,dummy_sim,task.request_rate,/*is_residue=*/false,/*interfereence=*/false);
        // check if this is because 2*L(b=1) > SLO
        if (batch_size ==0){
          float latency = GetLatency(type,task.id,1,temp_node_ptr->resource_pntg);
          if ( 2*latency > task.SLO) {
            batch_size=1;
  #ifdef SCHED_DEBUG
        cout << "[InitSaturateTrp] 2*L(b=1) > SLO, fixing batch size to 1 "<< endl;
    #endif
  
          }
          else{
              cout << "[InitSaturateTrp] cannot setup SaturateTable for task : " << task.id << endl;
              return;
          }
        }
        float latency = GetLatency(type,task.id,batch_size,temp_node_ptr->resource_pntg);
        float trp = batch_size * 1000.0 / latency;
        SatTrpEntry new_entry;
        new_entry.max_batch=batch_size;
        new_entry.part=part;
        new_entry.sat_trp=trp;
        new_entry.type=type;
        new_table->push_back(new_entry);
    #ifdef SCHED_DEBUG
      cout << "[InitSaturateTrp] setted up resource pntg: " << new_entry.part <<", as trp:  "<< new_entry.sat_trp<< endl;
  #endif
      }
    }
    _per_model_sat_table[task.id]=new_table;
}

void IncrementalScheduler::SetSelfTuning(bool use){
    _SELF_TUNING=use;
    _USE_INTERFERENCE=false;
}


void IncrementalScheduler::EstimateTrp(string device, Task &task, int rate, vector<NodePtr> &output_vec, const int MAX_PART){
  if(rate <=TRP_SLACK) return; // yes... if the rate is just too small then return 
  #ifdef SCHED_DEBUG
  printf("[EstimateTrp] rate: %d called for model id: %d  \n",rate,task.id );
  #endif 
  int temp_batch;
  int max_part;
  if(_SELF_TUNING){
    GetEstimateTrpST(device, task, rate, output_vec,MAX_PART);
    return;
  }
  else
  max_part = GetMaxReturnPart(task,device);
  float limit = MaxSaturateTrp(task, temp_batch,max_part, device);
  #ifdef SCHED_DEBUG
  cout << "max_part: " << max_part << " limit: "<<limit << " rate: " << rate << endl;
  #endif
  if(limit < rate){
      //allocate saturate node and residue node 
      int sat_part=0;
      float sat_trp=0;
      int sat_batch=0;

      sat_part=max_part;
      sat_trp=MaxSaturateTrp(task,sat_batch,sat_part,device);

      // make saturate node and allocate task
      NodePtr temp_node_ptr = MakeEmptyNode(output_vec.size(),sat_part,device); 
      TaskPtr temp_task_ptr = CreateNewTaskPtr(task.id,rate,task.SLO,sat_batch,sat_trp); 
      temp_node_ptr->occupancy=1;
      temp_node_ptr->type=device;
      temp_node_ptr->vTaskList.push_back(temp_task_ptr);
      temp_node_ptr->duty_cycle = GetLatency(device, task.id,sat_batch,temp_node_ptr->resource_pntg);
      output_vec.push_back(temp_node_ptr);

      // recursive call 
      EstimateTrp(device,task,rate-sat_trp, output_vec, MAX_PART);
  }
  else{ // this is it!! this time allocate residue node and return
      //NodePtr temp_node_ptr = MakeEmptyNode(output_vec.size(),100);
      NodePtr temp_node_ptr = MakeEmptyNode(output_vec.size(),max_part,device);
      int max_batch;
      int min_part = GetMinPart(device, task, temp_node_ptr,rate,max_batch);
      temp_node_ptr->resource_pntg=min_part;
      float latency = GetLatency(device, task.id,max_batch,min_part);
      #ifdef SCHED_DEBUG
      printf("[EstimateTrp]residue node- latency: %lf, batch_size: %d, rate: %d, part: %d  \n", latency, max_batch, rate, min_part );
      #endif
      temp_node_ptr->duty_cycle= max(max_batch * (float(1000.0)  / rate), latency);
      temp_node_ptr->type = device;
      float trp = (max_batch * 1000.0) / temp_node_ptr->duty_cycle;
      TaskPtr temp_task_ptr = CreateNewTaskPtr(task.id,rate,task.SLO,max_batch,trp);
      temp_node_ptr->occupancy= latency / temp_node_ptr->duty_cycle;
      temp_node_ptr->vTaskList.push_back(temp_task_ptr);
      output_vec.push_back(temp_node_ptr);
  }
}

void IncrementalScheduler::GetEstimate(Task &task, vector<NodePtr> &output_vec, const int MAX_PART){
    #ifdef SCHED_DEBUG
    cout << "[GetEstimate] function called for task id : " << task.id<< "rate: " << task.request_rate<< endl;
    cout << "[GetEstimate] max_part : " << MAX_PART<< endl;
    #endif
    map<string,int> _type_parts_table;
    string _most_cost_effective_type;
    int _min_parts = MAX_PART * GetMaxGPUs();

    for(auto type_num_pair : _type_to_num_of_type_table){
      output_vec.clear();
      EstimateTrp(type_num_pair.first, task,task.request_rate,output_vec, MAX_PART);
      int sum_of_parts=0;
      for(auto item : output_vec)
      {
        sum_of_parts += item->resource_pntg;
        if(sum_of_parts < _min_parts){
          _min_parts = sum_of_parts;
          _most_cost_effective_type=type_num_pair.first;
        } 
      }
    }
  output_vec.clear();
  EstimateTrp(_most_cost_effective_type, task,task.request_rate,output_vec, MAX_PART);
}


bool IncrementalScheduler::CheckForInterference(string device, NodePtr the_node, NodePtr the_real_node, SimState &sim){
    NodePtr neighbor_node;
    #ifdef SCHED_DEBUG
    cout << __func__ << "the real node : [" << the_real_node->id << "," <<the_real_node->resource_pntg << "]" << endl;
    #endif

    if(the_real_node->resource_pntg==100) return false;
    if(!GetOtherNode(the_real_node,neighbor_node,sim)){
        for(auto task_ptr : neighbor_node->vTaskList){
            float pure_latency = _name_to_dev_perf_model_table[device].getLatency(GetModelName(task_ptr->id),task_ptr->batch_size,neighbor_node->resource_pntg);
            // form a temporal GPU for getting interference;
            GPU temp_gpu;
            GPUPtr temp_gpu_ptr = make_shared<GPU>(temp_gpu);
            temp_gpu_ptr->vNodeList.push_back(neighbor_node);
            NodePtr temp_node_ptr= MakeEmptyNode(neighbor_node->id,the_real_node->resource_pntg,the_real_node->type);
            for(auto task_ptr : the_node->vTaskList) temp_node_ptr->vTaskList.push_back(task_ptr);

            double interference = GetInterference(device, task_ptr->id,task_ptr->batch_size,neighbor_node,temp_gpu_ptr);
            float duty_cycle = neighbor_node->duty_cycle;
            float latency = pure_latency *(interference-1.0)+ GetBatchLatency(GetModelName(task_ptr->id),task_ptr->batch_size) + duty_cycle;
  
            #ifdef SCHED_DEBUG
            cout << "[findBestFit_other_node] resource_pntg: " << neighbor_node->resource_pntg << endl;
            cout << "[findBestFit_other_node] latency: " << latency << " duty_cycle: " << duty_cycle << endl;
            cout << "[findBestFit_other_node] SLO: " << task_ptr->SLO << endl;
            #endif
            
            if(latency + duty_cycle> task_ptr->SLO) return true;
          
        }
    }
    return false;
}

bool IncrementalScheduler::FindBestFit(SimState &input_sim,NodePtr &input,vector<NodePtr> &exclude_vec,NodePtr &output){
  int min_diff = 200;
  NodePtr min_ptr;
  string type = input->type;
  for (auto gpu_ptr : input_sim.vGPUList)
  {
    if (gpu_ptr->TYPE != type)
      continue;
    for (auto node_ptr : gpu_ptr->vNodeList)
    {
      if (node_ptr->resource_pntg < input->resource_pntg || !node_ptr->vTaskList.empty())
        continue;
      vector<NodePtr>::iterator fit = find(exclude_vec.begin(), exclude_vec.end(), node_ptr);
      if (fit != exclude_vec.end())
        continue;
      bool skip = false;
      assert(input->vTaskList.size() == 1);
      TaskPtr task_ptr = input->vTaskList[0];
        // checkwhether memory is OK 
        #ifdef CHECK_MEM
    #ifdef SCALE_DEBUG
    cout << __func__ << ": checking memory for " << task_ptr->id << " on " << node_ptr->id << endl;
    #endif

          if(!DoesFitMemLimit(gpu_ptr,task_ptr->id, node_ptr)) 
          {
            #ifdef SCHED_DEBUG
            cout << __func__ << ": failed mem check" << endl;
            #endif
            skip=true;
          }
        #endif
        // check whether interferences will be OK
        // check for input node
          float latency = GetLatency(type,task_ptr->id,task_ptr->batch_size,node_ptr, input_sim);
          if(latency + input->duty_cycle > task_ptr->SLO ){
            #ifdef SCHED_DEBUG
            cout << __func__ << ": failed SLO check" << endl;
            #endif
            skip=true;
          } 
        // check for neighboring node
          if(!skip) skip=CheckForInterference(type,input,node_ptr,input_sim);       
          #ifdef SCHED_DEBUG
          if(skip) cout << __func__ <<": failed interference check"<<endl;
          #endif
          if(skip) continue;
        
        // chose 100% nodes first 
        // check if the node is a 100% node, and partitoining is available
        #ifdef SCHED_DEBUG
        cout << __func__ << ": comparing: "<< input->resource_pntg << " with node: " << node_ptr->resource_pntg
        <<endl;
        #endif
        
        if(node_ptr->resource_pntg == 100 && (GetUseParts() == true)){
            min_diff=0;
            min_ptr=node_ptr;
        }    
        if(min_diff >= node_ptr->resource_pntg - input->resource_pntg){
            min_diff = node_ptr->resource_pntg - input->resource_pntg;
            min_ptr=node_ptr;
        }

      }
    }
    if(min_diff == 200) return EXIT_FAILURE;
    output=min_ptr;
    return EXIT_SUCCESS;
}
//
bool IncrementalScheduler::CheckFit(vector<NodePtr> &candidate_nodes, SimState &decision){
    //check how much is required
    int required_pntg=0;
    #ifdef SCHED_DEBUG
    printf("[CheckFit] Recieved %lu nodes for checking \n", candidate_nodes.size());
    printf("[ ");
    for(auto input_ptr : candidate_nodes){
      printf("%d",input_ptr->resource_pntg );
      printf(", ");
    }
    printf("]\n");

    #endif
    vector<NodePtr> flagged;
    map<NodePtr, int> remain_pntg;

    for(auto input_ptr : candidate_nodes){
        NodePtr result;
            #ifdef SCHED_DEBUG
            printf("[CheckFit] checking for part: %d \n", input_ptr->resource_pntg);
            #endif

        while(true){
          if(FindBestFit(decision,input_ptr, flagged, result)){
              #ifdef SCHED_DEBUG
              printf("[CheckFit] FAILED \n");
              #endif

              return EXIT_FAILURE;
          }
          if(remain_pntg.find(result) == remain_pntg.end()) // not in map
          {
            remain_pntg[result]=result->resource_pntg;
          }
          // the following case happens because BestFit does not consider tasks that were previously scheduled
          if(remain_pntg[result] < input_ptr->resource_pntg){
           flagged.push_back(result);
          }
          else break;
        }

        if(_REPARTITION && remain_pntg[result]==100)
          remain_pntg[result]-=input_ptr->resource_pntg;
        else
          remain_pntg[result]=0;

        if (remain_pntg[result] == 0)
          flagged.push_back(result);
    }
     #ifdef SCHED_DEBUG
        printf("[CheckFit] SUCCEEDED \n");
     #endif
    return EXIT_SUCCESS;
}

void CopyToNode(NodePtr org_node, NodePtr dst_node, bool enable_repart){
  //dst_node->id=org_node->id;
  dst_node->occupancy=org_node->occupancy;
  // if repartitioning is not enabled, resoource percentage should not be changed
  if(enable_repart) dst_node->resource_pntg=org_node->resource_pntg;
  dst_node->duty_cycle = org_node->duty_cycle;
  dst_node->vTaskList.clear();
  dst_node->reserved = org_node->reserved;
  //dst_node->dedup_num = org_node->dedup_num;
  for(auto task_ptr : org_node->vTaskList){
    dst_node->vTaskList.push_back(task_ptr);
  }
}

bool IncrementalScheduler::AllocateFit(const vector<NodePtr> &estimate, Task &task, SimState &decision){
  // compare pntg and allocate to best fit
  NodePtr best_fit_ptr;
  int initial_rate = task.request_rate;
  vector<NodePtr> flagged;
  for (auto input_ptr : estimate)
  {
      #ifdef SCHED_DEBUG
        printf("[Allocatefit] fitting part: %d \n", input_ptr->resource_pntg );
      #endif 
         bool isresidue;
        if(input_ptr->occupancy==1){
            isresidue=false;
        }
        else isresidue=true;
      if(FindBestFit(decision,input_ptr,flagged,best_fit_ptr)){
          printf("[AllocateFit] ran out of options\n");
          return EXIT_FAILURE;
      }
      int best_pntg = best_fit_ptr->resource_pntg;
      int input_pntg = input_ptr->resource_pntg;
      
      #ifdef SCHED_DEBUG
      printf("[AllocateFit] found pntg: %d, input pntg: %d \n",best_pntg,input_pntg );
      #endif
    
      if(best_pntg != 100 && best_pntg > input_pntg){
        // change setup of input node, since resource allocation needs to change
        input_ptr->resource_pntg=best_pntg;
        assert(input_ptr->vTaskList.size() == 1);
        TaskPtr temp = input_ptr->vTaskList[0];
        temp->batch_size=GetMaxBatch(*temp,input_ptr,decision,temp->request_rate,isresidue,true);
        if(temp->batch_size==0) return EXIT_FAILURE;
        float latency = GetLatency(input_ptr->type,temp->id,temp->batch_size,input_ptr,decision);
        if(!isresidue){
           input_ptr->duty_cycle = latency;
        }
        else input_ptr->duty_cycle = max((temp->batch_size * float(1000.0)) / temp->request_rate, latency);
        temp->throughput=(temp->batch_size * 1000.0 ) / input_ptr->duty_cycle;
        input_ptr->occupancy = latency / input_ptr->duty_cycle;
      }

      // if allocated with 100% node and required node is below 100%, split it for further use
      if(best_pntg == 100 && input_pntg < 100 && _REPARTITION){
          best_fit_ptr->resource_pntg=input_pntg;
          NodePtr new_node_ptr = MakeEmptyNode(best_fit_ptr->id,100-input_pntg,best_fit_ptr->type);
          new_node_ptr->id = best_fit_ptr->id;
          if(input_pntg == 50) new_node_ptr->dedup_num=1;
          decision.vGPUList[best_fit_ptr->id]->vNodeList.push_back(new_node_ptr);
      }

      //allocate to Node
      CopyToNode(input_ptr, best_fit_ptr, _REPARTITION);
      //update memory usage
      #ifdef CHECK_MEM
        AddGPUMemUsage(decision.vGPUList[best_fit_ptr->id], task.id, best_fit_ptr);
      #endif
      
      if(_USE_INTERFERENCE) {
        for(auto node_ptr : decision.vGPUList[best_fit_ptr->id]->vNodeList){
          //if(node_ptr->dedup_num == best_fit_ptr->dedup_num && node_ptr->resource_pntg == best_fit_ptr->resource_pntg) continue;
          if(node_ptr->occupancy ==1){
              int add_trp;
              int add_id;
              if(AdjustSatNode(node_ptr,decision, task)){
                  return EXIT_FAILURE;
              }
              std::cout << "remaining rate: " << task.additional_rate << endl;
          } 
          else AdjustResNode(node_ptr,decision);
        }
      }
      
      assert(best_fit_ptr->vTaskList.size() ==1); // there should be only one task
      task.request_rate-=best_fit_ptr->vTaskList[0]->throughput;
      #ifdef SCHED_DEBUG
      printf("[AllocateFit] Allocated successfully!! remaining rate of task %d : %d \n", task.id, task.request_rate+task.additional_rate);
      #endif
      
    }
    if(task.request_rate+task.additional_rate > TRP_SLACK) return EXIT_FAILURE;
    task.request_rate=0;
    return EXIT_SUCCESS;
}


bool IncrementalScheduler::MergeResidue(Task &task, SimState &input_sim){
#ifdef SCHED_DEBUG
      cout << "[MergeResidue] request_rate: " << task.request_rate << ", additional rate: " << task.additional_rate << endl; 
#endif
      int residue_cnt = 0;
      int request_rate = task.request_rate;
      int dummy_batch;
      float min_sat_trp = __FLT_MAX__;
      pair<string, int> type_part_pair;
      for (auto gpu_ptr : input_sim.vGPUList)
      {
        for (auto node_ptr : gpu_ptr->vNodeList)
        {
          if (node_ptr->occupancy < 1.0 && node_ptr->vTaskList.size() == 1)
          {
            // if the node is residue and has only one task
            if (node_ptr->vTaskList[0]->id == task.id)
            {
              request_rate += node_ptr->vTaskList[0]->throughput;
              residue_cnt++;
              // we can have several versions of saturate trps due to different types
              float sat_trp = MaxSaturateTrp(task, dummy_batch, node_ptr->resource_pntg, node_ptr->type);
              if (sat_trp < min_sat_trp)
              {
                min_sat_trp = sat_trp;
                type_part_pair.first = node_ptr->type;
                type_part_pair.second = node_ptr->resource_pntg;
            }
        }
        }
      }
    }
    #ifdef SCHED_DEBUG
    cout << "[MergeResidue]" << "residue count: "<<residue_cnt << " request_rate: " << request_rate << endl;
    cout << "[MergeResidue]" <<"limit trp: " << min_sat_trp << endl;
    #endif
    if( residue_cnt==0||(residue_cnt ==1 && request_rate < TRP_SLACK) || min_sat_trp < request_rate) {
      #ifdef SCHED_DEBUG
      cout << "[MergeResidue] will skip! because residue cant be improved"<<endl;;
      #endif
      if(task.request_rate > TRP_SLACK) return EXIT_FAILURE;
      else return EXIT_SUCCESS;
    }
  
    for(auto gpu_ptr : input_sim.vGPUList){
      for(auto node_ptr : gpu_ptr->vNodeList){
        if(node_ptr->occupancy < 1.0 && node_ptr->vTaskList.size() ==1){
          //if the node is residue and has only one task
          if(node_ptr->vTaskList[0]->id == task.id){
            // reset node_ptr setup
            SubtractGPUMemUsage(gpu_ptr,task.id,node_ptr);
            node_ptr->vTaskList.clear();
            node_ptr->occupancy=0;
            node_ptr->duty_cycle=0;
          }
        }
      }
    }
  #ifdef SCHED_DEBUG
    cout << "[MergeResidue] intially updated to task " << task.id << " request rate : "<<request_rate << endl;
  #endif
    bool found=false;
    for(auto gpu_ptr : input_sim.vGPUList){
      for(auto node_ptr : gpu_ptr->vNodeList){
        #ifdef CHECK_MEM
      #ifdef SCALE_DEBUG
      cout << __func__ << ": checking memory for " << task.id << " on " << node_ptr->id << endl;
      #endif
  
        if(!DoesFitMemLimit(gpu_ptr,task.id,node_ptr)){
          #ifdef SCHED_DEBUG
          cout << __func__ << ": failed memory check!" << endl;
          #endif
          continue;
        } 
        #endif
        if(node_ptr->vTaskList.empty() && node_ptr->type == type_part_pair.first && node_ptr->resource_pntg ==type_part_pair.second){
            Task temp_task;
            TaskPtr temp_task_ptr = make_shared<Task>(temp_task);
            temp_task_ptr->id=task.id;
            temp_task_ptr->SLO = task.SLO;
            temp_task_ptr->batch_size = GetMaxBatch(task,node_ptr,input_sim,request_rate,true,true);
  
            if(temp_task_ptr->batch_size == 0) continue; 
            node_ptr->duty_cycle = (temp_task_ptr->batch_size * 1000.0) / request_rate;
            node_ptr->occupancy = GetLatency(node_ptr->type,temp_task_ptr->id,temp_task_ptr->batch_size,node_ptr,input_sim) / node_ptr->duty_cycle;
            temp_task_ptr->throughput = temp_task_ptr->batch_size * 1000.0 / node_ptr->duty_cycle;
            temp_task_ptr->request_rate = temp_task_ptr->throughput;
            node_ptr->vTaskList.push_back(temp_task_ptr);
            task.request_rate = (request_rate > temp_task_ptr->throughput) ? request_rate - temp_task_ptr->throughput: 0; 
            #ifdef SCHED_DEBUG
            cout << "[MergeResidue] updated to task " << task.id << " request rate : "<<task.request_rate << endl;
            #endif
  
            found=true;
            break;
        }
      }
      if(found) break;
    }
    #ifdef SCHED_DEBUG
    cout << "[MergeResidue] ended with task " << task.id << " request rate : "<<task.request_rate << endl;
    #endif
    if(task.request_rate > TRP_SLACK) return EXIT_FAILURE;
  
  
    // Merge Nodes that have no tasks after merging
    for(auto gpu_ptr : input_sim.vGPUList){
      if(gpu_ptr->vNodeList.size() >1){
          int task_cnt=0;
          for(auto node_ptr : gpu_ptr->vNodeList){
            task_cnt += node_ptr->vTaskList.size();
          }
          if(task_cnt == 0 && _REPARTITION){
          #ifdef SCHED_DEBUG
          cout << "[MergeResidue] merging nodes of gpu: " << gpu_ptr->GPUID << endl;
          #endif
          // delete every node except first one
          gpu_ptr->vNodeList.erase(gpu_ptr->vNodeList.begin()+1, gpu_ptr->vNodeList.end());          
          gpu_ptr->vNodeList[0]->dedup_num=0;
          gpu_ptr->vNodeList[0]->resource_pntg=100;
          }
      }
    }
  return EXIT_SUCCESS;
}

bool cmp_nodeptr_dsc(const NodePtr &a, const NodePtr &b){
    return a->resource_pntg > b->resource_pntg;
}


bool cmp_nodeptr_asc(const NodePtr &a, const NodePtr &b){
  return a->resource_pntg < b->resource_pntg;
}

bool isModelScheduled(Task &task, SimState &decision){
  bool ret_val = false;
  for(auto gpu_ptr : decision.vGPUList){
    for(auto node_ptr : gpu_ptr->vNodeList){
      for(auto task_ptr: node_ptr->vTaskList){
        if(task.id == task_ptr->id) ret_val=true;
      }
    }
  }
  return ret_val;
}

bool IncrementalScheduler::AddModeltoSchedule(Task &task, SimState &decision){
  bool split=false;
  bool good_to_go=false;
  bool failed=false;
  task.additional_rate=0;
  // allocate min part, until this task does not require more gpu-lets
  // allocate saturate
  int prev_rate=task.request_rate;
  vector<NodePtr> estimate_result;
  bool scheduled=isModelScheduled(task,decision);
  #ifdef SCHED_DEBUG
  cout << __func__ << "model: " << task.id << " scheduled?: " << scheduled
  <<endl;
  #endif
  if(!scheduled){
      bool goto_adjust=false;
      GetEstimate(task, estimate_result, getMaxPartSize(decision));
      sort(estimate_result.begin(), estimate_result.end(),cmp_nodeptr_asc);
      if(!CheckFit(estimate_result,decision)){
        if(AllocateFit(estimate_result,task,decision)){
          goto_adjust=true;
        }
      }
      else goto_adjust=true;
    #ifdef SCHED_DEBUG
    std::cout << "AFTER CheckFit: " << std::endl;
    PrintResults(decision);
    #endif

    if(goto_adjust){ // if greedy fit failed
        // get empty partitions
        vector<NodePtr> empty_nodes;
        for(auto gpu_ptr : decision.vGPUList){
          for(auto node_ptr: gpu_ptr->vNodeList){
              if(node_ptr->vTaskList.empty()){
                empty_nodes.push_back(node_ptr);
              }
          }
        }
        if( !empty_nodes.empty()){
            // sort empty_nodes in descending order with respect to pntg
            sort(empty_nodes.begin(), empty_nodes.end(), cmp_nodeptr_dsc);
        }
        Readjust(task,empty_nodes, decision);
        #ifdef SCHED_DEBUG
        std::cout << "AFTER Readjust: " << std::endl;
        PrintResults(decision);
        #endif 

        if (!MergeResidue(task,decision)) return EXIT_SUCCESS;
        #ifdef SCHED_DEBUG
        std::cout << "AFTER MergeResidue: " << std::endl;
        PrintResults(decision);
        #endif

        failed=true;
        if(!_SELF_TUNING){
          vector<NodePtr> empty_vec;
          empty_vec.clear(); // just to make sure
          if(!AllocateTimeShare(task,decision, empty_vec)){
            failed=false;
          }
        }
        if(failed) return EXIT_FAILURE;
    }
  } // scheduled
  else{ // if this task was previously scheduled
      bool failed = false;
      vector<NodePtr> target_nodes;
      // nodes used when time sharing
      vector<NodePtr> ts_target_nodes;
      // Retrieve and clear nodesa
      // when clearing nodes that time share-task, be sure to just erase task only and not add to target_nodes
      std::cout << "AFTER SCHEDULED START: " << std::endl;
      PrintResults(decision);
      for(auto gpu_ptr : decision.vGPUList){
          for(auto node_ptr: gpu_ptr->vNodeList){
              printNodeInfo(node_ptr);
              if(node_ptr->vTaskList.empty()) target_nodes.push_back(node_ptr);
              else{
                bool found=false;
                for(int i =0; i < node_ptr->vTaskList.size(); i++){
                    if(node_ptr->vTaskList[i]->id == task.id){
                        node_ptr->vTaskList.erase(node_ptr->vTaskList.begin()+i);
                        ts_target_nodes.push_back(node_ptr);
                        found=true;
                        break;
                    }
                }
                // update occupancy
                float latency_sum=0;
                for(auto task_ptr: node_ptr->vTaskList){
                  latency_sum += GetLatency(node_ptr->type,task_ptr->id,task_ptr->batch_size,node_ptr,decision);
                }
                node_ptr->occupancy = (latency_sum / node_ptr->duty_cycle);
                // if "task" was the only task scheduled then add, if not then just continue(and leave it to AllocateTimeShare if neccessasry)
                if(node_ptr->vTaskList.empty()) target_nodes.push_back(node_ptr); 
              }
          }
      }
      if(!target_nodes.empty()) sort(target_nodes.begin(), target_nodes.end(), cmp_nodeptr_dsc);

      #ifdef SCHED_DEBUG
      cout <<"task: "<< task.id << " Len of target nodes: " << target_nodes.size() <<endl;
      cout <<"task: "<< task.id << " Len of ts target nodes: " << ts_target_nodes.size() <<endl;
      for(auto n_ptr : target_nodes){
        cout << "[ " << n_ptr->id << "," << n_ptr->resource_pntg << "," <<n_ptr->dedup_num<< "]" <<endl;
      }
      cout << "--------" << endl;
      if(!ts_target_nodes.empty())
      for(auto n_ptr : ts_target_nodes){
        cout << "[ " << n_ptr->id << "," << n_ptr->resource_pntg << "," <<n_ptr->dedup_num<< "]" <<endl;
      }
      
      #endif 
      Readjust(task,target_nodes,decision);
      std::cout << "AFTER SCHEDULED Readjust: " << std::endl;
      PrintResults(decision);
      if(!MergeResidue(task,decision)) return EXIT_SUCCESS;
      std::cout << "AFTER SCHEDULED MergeResidue: " << std::endl;
      PrintResults(decision);

      failed=true;
      if(!_SELF_TUNING){
        if(!AllocateTimeShare(task,decision, ts_target_nodes)){
          failed=false;
        } 
      }
      if(failed) return EXIT_FAILURE;

  } 
  return EXIT_SUCCESS;
}


// check whether decistion needs to be reverted 
// used as a gadget function for Reqdjust
bool IncrementalScheduler::CheckNeedtoRevert(NodePtr &new_node_ptr, Task &task, SimState &input){
    // added memory check, if it does not fit, revert
    bool revert=false;
  
  #ifdef CHECK_NETBW

  if(_isNLCInitated){
    if(!NLC.isBandwidthOK(input.vGPUList[new_node_ptr->id])){
      #ifdef SCHED_DEBUG
            cout << __func__ << ": netbw_check failed!" << endl;
      #endif
      revert=true;
    }
  }
  #endif

  

    if(_USE_INTERFERENCE){
#ifdef SCHED_DEBUG
      cout << __func__ << ": task.request_rate : " << task.request_rate << "task.additional_rate : " << task.additional_rate
      <<endl;
#endif 
      int additional_rate=0;
      // if the other node is also a saturated node, and SLO is volatted
      for(auto node_ptr : input.vGPUList[new_node_ptr->id]->vNodeList){
          if(node_ptr->dedup_num == new_node_ptr->dedup_num && node_ptr->resource_pntg == new_node_ptr->resource_pntg) continue;
          // if saturated node
          if(node_ptr->occupancy>=1){
            int rate_to_add=0;
            int task_id_to_add;
            if(AdjustSatNode(node_ptr,input)){

              revert=true;
              break;
            }
         }
          else{ // if residue node
            if(AdjustResNode(node_ptr,input)) revert=true;
          }
            
      }
      // if not going to be reverted, add request rate to take care of
      if(!revert){
        task.request_rate+=task.additional_rate;
        task.additional_rate=0;
      }

    }
    return revert;
}


// clear nodes of updated info
// used as a gadget function for Readjust
void IncrementalScheduler::RevertNode(NodePtr &node_ptr,Task &task, SimState &input){
      assert(node_ptr->vTaskList.size()<=1);
      #ifdef SCHED_DEBUG
      printf("[Readjust] need to revert results \n");
      #endif
      node_ptr->occupancy=0;
      node_ptr->vTaskList.clear();
      node_ptr->duty_cycle=0;
}

// readjusting algorithm: migrates tasks and changes partition
//if possible, try to squeeze taks into smaller partitions for given nodes
bool IncrementalScheduler::Readjust(Task &task, vector<NodePtr> &given, SimState &decision){
   int request_rate = task.request_rate;
    int max_batch;
    bool proceed_to_residue=false;
    sort(given.begin(), given.end(),cmp_nodeptr_dsc);
    if(_USE_INTERFERENCE){
      task.request_rate += task.additional_rate;
      task.additional_rate=0;
    }
    #ifdef SCHED_DEBUG
    printf("[Readjust] Called for task id : %d and rate %d \n", task.id, task.request_rate);
    #endif
 
      // readjusted saturate scheduliong
    while(task.request_rate+task.additional_rate> TRP_SLACK){
    #ifdef SCHED_DEBUG
          cout << "request_rate: " << task.request_rate << ", additional rate: " << task.additional_rate << endl; 
    #endif
        if(given.empty()) break;
        NodePtr temp_node_ptr = given.front();
        int temp_batch;
        // if allocated with 100% node and required node is below 100%, split it for further use
        if(temp_node_ptr->resource_pntg == 100 && _USE_PART && _REPARTITION){
          std::cout << "Following node will be splitted" << std::endl;
          printNodeInfo(temp_node_ptr);
          vector<NodePtr> candidates;
          const float  MAX_PART = getMaxPartSize(decision);
          EstimateTrp(temp_node_ptr->type,task,task.request_rate,candidates,MAX_PART);
          assert(!candidates.empty());
          int max_part = candidates[0]->resource_pntg;
          assert(candidates[0]->vTaskList.size() ==1);
          temp_batch=candidates[0]->vTaskList[0]->batch_size;
          if(max_part < 100){
            #ifdef SCHED_DEBUG
            cout << "[Readjust] Splitting node into " << max_part << " and " << 100-max_part << endl;
            #endif
            PrintResults(decision);
            temp_node_ptr->resource_pntg=max_part;
            NodePtr new_node_ptr = MakeEmptyNode(temp_node_ptr->id,100-max_part,temp_node_ptr->type);
            if(max_part == 50) new_node_ptr->dedup_num=1;
            decision.vGPUList[temp_node_ptr->id]->vNodeList.push_back(new_node_ptr);
            // also add to given resources for this task
            given.push_back(new_node_ptr);

            sort(given.begin()+1, given.end(),cmp_nodeptr_dsc);
          }
        }
        else{
          // get batch size for saturated partition
          temp_batch = GetMaxBatch(task,temp_node_ptr,decision,task.request_rate,false,true);
          // assuming we can modify the batch size later, we fix the temporary max batch size to 1
          if(temp_batch == 0) temp_batch=1;
        }
        int max_batch=temp_batch;
        
        float latency = GetLatency(temp_node_ptr->type,task.id,max_batch,temp_node_ptr,decision);
        float local_max_trp = max_batch * (1000.0 / latency);
        if(local_max_trp > task.request_rate){
          proceed_to_residue = true;
          break; 
        }
        TaskPtr temp_task_ptr = CreateNewTaskPtr(task.id,task.request_rate,task.SLO,max_batch,local_max_trp);
        // CHECK MEM BW 
        #ifdef CHECK_NETBW
        if(NLC.adjustBatchSizetoNetBW(temp_task_ptr,decision.vGPUList[temp_node_ptr->id])){
          return EXIT_FAILURE;
        }
        #endif
        

        #ifdef SCALE_DEBUG
            cout << __func__ << "(saturate): checking memory for " << task.id << " on " << temp_node_ptr->id << endl;
        #endif
        if(!DoesFitMemLimit(decision.vGPUList[temp_node_ptr->id],task.id,temp_node_ptr)){
          #ifdef SCHED_DEBUG
          cout << "Mem check failed!"<<endl;
          #endif
        }
        else{
          temp_node_ptr->occupancy=1;
          temp_node_ptr->vTaskList.push_back(temp_task_ptr);
          temp_node_ptr->duty_cycle = GetLatency(temp_node_ptr->type,task.id,max_batch,temp_node_ptr,decision);
          // and then check if this is OK
          bool revert=CheckNeedtoRevert(temp_node_ptr, task,decision);
          // revert decision if not OK
          if(revert ){
            RevertNode(temp_node_ptr,task,decision);
          }
          else{
            task.request_rate -= temp_task_ptr->throughput;
            #ifdef CHECK_MEM
            AddGPUMemUsage(decision.vGPUList[temp_node_ptr->id],task.id, temp_node_ptr);
            #endif 
            #ifdef SCHED_DEBUG
                printf("[Readjust] allocatd saturate node to [%d,%d,%d], remaining rate: %d \n", temp_node_ptr->id, temp_node_ptr->resource_pntg, temp_node_ptr->dedup_num,task.request_rate);
            #endif
          }
        }
        given.erase(given.begin(),given.begin()+1); // delete one node
    } // while loop

    if(proceed_to_residue){
        while(task.request_rate > TRP_SLACK){
        if(given.empty()) break;
        NodePtr temp_node_ptr = given.front();
        int max_pntg = temp_node_ptr->resource_pntg;
        #ifdef SCHED_DEBUG
          cout << __func__ << ": proceed_to_residue, task.id: "<< task.id << " request_rate " << task.request_rate << " additional_rate" << task.additional_rate << endl; 
        #endif
        max_batch=GetMaxBatch(task,temp_node_ptr,decision,task.request_rate,true,true);
        if(!max_batch) return EXIT_FAILURE;
        temp_node_ptr->duty_cycle =  max_batch * (1000.0  / task.request_rate);
        float local_max_trp = (1000.0*max_batch) / temp_node_ptr->duty_cycle;
        TaskPtr temp_task_ptr = CreateNewTaskPtr(task.id,task.request_rate,task.SLO,max_batch,local_max_trp);
        // CHECK MEM BW 
        #ifdef CHECK_NETBW
        if(NLC.adjustBatchSizetoNetBW(temp_task_ptr,decision.vGPUList[temp_node_ptr->id])){
          return EXIT_FAILURE;
        }       
        #endif
        float latency = GetLatency(temp_node_ptr->type,task.id,temp_task_ptr->batch_size,temp_node_ptr,decision);
        temp_node_ptr->occupancy=latency / temp_node_ptr->duty_cycle;
        #ifdef SCALE_DEBUG
            cout << __func__ << "(residue): checking memory for " << task.id << " on " << temp_node_ptr->id << endl;
        #endif
        if(!DoesFitMemLimit(decision.vGPUList[temp_node_ptr->id],task.id,temp_node_ptr)){
          #ifdef SCHED_DEBUG
          cout << "Mem check failed!"<<endl;
          #endif
        }
        else{
      
          temp_node_ptr->vTaskList.push_back(temp_task_ptr);
      
          // check if this is OK
          bool revert= CheckNeedtoRevert(temp_node_ptr,task,decision);
              // revert decision if not OK
          if(revert){
            RevertNode(temp_node_ptr,task,decision);
          }
          else{
            task.request_rate = (task.request_rate >= temp_task_ptr->throughput) ? task.request_rate - temp_task_ptr->throughput : 0;
            #ifdef CHECK_MEM
            AddGPUMemUsage(decision.vGPUList[temp_node_ptr->id],task.id,temp_node_ptr);
            #endif
            #ifdef SCHED_DEBUG
                printf("[Readjust] allocatd residue node, remaining rate: %d \n", task.request_rate);
            #endif
          } // if not revert
        }// check mem limit
        given.erase(given.begin(),given.begin()+1); // delete one node
        } // while task.reqeuest_rate 

      } // if (proceed_to_residue)

    if(task.request_rate > TRP_SLACK) return EXIT_FAILURE;
    else return EXIT_SUCCESS;
  }

  // tightens the rates of scheduling results
void IncrementalScheduler::ResidueTightening(const Task &task, SimState &decision ){
    // if node is a 1) residue (that serves throughput below original rate) and 2) non time sharing node, then tighten
  #ifdef SCHED_DEBUG
    cout << "[ResidueTightening] called with task id : " << task.id << endl;
    #endif
    for(auto gpu_ptr : decision.vGPUList){
      for(auto node_ptr : gpu_ptr->vNodeList){
        if(node_ptr->occupancy < 1 && node_ptr->vTaskList.size() == 1){
          TaskPtr task_ptr = node_ptr->vTaskList[0];
          if(task_ptr->id != task.id) continue;
          float est_trp;
          int min_batch=0;
          #ifdef SCHED_DEBUG
          cout << "[ResdiueTightening] org_batch: "<< task_ptr->batch_size << endl;
          #endif 
          for(int i=_MIN_BATCH; i< task_ptr->batch_size; i++){
            est_trp = (i * 1000.0) / (GetLatency(node_ptr->type,task_ptr->id,i,node_ptr,decision));

            if(est_trp > task_ptr->throughput){
              min_batch=i;
              break;
            } 
          }
          if(min_batch !=0){
          #ifdef SCHED_DEBUG
          cout << "[ResdiueTightening] org_batch: "<< task_ptr->batch_size << " new_batch: " << min_batch <<endl;
          cout << "[ResidueTightening] org_trp: " << task_ptr->throughput << "new_trp: " << est_trp << endl; 
          #endif
            task_ptr->batch_size = min_batch;
            task_ptr->throughput = est_trp;
          }           
       }
      }
    }
  }
int IncrementalScheduler::GetMaxReturnPart(const Task& task, string device){
    #ifdef SCHED_DEBUG
    cout << "[GetMaxReturnPart] received model id: " << task.id << endl; 
    #endif 
    map<int,float> per_part_trp;
    int dummy;
    // get throughput of every possible partition
    assert(_AVAIL_PARTS.size()>=1);
    if(_AVAIL_PARTS.size() == 1) return *_AVAIL_PARTS.begin();
    for(auto part : _AVAIL_PARTS){
      per_part_trp[part]=MaxSaturateTrp(task,dummy,part,device);
    }
    const float MIN_THRESHOLD=1.05;\
    uint len = _AVAIL_PARTS.size();
    float max_ret = 0;
    int max_part = _AVAIL_PARTS[len-1];
    #ifdef SCHED_DEBUG
    cout << "max_part initated to " << per_part_trp.end()->first << std::endl;  
    #endif
    for(int i=len-1; i > 1; i--){
      int low_part=_AVAIL_PARTS[i];
      int high_part=_AVAIL_PARTS[i-1];
      float low_trp=per_part_trp[low_part];
      float high_trp=per_part_trp[high_part];
  
      #ifdef SCHED_DEBUG
        cout << "low_part: "<< low_part << " high_part: "<< high_part << " low_trp: "<< low_trp << " high_trp: " << high_trp<<endl;
      #endif 
    
      if(high_trp/low_trp < MIN_THRESHOLD) continue;
      float ret = (high_trp-low_trp)/(high_part-low_part);
      if( ret > max_ret){
          max_ret=ret;
          max_part=high_part;
      } 
    }
    #ifdef SCHED_DEBUG
    cout << "[GetMaxReturnPart] returning max part:  " << max_part<< endl; 
    #endif 

    return max_part;
}
    
bool IncrementalScheduler::DoesFitMemLimit(GPUPtr &gpu_ptr, int model_id, NodePtr &node_ptr){
      assert(_map_model_id_to_mem_size[model_id] !=0);
      assert(gpu_ptr->TOTAL_MEMORY);
      int additional_memory=0;
      // if model is already loaded, part is also loaded no need to even check
      if(IsModelLoaded(gpu_ptr, node_ptr, model_id)) return true;
      if(!IsPartLoaded(gpu_ptr,node_ptr)){
        additional_memory=PYTORCH_DEFAULT_MEM_USAGE;
      }
      bool result= ((gpu_ptr->TOTAL_MEMORY)*MEM_ROOM - gpu_ptr->used_memory) > _map_model_id_to_mem_size[model_id] + additional_memory; 

      #ifdef SCHED_DEBUG
      cout << __func__ << "called for gpu: " << gpu_ptr->GPUID << " searching for " << node_ptr->resource_pntg <<endl;
      cout << __func__ << ": comparing left memory: " << gpu_ptr->TOTAL_MEMORY*MEM_ROOM - gpu_ptr->used_memory << " and model_mem: "<< _map_model_id_to_mem_size[model_id] << endl;
      cout << __func__ << ": with addtional " << additional_memory << endl; 
      #endif

      #ifdef SCALE_DEBUG
      if(!result){
          cout << __func__ << ": task id " << model_id << " DOES NOT fit on GPU " << gpu_ptr->GPUID
          <<endl;
      }
      else{
           cout << __func__ << ": task id " << model_id << " DOES fit on GPU " << gpu_ptr->GPUID
          <<endl;
      }

      #endif
      return result;
}

bool IncrementalScheduler::AddGPUMemUsage(GPUPtr &gpu_ptr, int model_id, NodePtr &node_ptr){
    #ifdef SCHED_DEBUG
      cout << __func__ << "called"  << endl;
    #endif 
        int additional_mem = 0;
        if(IsModelLoaded(gpu_ptr, node_ptr, model_id)) return EXIT_SUCCESS;        
        if(!IsPartLoaded(gpu_ptr,node_ptr)){
          additional_mem=PYTORCH_DEFAULT_MEM_USAGE;
        }
        if(DoesFitMemLimit(gpu_ptr, model_id,node_ptr)){
          gpu_ptr->used_memory += (_map_model_id_to_mem_size[model_id] + additional_mem);
          #ifdef SCHED_DEBUG
          cout << "gpu" << gpu_ptr->GPUID <<  " used_memory updated to " << gpu_ptr->used_memory << endl;
          cout << "gpu" << gpu_ptr->GPUID <<  " remaining memory: " << gpu_ptr->TOTAL_MEMORY - gpu_ptr->used_memory << endl;
          #endif 
          MemNode temp;
          temp.dedup_num=node_ptr->dedup_num;
          temp.part=node_ptr->resource_pntg;
          gpu_ptr->vLoadedParts.push_back(temp);
          #ifdef SCHED_DEBUG
          cout << "gpu" << gpu_ptr->GPUID <<  " loaded parts: ";
          for(auto mem_info : gpu_ptr->vLoadedParts){
            cout << "["<<mem_info.part << ", " << mem_info.dedup_num<< "]";
          }
          cout<<endl;
          #endif
          return EXIT_SUCCESS;
        }
    return EXIT_FAILURE;
}

bool IncrementalScheduler::SubtractGPUMemUsage(GPUPtr &gpu_ptr, int model_id, NodePtr &node_ptr){
      #ifdef SCHED_DEBUG
      cout << __func__ << "called"  << endl;
      #endif
      int additional_mem = 0;
      // if part is not even loaded... this is a huge error on the programmer's behalf
      assert(IsPartLoaded(gpu_ptr, node_ptr));
      // if model is not loaded, no need to substract
      if (!IsModelLoaded(gpu_ptr, node_ptr, model_id))
        return EXIT_SUCCESS;
      if (gpu_ptr->used_memory - _map_model_id_to_mem_size[model_id] < 0)
      {
        cout << __func__ << ": gpu " << gpu_ptr->GPUID << " used memory: " << gpu_ptr->used_memory << endl;
        cout << __func__ << ": CAN NOT subtract" << _map_model_id_to_mem_size[model_id] << endl;
        return EXIT_FAILURE;
        }
        gpu_ptr->used_memory -= _map_model_id_to_mem_size[model_id];
        #ifdef SCHED_DEBUG
        cout << "gpu" << gpu_ptr->GPUID <<  "used_memory updated to " << gpu_ptr->used_memory << endl;
        cout << "gpu" << gpu_ptr->GPUID <<  "remaining memory: " << gpu_ptr->TOTAL_MEMORY - gpu_ptr->used_memory << endl;
        #endif 
        return EXIT_SUCCESS;
}

void IncrementalScheduler::GetEstimateTrpST(string device, const Task &task, int rate, vector<NodePtr> &output_vec, const int MAX_PART)
{
  int _rate = rate;
  vector<int> parts = _AVAIL_PARTS;
  sort(parts.begin(), parts.end());

  // 1. find min part which can satisfy rate
  while (_rate > 0)
  {
    int max_batch;
    int max_trp = 0;
    int max_part = 0;
    //
    for (int part : parts)
    {
      max_part = part;
      max_trp = MaxSaturateTrp(task, max_batch, part, device);
      if (max_trp > _rate)
        break;
    }
    assert(max_trp != 0 && max_part != 0);
    NodePtr temp_node_ptr = MakeEmptyNode(output_vec.size(), max_part, device);
    float latency = GetLatency(device, task.id, max_batch, max_part);
          #ifdef SCHED_DEBUG
          printf("[EstimateTrpST]node- latency: %lf, batch_size: %d, rate: %d, part: %d  \n", latency, max_batch, _rate, max_part );
          #endif
          temp_node_ptr->duty_cycle= max_batch * (1000.0  / _rate);
          Task new_task;
          new_task.id=task.id;
          new_task.request_rate=_rate;
          new_task.batch_size=max_batch;
          new_task.SLO=task.SLO;
          new_task.throughput=max_trp;
          TaskPtr temp_task_ptr = make_shared<Task>(new_task);          
          #ifdef SCHED_DEBUG
          printf("[EstimateTrpST] batch_size after allocating : %d, trpt: %lf \n", temp_task_ptr->batch_size, temp_task_ptr->throughput);
          #endif
          temp_node_ptr->occupancy= latency / temp_node_ptr->duty_cycle;
          temp_node_ptr->vTaskList.push_back(temp_task_ptr);
          output_vec.push_back(temp_node_ptr);
          _rate-=max_trp;
        }

}

void IncrementalScheduler::setupNetworkChecker(string json_file){
  if(NLC.setupPerTaskInputDimension(json_file)){
    _isNLCInitated=false;
  }
  else _isNLCInitated=true;
}

bool IncrementalScheduler::inspectNetworkBW(SimState &input){
   return NLC.isBandwidthOK(input);
}
 
} // namespace:squishy_bin_packing
