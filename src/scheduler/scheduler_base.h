#ifndef _SCHEDULER_BASE_H__
#define _SCHEDULER_BASE_H__

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "device_perf_model.h"

using namespace std;


typedef struct _task
{
  int id; 
  int request_rate; // mutable rate, will be chnged during scheduling
  int ORG_RATE; // the original rate, 
  int additional_rate; // additional rate due to interference, only used during scheduling
  int SLO;
  int batch_size;
  float throughput;
} Task;

typedef shared_ptr<Task> TaskPtr;

typedef struct _MemNode
{
	int part;
	int dedup_num; //dedup_num added for discerning between 50:50 partitions
} MemNode;

typedef struct _Node
{
  vector<TaskPtr> vTaskList;
  float duty_cycle;
  float occupancy;
  int resource_pntg;
  int id; // do not confuse with task ID, this is GPU node id;
  bool reserved; // flag indicating whether is reserved or not
  int dedup_num; // this is to distinguish between same partitions e.g) 50-50 nodes
  string type; // the type of GPU that this node is on, empty if not allocated to a GPU, same as TYPE
} Node;

typedef shared_ptr<Node> NodePtr;

typedef struct _GPU
{
    int GPUID;
  	int TOTAL_MEMORY;
  	int used_memory;
	string TYPE;
	vector<MemNode> vLoadedParts;
   	vector<NodePtr> vNodeList;
} GPU;

typedef shared_ptr<GPU> GPUPtr;

typedef struct _SimState
{
  vector<GPUPtr> vGPUList;
  vector<int> parts;
  int next_dummy_id;
} SimState;

typedef struct _SatTrpEntry
{
	int part;
	int max_batch;
	float sat_trp;
	string type;
} SatTrpEntry;

namespace squishy_bin_packing{

class BaseScheduler {

	public:
		BaseScheduler();
		~BaseScheduler();
		bool InitializeScheduler(string sim_config_json_file, \
								string mem_config_json_file,string device_config,string res_dir, \
								vector<string> &batch_filenames);
		void SetupTasks(string task_csv_file, vector<Task> *task_list);
		void SetupAvailParts(vector<int> input_parts);
		void InitDevMems(SimState &input);
		void UpdateDevMemUsage(vector<int> &used_mem_per_dev, SimState &input);
		void InitiateDevs(SimState &input, int nDevs);
		bool SquishyBinPacking(vector<Task> *task_list, SimState &output);
		void ResetScheduler(SimState &input,std::vector<Node> node_vec);
		bool GetUseParts();
		vector<int> GetAvailParts();
		int GetMaxGPUs();
		int SetMaxGPUs(int num_gpu);
		bool GetChoseFirst();
		int GetMaxBatchSize();
		int GetModelMemUSsage(int model_id);
		string GetModelName(int id);
		int GetModelID(string model_name);
		vector<string> GetScheduledModels();
		NodePtr MakeEmptyNode(int gpu_id, int resource_pntg, string type);
		float GetLatency(string device, int model_num, int batch, int part); // used for non-interference version of getting latency
		float GetLatency(string device, int model_num, int batch, NodePtr self_node, SimState &input); // used for getting latency also considering interference	
	protected:
		bool ScheduleSaturate(vector<Task> *session, vector<NodePtr> *node_list, vector<Task> *residue_list, SimState &input);
		bool ScheduleResidue(vector<NodePtr> *node_list, vector<Task> *residue_list, SimState &input);
		bool postAdjustInterference(SimState &input);
		bool AdjustResNode(NodePtr &node_ptr, SimState &simulator);
		bool AdjustSatNode(NodePtr &node_ptr, SimState &simulator);
		bool AdjustSatNode(NodePtr &node_ptr, SimState &simulator, Task &task_to_update);

		int SetupScheduler(string sim_config_json_file);
		int setupDevicePerfModel(string devcice_config_json_file, string res_dir);
		int SetupPerModelMemConfig(string mem_config_json_file);
		void SetupBatchLatencyTables(string res_dir, vector<string> &filenames);
		void InitDevMem(GPUPtr gpu_ptr, int mem);
		bool MergeNodes(NodePtr a, NodePtr b, NodePtr dst, SimState &input);
		bool GetOtherNode(NodePtr input_the_node, NodePtr &output_the_other_node, SimState &sim);
		bool IsSameNode(NodePtr a, NodePtr b);
		bool IsPartLoaded(GPUPtr gpu_ptr, NodePtr node_ptr);
		bool IsModelLoaded(GPUPtr gpu_ptr, NodePtr node_ptr, int model_id);
		void FillReservedNodes(SimState &input);
		bool GetNextEmptyNode(NodePtr &new_node, SimState &input);
		float GetInterference(string device, int a_id, int b_id, int a_batch, int b_batch, int partition_a, int partitoin_b);
		float GetInterference(string device, int model_id, int batch_size, NodePtr node_ptr, GPUPtr gpu_ptr);
		float GetBatchLatency(string modelname, int batch);
		int GetMaxBatch(Task &self_task, const NodePtr &self_node, SimState &input, int &req, bool is_residue, bool interference);
		bool AssociateGPU(NodePtr &dst_node, GPUPtr &dst_gpu);
		bool FillFreeGPUs(NodePtr dst_node, SimState &input);
		// EXPERIMETNAL return 99%ile of Poisson CDF for given rate
		int Return99P(const int mean);
		double CalcPoissProb(int actual, int mean);
		void printNodeInfo(const NodePtr &node);
		bool _BATCH_LATENCY = 1;
		int _MAX_GPU;
		int _MAX_PART = 2;
		int _MIN_BATCH = 1;
		int _MAX_BATCH = 32;
		int _MODEL_LIMIT;
		bool _USE_PART;
		float _RESIDUE_THRESHOLD;
		float _SLO_RATIO;
		float _LATENCY_RATIO;
		bool _USE_INTERFERENCE;
		float _RESERVE_RATIO;
		bool _skip_check;
		int _PRIOR_RESERVE_NUM;
		bool _CHECK_LATENCY = true;
		bool _USE_INCREMENTAL;
		bool _CHOSE_FIRST;
		bool _SELF_TUNING = false;
		// flag indicating whether to allow repartitioning, true by default
		bool _REPARTITION = true;
		vector<int> _AVAIL_PARTS = {100};
		vector<string> _models_managed;
		// TODO: let a function  and recieve inputs as parameters/file
		// for now we hard code
		map<int, float> _batch_latency_1_28_28;
		map<int, float> _batch_latency_3_224_224;
		map<int, float> _batch_latency_3_300_300;

		vector<string> ID_TO_MODEL_NAME = {"lenet1", "lenet2", "lenet3",
											"lenet4", "lenet5", "lenet6", "googlenet",
											"resnet50", "ssd-mobilenetv1", "vgg16",
											"mnasnet1_0", "mobilenet_v2", "densenet161", "bert"};
		map<int, int> _map_model_id_to_mem_size;
		map<string, DevPerfModel> _name_to_dev_perf_model_table;
		map<string, int> _type_to_num_of_type_table;
		
	};
} //squishy_bin_packing


#else
#endif
