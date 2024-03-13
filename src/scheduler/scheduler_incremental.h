#ifndef _SCHEDULER_INCREMENTAL_H__
#define _SCHEDULER_INCREMENTAL_H__

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "scheduler_base.h"
#include "network_limit.h"
using namespace std;

namespace squishy_bin_packing{
class IncrementalScheduler : public  BaseScheduler{
	public:
		IncrementalScheduler();
		~IncrementalScheduler();
        bool SquishyBinPacking(vector<Task> *task_list, SimState &prev_output, SimState &new_output, bool allow_repart=false);
	    void InitSaturateTrp(Task &task);
		void setupNetworkChecker(string json_file);
		bool inspectNetworkBW(SimState &input);
		// also used outside of scheduler
		bool DoesFitMemLimit(GPUPtr &gpu_ptr, int model_id, NodePtr &node_ptr);
		bool AddGPUMemUsage(GPUPtr &gpu_ptr, int model_id, NodePtr &node_ptr);
		void SetSelfTuning(bool use);
	protected:
		bool ScheduleSaturate(vector<Task> *session, vector<NodePtr> *node_list, vector<Task> *residue_list);
		bool ScheduleResidue(vector<NodePtr> *node_list, vector<Task> *residue_list);
		bool IncrementalScehduling(vector<Task> &session, SimState &decision);
		bool ElasticPartitioning(vector<Task> &session, SimState &decision);
		bool AddModeltoSchedule(Task &task, SimState &decision); 
		void ResidueTightening(const Task &task, SimState &decision);
		bool CheckFit(vector<NodePtr> &candidate_nodes, SimState &decision);
		// gets list of nodes for allocating task
		void GetEstimate(Task &task, vector<NodePtr> &output_vec, const int MAX_PART);
		bool AllocateFit(const vector<NodePtr> &estimate, Task &task, SimState &decision);
		bool Readjust(Task &task, vector<NodePtr> &given, SimState &decision); 
		bool MergeResidue(Task &task, SimState &input_sim); 
		bool AllocateTimeShare(Task &task, SimState &sim, vector<NodePtr> &target_nodes);
		int GetMinPart(string device, Task task, const NodePtr node_ptr, int &residue_rate, int &result_batch); 
		void EstimateTrp(string device, Task &task, int rate, vector<NodePtr> &output_vec, const int MAX_PART);
		float MaxSaturateTrp(const Task &task, int &output_batch, const int resource_pntg, string type);
		bool FindBestFit(SimState &input_sim,NodePtr &input,vector<NodePtr> &exclude_vec ,NodePtr &output);
		bool CheckForInterference(string device, NodePtr the_node, NodePtr the_other_node, SimState &sim);
		int GetMaxReturnPart(const Task &task, string device);
		bool SubtractGPUMemUsage(GPUPtr &gpu_ptr, int model_id, NodePtr &node_ptr);
		void RevertNode(NodePtr &node_ptr, Task &task, SimState &input);
		bool CheckNeedtoRevert(NodePtr &node_ptr, Task &task, SimState &input);
		void GetEstimateTrpST(string device, const Task &task, int rate, vector<NodePtr> &output_vec, const int MAX_PART);
		int GetMaxReturnPartST(const Task &task, const int rate);

		map<int, vector<SatTrpEntry>*> _per_model_sat_table;
		// the amount of memory whenever a new gpu-let needs to be added
		const int PYTORCH_DEFAULT_MEM_USAGE = 1230; 	
		// amount of memory available actually used, exists for stable exeucution
		// check Memory with a slightly tighter standard(0.9 of capacity) due to engineering issues
		const float MEM_ROOM = 0.9;
		NetworkLimitChecker NLC;
		bool _isNLCInitated=false;
	};
} //squishy_bin_packing


#else
#endif
