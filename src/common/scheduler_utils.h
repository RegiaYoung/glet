#ifndef _SCHEDULER_H__
#define _SCHEDULER_H__

#include <vector>
#include <string>

#include "scheduler_base.h"

void ResetScheduler(std::vector<Node> node_vec); 
void PrintResults(SimState &input);
int GetNumofUsedGPUs(SimState &input);
void PrintMemState(const GPUPtr &gpu_ptr);
void CopyToOutput(SimState &input, SimState &output);
void RecoverScheduler(const SimState &backup, SimState &output);
void CopySession( vector<Task> &org_session, vector<Task> &dst_session); 
TaskPtr CreateNewTaskPtr(int id=0, int request_rate=0, int SLO=0, int batch_size=0, float throughput=0);
TaskPtr CreateNewTaskPtr(int id=0, int request_rate=0, int ORG_RATE=0, int additiional_rate=0, 	int SLO=0, int batch_size=0, float throughput=0);
TaskPtr CreateNewTaskPtr(Task &task);
int getMaxPartSize(const SimState &input);
int getMinPartSize(const SimState &input);

#else
#endif
