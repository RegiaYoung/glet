#ifndef _LAT_MODEL_H__
#define _LAT_MODEL_H__

#include <string> 
#include <vector>
#include <map>
#include <unordered_map>
using namespace std;

typedef struct _entry
{
  int batch;
  int part;
  float latency;
  float gpu_ratio;
} entry;

class LatencyModel {
	public:
	void setupTable(string TableFile);
	float GetLatency(string model, int batch, int part);
        float getGPURatio(string model, int batch, int part);
        int makeKey(int batch, int part);
        entry* parseKey(int key);
	private:
        float GetBatchPartInterpolatedLatency(string model, int batch, int part);
        float GetBatchInterpolatedLatency(string model, int batch, int part);
        map<string, unordered_map<int,float>*> perModelLatnecyTable;
        map<string, unordered_map<int,float>*> perModelGPURatioTable;
        map<string, map<int,vector<int>>> perModelBatchVec;
};

#else
#endif
