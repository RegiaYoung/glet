#ifndef _APP_SPEC_H__
#define _APP_SPEC_H__
#include <vector>
#include <string>
#include <map>
#include <torch/script.h>
#include <torch/serialize/tensor.h> 
#include <torch/serialize.h>

using namespace std;

enum TensorDataOption {KFLOAT32=0,KINT64=1};

typedef struct _CompDep{
    string type;
    string name;
    int id; 
    int SLO; // node-wise SLO
    int predecessor; // used for finding critical path
    vector<int> input; 
    vector<int> output;
} CompDep;

typedef struct _TensorSpec{
    vector<int> dim;
    vector<int> output;
    int id;
    TensorDataOption dataType;
} TensorSpec;

class AppSpec{
    public:
        AppSpec(string name);
        ~AppSpec();
        string getName();
        float getNodeSLO(int id);
        void setGlobalVecID(int id);
        int getGlobalVecID();
        int setupModelDependency(const char* AppJSON); //recieves filename as input
        string getOutputComp();
        vector<CompDep> getModelTable();
        vector<TensorSpec> getInputTensorSpecs();
        void addInputSpec(TensorSpec &Tensor);
        void setOutputComp(string outputComp);       
        torch::Tensor aggregateOutput(string &CompType, vector<torch::Tensor> &Inputs);
        void calcCriticalPath(); // compute CP and allocates SLO budget
        int sendOutputtoClient(int socketFD, int  taskID);
        vector<int> getNextDsts(int curr_id);
        string getModelName(int id);
        bool isOutput(int id);
        vector<int> getInputforOutput();
        unsigned int getTotalNumofNodes();
        void printSpecs();

#ifdef DEBUG
        void printCritPath();
#endif
    private:
        string _name;
        string _outputComp;
        vector<TensorSpec> inputTensors;
        vector<CompDep> _ModelTable;
        vector<CompDep> _CriticalPath;
        map<int,double> _RequestIntervals;
        map<int,int> _perNodeSLO; // per node SLO, should be allocated after call to 'calcCriticalPath'
        map<int, float> _perNodeLatency; // per node latency, reads profiled data before hand and set with 'setNodeLatency'
        double _SyncGran;
        int _globalVecId; // used for indexing which AppSpec, in the AppSpecVec global vector
};
#endif 
