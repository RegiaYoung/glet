#include "input.h"
#include "json/json.h"
#include <opencv2/opencv.hpp>
#include <vector> 
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "router.h"
#include "torchutils.h"
#define NZ 100 
using namespace cv;
vector<torch::Tensor> glbImgTensors;
unsigned int ID=0;
extern GlobalScheduler gScheduler;
extern SysMonitor ServerState;
const int DEFAULT_MAX_BATCH=32;
const int DEFAULT_MAX_DELAY=0;
mutex img_index_mtx;

#ifdef BACKEND
extern map<int,TensorSpec*> MapIDtoSpec;
#endif
torch::Tensor convertToTensor(float *input_data, int batch_size, int nz){
            return convert_LV_vectors_to_tensor(input_data, batch_size,nz);
}

int configAppSpec(const char* ConfigJSON, SysMonitor &SysState, string res_dir){
#ifdef DEBUG
    printf("%s: Reading %s \n", __func__, ConfigJSON);
#endif 
    Json::Value root;
    std::ifstream ifs;
    ifs.open(ConfigJSON);
    if(!ifs){
        cout << __func__ << ": failed to open file: " << ConfigJSON
        <<endl;
        exit(1);
    }


    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    if (!parseFromStream(builder, ifs, &root, &errs)) {
    std::cout << errs << std::endl;
    return EXIT_FAILURE;
    ifs.close();
    }
    for(unsigned int i=0; i<root["App_specs"].size(); i++){

        AppSpec *temp = new AppSpec(root["App_specs"][i]["name"].asString());
        if(readAppJSONFile(root["App_specs"][i]["file"].asCString(), *temp, res_dir)){
            cout << __func__ << ": failed to setup " << root["App_specs"][i]["file"].asCString()
            <<endl;
            continue;
        }
        int len = SysState.AppSpecVec.size();
        temp->setGlobalVecID(len);
        SysState.AppSpecVec.push_back(*temp);

    }
    ifs.close();
  return EXIT_SUCCESS;
}

int readAppJSONFile(const char* AppJSON, AppSpec &App, string res_dir){
    Json::Value root;
    std::ifstream ifs;
    string full_name = res_dir + "/"+string(AppJSON);
#ifdef DEBUG
    printf("%s: Reading App JSON File: %s \n", __func__, full_name.c_str());
#endif 
    ifs.open(full_name);
    // fail check
    if(!ifs){
        cout << __func__ << ": failed to open file: " << full_name
        <<endl;
        exit(1);
    }

    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    if (!parseFromStream(builder, ifs, &root, &errs)) {
        std::cout << errs << std::endl;
        ifs.close();
        return EXIT_FAILURE;
    }
	//set input dimensions
	std::vector<int> InputDim;
	for(unsigned int i=0; i<root["Input"].size(); i++){
		TensorSpec *tpTensor=new TensorSpec();
		for(unsigned int j=0; j<root["Input"][i]["InputDim"].size();j++){
			tpTensor->dim.push_back(root["Input"][i]["InputDim"][j].asInt());
       
		}
        tpTensor->id=root["Input"][i]["ID"].asInt();

		if(root["Input"][i]["InputType"].asString() == "FP32")
			tpTensor->dataType=KFLOAT32;
		else if(root["Input"][i]["InputType"].asString() == "INT64")
			tpTensor->dataType=KINT64;
		else{
			LOG(ERROR) << "wrong type of InputType: "<< root["Input"][i]["InputType"].asString() <<endl;
			return EXIT_FAILURE;
		}
        for(unsigned int j=0; j<root["Input"][i]["output"].size();j++){
            tpTensor->output.push_back(root["Input"][i]["output"][j].asInt());
        }
        #ifdef BACKEND
        
        string name = App.getName();
        MapIDtoSpec[ServerState.PerModeltoIDMapping[name]]=tpTensor;

        #else
		App.addInputSpec(*tpTensor);
        #endif
	}
    #ifdef BACKEND  
    #else
    //setup model-wise restrictions
	 const int DEFAULT_MAX_BATCH=32;
	const int DEFAULT_MAX_DELAY=0;
    for(unsigned int i=0; i < root["Models"].size();i++){
        gScheduler.addtoNetNames(root["Models"][i]["model"].asString());
        if (root["Models"][i].isMember("max_batch"))
            gScheduler.setMaxBatch(root["Models"][i]["model"].asString(), "gpu", root["Models"][i]["max_batch"].asInt());
        else
             gScheduler.setMaxBatch(root["Models"][i]["model"].asString(), "gpu", DEFAULT_MAX_BATCH);   
       if (root["Models"][i].isMember("SLO")) gScheduler.setupModelSLO(root["Models"][i]["model"].asString(), root["Models"][i]["SLO"].asInt());
    }
	//setup model dependencies
    if(App.setupModelDependency(full_name.c_str())){
        cout << __func__ << ": failed to setup dependency for " << full_name
        <<endl;
        return EXIT_FAILURE;
    }
    #endif
    ifs.close();
    return EXIT_SUCCESS;
}


