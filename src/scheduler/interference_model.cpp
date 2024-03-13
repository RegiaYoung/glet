#include "interference_model.h"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace interference_modeling;


void interference_modeling::interference_model::setup(string model_const_file, string util_file)
{
	this->_setup(model_const_file, util_file);
}

void interference_modeling::interference_model::_setup(string input_file1, string input_file2)
{
		
	string line;
	string token;
	//set INT_MODEL_constants
	ifstream infile1(input_file1);
	getline(infile1, line); // skip first line
	getline(infile1, line);
	istringstream ss2(line);
	for(int i=0;i<5;i++){
		getline(ss2, token, ',');
		interference_modeling::interference_model::INT_MODEL_constants.push_back(stod(token));
	}
	//set UTIL
	ifstream infile2(input_file2);
	getline(infile2, line);// skip first line
	while(getline(infile2, line)){
		pair<string, int> model_batch;
		vector<double> const_set;
		istringstream ss(line);
		getline(ss, token, ',');
		model_batch.first=token; // name
		getline(ss, token, ',');
		model_batch.second=stoi(token); //batch size	
        getline(ss, token, ','); // duration
		const_set.push_back(stod(token));
		for(int i=0;i<5;i++){
			getline(ss, token, ','); // sm_util, l2_util, mem_util, ach_occu, the_occu
			const_set.push_back(stod(token)/100);
		}
		interference_modeling::interference_model::UTIL[model_batch]=const_set;
	}
}

double interference_modeling::interference_model::get_interference(string my_model, int my_batch, int my_partition, string your_model, int your_batch, int your_partition)
{
    if (my_model == "lenet1" || my_model == "lenet2" || my_model == "lenet3" || my_model == "lenet4" || my_model == "lenet5" || my_model == "lenet6"){
            my_model="lenet";
    }
    if(your_model == "lenet1" || your_model == "lenet2" || your_model == "lenet3" || your_model == "lenet4" || your_model == "lenet5" || your_model == "lenet6"){
            your_model="lenet";
    }

	int my_batch_below=(int)exp2((int)log2(my_batch));
	int my_batch_top=min(32, my_batch_below*2);
	int your_batch_below=(int)exp2((int)log2(your_batch));
	int your_batch_top=min(32, your_batch_below*2);

	pair<string, int> my_info_below(my_model, my_batch_below);
	pair<string, int> my_info_top(my_model, my_batch_top);
	pair<string, int> your_info_below(your_model, your_batch_below);
	pair<string, int> your_info_top(your_model, your_batch_top);

	double my_batch_ratio=1.0;
	if(my_batch_top!=my_batch_below){
		my_batch_ratio=(double)(my_batch-my_batch_below)/(my_batch_top-my_batch_below);
	}

	double your_batch_ratio=1.0;
	if(your_batch_top!=your_batch_below){
		your_batch_ratio=(double)(your_batch-your_batch_below)/(your_batch_top-your_batch_below);
	}

    double my_l2_util = interference_modeling::interference_model::UTIL[my_info_top][2] * my_batch_ratio + \
	interference_modeling::interference_model::UTIL[my_info_below][2] * (1-my_batch_ratio);
    double your_l2_util = interference_modeling::interference_model::UTIL[your_info_top][2]*your_batch_ratio + \
	interference_modeling::interference_model::UTIL[your_info_below][2]*(1-your_batch_ratio);
    double my_dram_util = interference_modeling::interference_model::UTIL[my_info_top][3] * my_batch_ratio + \
	interference_modeling::interference_model::UTIL[my_info_below][3] * (1-my_batch_ratio);
    double your_dram_util = interference_modeling::interference_model::UTIL[your_info_top][3]*your_batch_ratio + \
	interference_modeling::interference_model::UTIL[your_info_below][3]*(1-your_batch_ratio);
    my_l2_util = my_l2_util *my_partition/100.0;
    your_l2_util = your_l2_util * your_partition/100.0;
    my_dram_util = my_dram_util *my_partition/100.0;
    your_dram_util = your_dram_util * your_partition/100.0;

	double alpha = interference_modeling::interference_model::INT_MODEL_constants[0]*my_l2_util+\
                    interference_modeling::interference_model::INT_MODEL_constants[1]*your_l2_util+\
                    interference_modeling::interference_model::INT_MODEL_constants[2]*my_dram_util+\
					interference_modeling::interference_model::INT_MODEL_constants[3]*your_dram_util+\
					interference_modeling::interference_model::INT_MODEL_constants[4];

	return alpha;

}
