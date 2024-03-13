#ifndef _INTERFERENCE_MODEL_H__
#define _INTERFERENCE_MODEL_H__

#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <utility>
#include <cmath>

using namespace std;

typedef struct _interference_model_info{
    map<int, pair<float,float>> parameters;
	int id; 
    string name;
} interference_model_info;

namespace interference_modeling {
	class interference_model{
	public:
        void setup(string model_const_file, string util_file);
		double get_interference(string my_model, int my_batch, int my_partition, string your_model, int your_batch, int your_partition);
    private:
   		void _setup(string input_file1, string input_file2);
		vector<double> INT_MODEL_constants;
		map<pair<string, int>, vector<double>> UTIL;
	};
}
#endif
