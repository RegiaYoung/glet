#include <torch/script.h> // One-stop header
#include <torch/torch.h>

#include <iostream>
#include <string>
#include <memory>
#include <sys/time.h>
#include <pthread.h>
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <queue>
#include <condition_variable>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/un.h>


#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "socket.h"
#include "img_utils.h"
#include "torchutils.h"
#include "common_utils.h" //printTimeStamp moved to here
#include "exp_model.h"

#define IMAGENET_ROW 224
#define IMAGENET_COL 224

#define BATCH_BUFFER 10 // buffer used when reading, and chosing random data
#define FLUX_INTERVAL 20 // interval of changing random interval when fluctuating reqeust rate

namespace po = boost::program_options; 
using namespace cv;

// logging related variables
uint64_t *arrStartTime;
uint64_t *arrEndTime;
string DIST;


bool USE_FLUX;
bool USE_FLUX_FILE;
vector<int> gRandRate;
double gRandMean;
int gRate;
int gStdRate;
uint64_t gNumRequests;
int gBatchSize;
string gHostName;
int gPortNo;
int gSocketFD;
string gTask;

std::vector<Mat>input_data;
std::vector<torch::Tensor> input_img_tensor;
/*input related global variables*/
bool SKIP_RESIZE = false;
bool USE_IMG; // flag indicating whether to use custom image loader, imagenet, camera data use this
bool USE_MNIST; // flag indicating whether to use mnist data loader or not.
bool USE_NLP; // flag indicating whether to use nlp data or not FOR NOW we just use random data

//threads and initFunctions
pthread_t initRecvThread();  // inits thread for receiving results from the server
pthread_t initSendThread(); // sends to the server

void *recvRequest(void *vp);  
void *sendRequest(void *vp);

const char* gCharNetName;

std::vector<torch::Tensor> gMNISTData; // global vector storing mnist data
std::vector<torch::Tensor> gTokenData;

po::variables_map parse_opts(int ac, char** av) {
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Produce help message")
    ("task,t", po::value<std::string>()->default_value("alexnet"),
     "name of task: game, traffic, model name")
    ("portno,p", po::value<int>()->default_value(8080), "Server port")
    ("hostname,o",po::value<std::string>()->default_value("localhost"),"Server IP addr")
    ("batch,b", po::value<int>()->default_value(1),"size of batch to send") 
    ("requests,r",po::value<int>()->default_value(1),"how many requests are going to be issued to the server" ) 
    ("rate,m,",po::value<int>()->default_value(100)," rate (in seconds)")
    ("input,i",po::value<string>()->default_value("input.txt"),"txt file that contains list of inputs")
    ("skip_resize,sr", po::value<bool>()->default_value(false), "skip resizing?")
    ("root_data_dir", po::value<string>()->default_value("./"), "root directory which holds data for generator")
    ("dist", po::value<string>()->default_value("uni"), "option specifying which dist. to use e.g.) uni, exp, zipf")
    ("flux", po::value<bool>()->default_value(false), "flag value indicating whether to fluctuate or not")
    ("std_rate",po::value<int>()->default_value(0)," standard rate when flux is enabled (in seconds)")
    ("flux_file", po::value<string>()->default_value("no_file"), "file holding randon rates to experiment");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm); 
    po::notify(vm); 
    if (vm.count("help")) {
        std::cout << desc << "\n"; exit(1);   
    } 
    return vm;

}

std::vector<torch::Tensor> readMNISTData(string data_root){
    std::vector<torch::Tensor> ret_vector;
    auto mnist_data_loader = torch::data::make_data_loader(
    torch::data::datasets::MNIST(data_root+"/mnist").map(
    torch::data::transforms::Stack<>()),
      /*batch_size=*/1);
    for (torch::data::Example<>& batch : *mnist_data_loader) {
      torch::Tensor temp = batch.data;
	ret_vector.push_back(temp);
    }
    return ret_vector;
}

std::vector<Mat> readImgData(const char *path_to_txt, int batch_size, string data_root_dir)
{
    std::ifstream file(path_to_txt);
    std::string img_file;
    std::vector<Mat> imgs;
    for (int i = 0; i < batch_size; i++)
    {
        if (!getline(file, img_file))
            break;
        Mat img;
        string full_name = data_root_dir + "/" + img_file;
        img = imread(full_name, IMREAD_COLOR);
        if (img.empty())
            LOG(ERROR) << "Failed to read  " << full_name << "\n";
        //#ifdef DEBUG
        LOG(ERROR) << "dimensions of " << full_name << "\n";
        LOG(ERROR) << "size: " << img.size() << "\n"
                   << "row: " << img.rows << "\n"
                   << "column: " << img.cols << "\n"
                   << "channel: " << img.channels() << "\n";
        //#endif
        imgs.push_back(img);
        }
        if (gBatchSize < 1) {LOG(FATAL) << "No images read!"; exit(1);}

        std::vector<Mat>::iterator it;
#ifdef DEBUG
    LOG(ERROR) << "read " <<gBatchSize << "images \n";
#endif
    return imgs;
    
}

std::vector<int> readTraceFile(string path_to_trace_file){
	std::ifstream infile(path_to_trace_file);
	vector<int> retVec;
	std::string line;
	while(std::getline(infile,line))
	{
		std::istringstream iss(line);
		float a;
		if (! (iss >> a)) {break;}
 	        retVec.push_back(int(a));

	}
	return retVec;
}

torch::Tensor getInput(const char* netname){
    torch::Tensor input;
#ifdef DEBUG
        printf("get input for %s \n", netname);
#endif
    if (strcmp(netname, "lenet1") == 0)
    {
        input = torch::randn({gBatchSize,1,28, 28});
    }
    else if(strcmp(netname,"ssd-mobilenetv1")==0){
        input = torch::randn({gBatchSize,3,300, 300});
    }
    else if(strcmp(netname,"resnet50")==0 ||strcmp(netname,"googlenet")==0 ||strcmp(netname,"vgg16")==0 ){
        input = torch::randn({gBatchSize,3,224, 224});
    }
    else if(strcmp(netname, "bert")==0){
        auto options = torch::TensorOptions().dtype(torch::kInt64);
        input = torch::randint(/*high=*/500, {gBatchSize, 14},options);
    }
    else{
        printf("unrecognized task: %s \n", netname);
        exit(1);
    }
    return input;

}

void generateTokenIDs(vector<torch::Tensor> &global_mem, string task){
    cout << "called!" <<endl;
    for(int i =0; i < gBatchSize * BATCH_BUFFER; i++){
        global_mem.push_back(getInput(task.c_str()));
    }    
}

void setupGlobalVars(po::variables_map &vm){
    SKIP_RESIZE = vm["skip_resize"].as<bool>(); 
    gRate =  vm["rate"].as<int>();
    assert(gRate!=0);
    gRandMean = double(1.0)/ gRate;
    gTask  = vm["task"].as<std::string>(); 
    gCharNetName = gTask.c_str();
    USE_FLUX_FILE=false;
    USE_FLUX = vm["flux"].as<bool>();
    if(USE_FLUX){
	    if(vm["flux_file"].as<string>() != "no_file"){
	        USE_FLUX_FILE=true;
		    cout << "USING flux file: " << vm["flux_file"].as<string>() << endl;
		    gRandRate=readTraceFile(vm["flux_file"].as<string>());
	    }
        else{
            std::cout<<"Must specify flux file when flag flux is set!" << std::endl;
            exit(1);
        }

    }
	gStdRate=0;

    DIST= vm["dist"].as<string>();
    gNumRequests=vm["requests"].as<int>();
    if(gTask == "lenet1"){
        USE_IMG=0;
        USE_MNIST=1;
        USE_NLP=0;
    }
    else if(gTask == "game"){
        USE_IMG=1;
        USE_MNIST=1;
        USE_NLP=0;
    }
    else if(gTask == "bert"){
        USE_IMG=0;
        USE_MNIST=0;
        USE_NLP=1;   
    }
    else { // traffic, ssd-mobilenetv1, and etc.
        USE_IMG=1;
        USE_MNIST=0;
        USE_NLP=0;   

   }
    gBatchSize= vm["batch"].as<int>();
    assert(gBatchSize!=0);
    gHostName = vm["hostname"].as<string>();
    gPortNo = vm["portno"].as<int>();

}

void readInputData(po::variables_map &vm){
    /*reads image data*/
    // reads BATCH_BUFFER times more images than it will be read
    if(USE_IMG){
        std::string path_to_txt = vm["input"].as<std::string>();
        input_data = readImgData(path_to_txt.c_str(), gBatchSize*BATCH_BUFFER, vm["root_data_dir"].as<string>());
        cout << "Read " << input_data.size() << " images " << endl;
        std::vector<Mat> inputImages;
        for (unsigned int i = 0; i < input_data.size(); i++)
        {
            inputImages.clear();
            torch::Tensor input;
            inputImages.push_back(input_data[i]);
            if (!SKIP_RESIZE)
                input = serialPreprocess(inputImages, IMAGENET_ROW, IMAGENET_COL);
            else
                input = convert_images_to_tensor(inputImages);
            input_img_tensor.push_back(input);
	    }
    }

    /*reads mnist data set*/
    if(USE_MNIST){
	#ifdef  DEBUG
	    cout << "READ MNIST" << endl;
	#endif
        string DATA_ROOT= vm["root_data_dir"].as<string>();
        gMNISTData = readMNISTData(DATA_ROOT);
    }

    /*generates NLP data*/
    // for now we generate a random vector for input data(simulating token IDs)
    if(USE_NLP){
        generateTokenIDs(gTokenData, gTask);
    }


}

int main(int argc, char** argv) {
    printTimeStampWithName(gCharNetName, "START PROGRAM");
    
    /*get parameters for this program*/
    po::variables_map vm = parse_opts(argc, argv);
    setupGlobalVars(vm);
    #ifdef DEBUG
    cout << "Finished setting up global variables" << endl;
    #endif
    readInputData(vm);
    #ifdef DEBUG
    cout << "Finished reading input data" << endl;
    #endif
 
    float *preds = (float*)malloc(gBatchSize * sizeof(float));
    /*start threads*/
    pthread_t sendThreadID;
    pthread_t recvThreadID;
    gSocketFD = CLIENT_init((char*)gHostName.c_str(), gPortNo, false);
    if (gSocketFD < 0) exit(0);
    int opt_val=1;
    setsockopt(gSocketFD,IPPROTO_TCP,TCP_NODELAY, (void*)&opt_val, sizeof(int));
    printf("connected to server!\n");
    arrStartTime = (uint64_t *)malloc(gNumRequests * sizeof(uint64_t));
    arrEndTime = (uint64_t *)malloc(gNumRequests * sizeof(uint64_t));
       
    sendThreadID = initSendThread();
    recvThreadID = initRecvThread();
    pthread_join(sendThreadID, NULL);
    pthread_join(recvThreadID, NULL);
    SOCKET_close(gSocketFD, false);
    for( int i=0; i<gNumRequests; i++){
        cout << "Respond Time: " << to_string(double(arrEndTime[i]-arrStartTime[i])/1000000) << " for Request ID "<< i<<endl;
    }
    printTimeStampWithName(gCharNetName, "END PROGRAM");
    return 0;
}

int gCount=0;
double getNextRandRate(vector<int> &rand_rates){
	int index = gCount++ % rand_rates.size();
#ifdef DEBUG
	cout << "[getNextRandRate] index: " << index << ", rate: " << rand_rates[index] << endl;
#endif 
	if(rand_rates[index]<0.001) return 0.0;
	return 1.0 / double(rand_rates[index]);
}

pthread_t initSendThread(){
    pthread_attr_t attr;
    pthread_attr_init(&attr);   
    pthread_attr_setstacksize(&attr, 1024 * 1024); 
    pthread_t tid;
    if (pthread_create(&tid, &attr, sendRequest, NULL) != 0)   
        LOG(ERROR) << "Failed to create a request handler thread.\n";   
    return tid;
}


void *sendRequest(void *vp){
    ExpModel *pRandModel = new ExpModel();
    int ret;
    // send request name (before sending data)a
    SOCKET_send(gSocketFD, (char*)gTask.c_str(), MAX_REQ_SIZE, false);
    // send how many inputs to expect for this app
    int NUM_INPUT=0; // number of tesnors to send, according to application
    NUM_INPUT = (USE_IMG && 1) + (USE_MNIST && 1) + (USE_NLP && 1);
    printf("number of inputs to send: %d \n", NUM_INPUT);
    SOCKET_txsize(gSocketFD, NUM_INPUT);
    uint64_t start; 
    uint64_t end;
    double waitMilli;
    uint64_t last_flux=getCurNs();
    double l_rand_mean;
    if(USE_FLUX_FILE)
	    l_rand_mean = getNextRandRate(gRandRate);
    else
    	l_rand_mean = gRandMean;
    printTimeStampWithName(gCharNetName, "START send_request");
    for (uint64_t i =0; i < gNumRequests; i++){
	    if(USE_FLUX){
		    // if rate is fluctuating follow, do it until FLUX_INTERVAL seconds
		   if(double(getCurNs() - last_flux)/1000000000 > FLUX_INTERVAL){
		    double old_rand_mean = l_rand_mean;
		    if(USE_FLUX_FILE){
			    l_rand_mean = getNextRandRate(gRandRate);             
#ifdef DEBUG
			    cout<<__func__<<": old mean interval: " << old_rand_mean << "new mean interval"<< l_rand_mean << endl;
#endif
		    }
	        last_flux=getCurNs();	   
		   }
		}
		if(DIST == "exp" ) {
            double rand_mean=pRandModel->randomExponentialInterval(l_rand_mean,1,1);
            waitMilli = rand_mean * 1000;
        }
		else if(DIST == "uni") waitMilli = l_rand_mean  * 1000;
		else{
			printf("unrecognized distribution: %s, exiting now \n", DIST.c_str());
			exit(1);
		}

#ifdef DEBUG
		printf("waiting for %lf milliseconds \n", waitMilli);
#endif   
 	    usleep(waitMilli * 1000);
        //form request(s)
        #ifdef DEBUG
	    start = getCurNs();
        #endif
        if(USE_IMG){
            std::vector<Mat> inputImages;
            int rand_val = std::rand() % (gBatchSize * BATCH_BUFFER);
            torch::Tensor input = input_img_tensor[rand_val];
#ifdef DEBUG   
        printf("IMG input tensor dimension: ");
        for(int j =0; j<input.dim(); j++ )
            printf("%lu, ",input.size(j));
        printf("\n");
#endif 
        #ifndef NO_NET 	
                sendTensorFloat(gSocketFD ,i ,input);
        #endif
#ifdef DEBUG
	cout <<"sending " << i+1 << "as tid to serverr "<<endl;
#endif
        SOCKET_txsize(gSocketFD, i+1);

        }
        if(USE_MNIST){
            int rand_val = std::rand() % (gBatchSize * BATCH_BUFFER);
            torch::Tensor input = gMNISTData[rand_val];
#ifdef DEBUG   
        printf("MNIST input tensor dimension: ");
        for(int j =0; j<input.dim(); j++ )
            printf("%lu, ",input.size(j));
        printf("\n");
#endif 
        #ifndef NO_NET
            sendTensorFloat(gSocketFD, i, input);
        #endif
#ifdef DEBUG
	cout <<"sending " << i+1 << "as tid to serverr "<<endl;
#endif
        SOCKET_txsize(gSocketFD, i+1);

        }
        if(USE_NLP){

            int rand_val = std::rand() % (gBatchSize * BATCH_BUFFER);
            torch::Tensor input = gTokenData[rand_val];
#ifdef DEBUG   
        printf("NLP input tensor dimension: ");
        for(int j =0; j<input.dim(); j++ )
            printf("%lu, ",input.size(j));
        printf("\n");
#endif 
        #ifndef NO_NET
            sendTensorLong(gSocketFD, i, input);
        #endif
 #ifdef DEBUG
	cout <<"sending " << i+1 << "as tid to serverr "<<endl;
#endif
         SOCKET_txsize(gSocketFD, i+1);
             
        }
       // need to send task id to frontend
	    arrStartTime[i] = getCurNs();
        #ifdef DEBUG
	    end = getCurNs(); 
	    cout << "Sending Time: " << to_string(double(end-start)/1000000) << " for Request ID "<< i+1<<endl;
        #endif
        }
	    delete pRandModel;
	return (void*)0;
    
}

pthread_t initRecvThread() {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024); // set memory size, may need to adjust in the future, for now set it to 1MB
    pthread_t tid;
    if (pthread_create(&tid, &attr, recvRequest, NULL) != 0)
        LOG(ERROR) << "Failed to create a request handler thread.\n";
    return tid;

}
void* recvRequest(void *vp)
{
    int taskID;
    int opt_val = 1;
    int taskcnt=0;
    while (taskcnt < gNumRequests)
    {
        taskID = SOCKET_rxsize(gSocketFD);
        if (setsockopt(gSocketFD, IPPROTO_TCP, TCP_QUICKACK, &opt_val, sizeof(int)))
        {
            perror("recvResult setsocketopt");
        }
#ifdef DEBUG
    cout << "DEBUG: recieved ACK for task ID :" << taskID << endl;
    cout << "DEBUG: received " << taskcnt+1 << " so far" << endl;
#endif 
    // taskID is added by 1 when sent to server
    arrEndTime[taskID-1] = getCurNs();
    taskcnt++;
  } 
  printTimeStampWithName(gCharNetName, "END recv_request");
  return (void*)0;
}
