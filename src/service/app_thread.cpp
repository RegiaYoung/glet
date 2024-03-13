#include "thread.h"
#include "input.h"
extern FILE* pAppLogFile;
extern SysMonitor ServerState;
map<string,deque<shared_ptr<AppInstance>>*> PerAppQueue;
map<string,mutex*> PerAppMtx;
map<string,condition_variable*> PerAppCV;


void configureAppSpec(string config_json,string resource_dir, SysMonitor &state){
  configAppSpec(config_json.c_str(), state, resource_dir);
  for(unsigned int i =0; i < state.AppSpecVec.size(); i++){
    initAppThread(&state.AppSpecVec[i]);
    usleep(500*1000);
  }
}

void addtoAppQueue(shared_ptr<AppInstance> appinstance, bool dropped){
    string StrName = appinstance->getName();
    if(dropped){
        appinstance->setDropped(true);
    }
    else
        appinstance->setDropped(false);
    #ifdef DROP_DEBUG
        cout<<" task id : " << appinstance->getTaskID() << " dropped: " << appinstance->isDropped()<<endl;
     #endif 
    PerAppMtx[StrName]->lock();
    PerAppQueue[StrName]->push_back(appinstance);
    PerAppMtx[StrName]->unlock();
    PerAppCV[StrName]->notify_one();
}

pthread_t initAppThread(AppSpec *App){
    pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 8*1024 * 1024); // set memory size, may need to adjust in the future, for now set it to 1MB
        pthread_t tid;
        if (pthread_create(&tid, &attr, initApp, (void*)App) != 0)
            LOG(ERROR) << "Failed to create a initApp thread.\n";
        return tid;
}

void* initApp(void *args){
	AppSpec* App = (AppSpec*)args;
    string AppName = App->getName();
    PerAppQueue[AppName]= new deque<shared_ptr<AppInstance>>;
    PerAppMtx[AppName]=  new mutex();
    PerAppCV[AppName]=   new condition_variable();
    deque<shared_ptr<AppInstance>> *pAppQueue=PerAppQueue[AppName];
    int CompletedTasks=0;
    while(1){
        unique_lock<mutex> lk(*PerAppMtx[AppName]);
        PerAppCV[AppName]->wait(lk, [&pAppQueue]{return pAppQueue->size();});
        shared_ptr<AppInstance> task = pAppQueue->front();
        pAppQueue->pop_front();
        lk.unlock();
        if(!ServerState.IS_LOCAL){
            int ret = App->sendOutputtoClient(task->getSocketFD(), task->getTaskID());
            if(ret != EXIT_SUCCESS) {
                printf("ERROR in sending output to client !\n");
            }
        }
        task->writeToLog(pAppLogFile);
    } //infinite while loop
	return (void*)0;
}
