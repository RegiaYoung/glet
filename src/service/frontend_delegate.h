#ifndef _FRONTED_DELEGATE_H__ 
#define _FRONTEND_DELEGATE_H_

#include <string>

class FrontendDelegate : public BaseDelegate {
    public: 
        FrotendDelegate();
        ~FrontendDelegate();
        int disconnect(std::string frontend_addr); 
        int sendOutput(int TEMP_PARAM);
    private: 
        std::string frontend_addr;
};


#else
#endif
