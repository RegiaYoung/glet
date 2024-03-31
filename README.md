# GLET-Orion-MOD
多GPU的异构ML推理负载细粒度调度框架，基于glet架构添加负载类型维度。



To maximize the resource efficiency of inference servers, we proposed a key mechanism to exploit hardware support for spatial
partitioning of GPU resources. With the partitioning mechanism, a new abstraction layer of GPU resources is created with
configurable GPU resources. The scheduler assigns requests
to virtual GPUs, called **Glets**, with the most effective amount
of resources. The prototype framework auto-scales the required number of GPUs for a given workloads, minimizing the cost for cloud-based inference servers.
The prototype framework also deploys a remedy for potential interference
effects when two ML tasks are running concurrently in a GPU.

# Evaluated Environment:

## OS/Software
- Ubuntu 22.04
- Linux Kernel 6.5.0
- CUDA 11.4
- cuDNN 9.0
- PyTorch 1.11

## Hardware

The prototype was evaluated with multi-GPU server with the following hardware:

- RTX A6000 with 48GB global memory (but reccommend to use Nvidia 20s/30s GPU)
- intel Xeon(R) Silver 4210R
- Servers connected with 10 GHz Ethernet (virtual)

# Getting Started with Docker Image:

We highly recommend using the docker image we provide since it contains all required libraries, code and scritps.
The docker image can be obtained pulled from docker hub.
> docker pull sbchoi/glet-server:latest

## Prerequisites for running docker image

1. docker ver >= 20

2. nvidia-docker, for utilizing NVIDIA GPUs.

## Steps for executing toy example
1. After extracting glet.zip, go to 'glet/scripts' directory
> cd glet/scripts

2. Start MPS with **sudo** (or else MPS will not be available to docker)
> sudo ./start_MPS.sh

3. Create an overlay network with dockers. The script will create an attachable network for subnet range 10.10.0.0/24.

> cd ../docker_files  
./create_docker_overlay_network.sh

4. On separate terminals, run backend servers on each terminal (private IP address should match those listed in 'glet/resource/Backend_docker.json'). Each script will setup a backend server for GPU 0 and GPU 1 respectively.
> ./run_interactive_backend_docker.sh 0 10.10.0.3  
./run_interactive_backend_docker.sh 1 10.10.0.4

5. On another terminal, run the frontend server. 
> ./run_interactive_frontend_docker.sh oracle 10.10.0.20 1

6. On another terminal, run the clients.
> ./run_interactive_client_docker.sh 10.10.0.21 test-run

7. As an result 'log.txt' will be created under 'glet/scripts', which is a logging file that has various results when each request was executed in the server.

8. Terminate process on each terminal to end the example.

9. (Optional) in order to shutdown MPS, execute 'shutdown_MPS.sh' under 'glet/scripts'
> sudo ../shutdown_MPS.sh


## Steps for producing custom docker server image

We also provide the 'base' image we have used. Note that all prerequisites listed in [Getting Started with Native Binaries](#Getting-Started-with-Native-Binaries:) must be installed.

1. Pull the base image.
> docker pull sbchoi/glet-base:latest

2. Execute building script stored under 'glet/docker_files'. 
> cd glet/docker_files && ./build_backend_docker.sh

This will build an image with 'Dockerfile' and the image will be tagged with 'glet-server:latest'.

# Getting Started with Native Binaries:

## Prerequisites

Install the following libraries and drivers to build the prototype

- LibTorch(PyTorch library for C++) = 1.10 (tar file provided)

- CUDA >= 10.2

- CUDNN >= 7.6

- Boost >= 1.6 (script provided)

- opencv >= 4.0 (script provided)

- cmake >= 3.19, use cmake to build binaries (script provided)

when using Ampere GPU, you need to use CUDA11 and newer LibTorch, you may need to modify some torch API to arrange it.

## Steps for building

1. Download and extract glet.zip.

> unzip glet.zip

2. Install all dependencies. The script contains some **'sudo'** instructions. Make sure your account has sudo privilege.

> cd glet/  
./install_dependencies.sh

3. Place LibTorch directory under the root('glet/') of this project directory, as 'glet/libtorch' along with other directories. (Or you can just unpack the libtorch.tar we have provided)

> tar xvf libtorch.tar

4. Go to the 'scripts' directory and execute the building script.
> cd scripts   
./build_all.sh

5. All successfully compiled binaries under the 'glet/bin' directory. 

Please refer to  **'binaries.md'** for more details of each binary.

## Example scripts

We have provided example scripts for the binaries. Please note that all of the following scripts are also available in the docker images we provide.

1.  glet/script/**executeFrontend.sh**: script for running frontend server, accepts scheduler name and a flag for enabling dropping requests as parameters

Example) Frontend server uses mps_static scheduler and wishes to drop requests if necessary.

> ./executeFrontend.sh mps_static 1

2. glet/script/**executeBackend.sh**: script for running frontend server, accepts the number of GPUs to manage as parameter

Example) Backend server manages two GPUs.

>./executeBackend.sh 2

3. glet/script/**execReqGen.sh**: script for generating requests to the frontend server.

Example) Generate requests for resnet50, 1000 requests, batch size of 1, 100 rps, with exponential distribution intervals, no file with predefined rates

> ./execReqGen.sh resnet50 1000 1 100 1

4. glet/resource/proxy/**startProxy.sh**: script for starting proxy, the script is called from the backend server. *Please refer to src/service/proxy_ctrl.cpp:bootProxy*

Example) Start a proxy on GPUID 0, with 50% resource, deduplication enabled, using ../ModelList.txt as model list, being the first partition, for total 2 GPUs in the server, where each GPU has among 7 types of partition
>./startProxy.sh 0 50 1 ../ModelList.txt 0 2 7


5. glet/script/**executeLocal_example.sh**: script for executing tasks with standalone_inference. Uses a locally installed GPU to execute inference.

Example) Excute resnet50 model, 1000 requests, with batch size of 16, waiting 0.1s between executions, for GPU 50% resource

>./executeLocal_example.sh resnet50 1000 16 0.1 50

6. glet/script/**executeScheduler_example.sh**: script for scheduling tasks given in file given for parameter --task_config. Please refer to *binary.md* for more details of more files.

Example) schedule tasks on ../resource/rates.csv, configured by ../resource/sim-config.json

>./executeScheduler_example.sh


