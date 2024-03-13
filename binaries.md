
# Details of Binaries

## List of Binary Files

After successfully building each binary, the following binaries will be built under the 'bin/' directory. Below is a brief description of each binary.

- **frontend**: The frontend server, communicates with *N* backends and *M* clients
- **backend**: The backend server, communicates with *one* frontend
- **client**: The request generator, communicates with *one* frontend
- **proxy**: The implementation of a glet. Receives data and commands from backend server
- **standalone_scheduler**: Binary for executing the scheduler alone. Useful for genearating scheduling results statically and debugging scheduling decisions
- **standalone_inference**: Binary for executing inference on GPU. Useful for debugging and profiling utilization.

## Files Required by Binaries

### Pytorch models

All models must be stored under 'resource/models' directory saved as PyTorch script (.pt)files.

### Configuration JSON Files

The list below is a description for each file used for experimentation. Please adjust the contents of each file according to your purpose.

We have also provided some examples files under the '/resource' directory to give a better view of how to setup frontend servers, backend servers, and the request generator.

1. glet/resource/sim-config.json: Configuration for variables for scheduling.

2. glet/resource/mem-config.json: List of models and how much memory is consumed by each model.

3. glet/resource/latency.csv: List of batch,latency for every model supported.


4. glet/resource/proxy_config.json: List of models and required information for executing on proxy servers, used by both proxy and backend server.


5. glet/resource/rates.csv:  Initial rate for each model (used only in oracle scheduling for **frontend**).

6. glet/resource/config.json: Name of each application (group of models), and directory of each configuration file.

7. glet/resource/ModelList.txt: List of gpu,partition size, corresponding model, batch size and duty cycle(ms). The result of **standalone_scheduler** (used only in mps_static scheduling by **frontend** ).

8. glet/resource/device-config.json: Lists required files of per-device performance models.

9. glet/resource/BackendList.json: List of backend nodes, each IP address for data, control channel, type and number of GPUs.

10. glet/resource/input.txt: list of ImageNet input files for the request generator to read.

11. glet/resource/input-camera.txt: list of camera input images for the request generator to read.

12. glet/resource/input_config.json: list of input data specification for each model. Contains dimension and type for each model.

13. glet/resource/example_flux.txt: Request rate(RPS) for every interval. Interval can be adjusted in **client**

14. glet/resource/backend_docker_list.txt: List of IPs of the servers running backend servers and each corresponding private IP (used inside of the docker network).

15. glet/resource/1_28_28.txt: Batching overhead per batch size. We are planning for a more systematic fix for this overhead.