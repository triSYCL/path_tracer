## RAY TRACER

Clone the reposity with:
```sh
git clone https://gitenterprise.xilinx.com/kranipet/Ray_Tracer.git
```
### Compiling with CMake

Create a `build` directory inside the cloned repo and jump into it.
From there, run:
```sh
cmake .. -DCMAKE_MODULE_PATH=.../triSYCL/cmake
```
Build the project with:
```sh
cmake --build .
```
This creates the executable . Now you can run the SYCL version of Ray Tracer with::
```sh
time RT_SYCL/sycl-rt >! result.ppm
```
This results in the image ``result.ppm`` produced by the Ray Tracer.

![img](https://media.gitenterprise.xilinx.com/user/1485/files/1ee6df00-fcd2-11ea-9ba0-d7675c81372d)
