# Path tracer

This is an experimental path tracer using C++20 and SYCL for
acceleration.

![img](doc/SmokeSphere.jpg)

This is using triSYCL for now but it might work even better on some
other SYCL implementations. Contributions and optimizations welcome!

The main focus here is to replace classic features like pointers or
dynamic polymorphism that does not work (well) on heterogeneous with
more modern constructs such as `std::variant` and `std::visit`.

## Features

- motion blur;
- depth of field;
- materials:
  - smoke;
  - textures;
  - Lambertian material;
  - dielectric material;
  - metallic roughness;
  - light;
- geometry:
  - spheres;
  - triangles;
  - x/y/z-rectangles;
  - boxes;

## Required dependancies

In addition to triSYCL, this project requires the following dependancies:

 - the [stb](https://github.com/nothings/stb) image manipulation library;

On Linux, there's a good chance it can be installed with your package manager :

On Ubuntu/Debian :

```sh
sudo apt install libstb-dev
```

On Archlinux, install the [stb](https://aur.archlinux.org/packages/stb) package from AUR.

## Compiling

Clone the reposity such as with:
```sh
git clone git@github.com:triSYCL/path_tracer.git
```

Create a `build` directory for example inside the cloned repository
and jump into it.

From there, assuming you have the https://github.com/triSYCL/triSYCL
repository somewhere, run:
```sh
cmake .. -DCMAKE_MODULE_PATH=<absolute_path_to>/triSYCL/cmake
```

The project defaults to a Release build configuration.
If you wish to debug, configure your build settings as follow:

```sh
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MODULE_PATH=<absolute_path_to>/triSYCL/cmake
```

It is also possible to build with https://github.com/triSYCL/sycl or https://github.com/intel/llvm/tree/sycl
```sh
cmake .. --DCMAKE_MODULE_PATH=<absolute_path_to>/triSYCL/cmake -DTRISYCL_OPENMP=OFF -DSYCL_CXX_COMPILER=<path_to_sycl_build>/bin/clang++ -DSYCL_DEVICE_TRIPLE=fpga64_sw_emu
# the triple fpga64_sw_emu is only available with https://github.com/triSYCL/sycl
```

The triSYCL cmake path and options are required for some cmake macros they define.

For FPGA execution you might add `-DUSE_SINGLE_TASK=ON` on the
previous `cmake` configuration to use a SYCL execution based on a
`.single_task()` instead of `.parallel_for()`, probably more efficient
on FPGA.

Build the project with:
```sh
cmake --build . --verbose --parallel `nproc`
```
This creates the executable.

## Running

Now you can run the path tracer with:
```sh
time ./sycl-rt 800 480 50 100
```
This results in the image ``out.png`` produced by the path tracer.

Parameters to the executable are 

+ The output image width (here 800)
+ The output image height (here 480)
+ The maximum bouncing depth of the ray (here 50)
+ The number of samples per pixel (here 100)


## Bibliography

Some references that were tremendously useful in writing this project:

1. [Path tracing](https://en.wikipedia.org/wiki/Path_tracing)

2. [Ray Tracing in One Weekend - Peter
Shirley](https://raytracing.github.io/books/RayTracingInOneWeekend.html)

3. [Ray Tracing: The Next Week - Peter
Shirley](https://raytracing.github.io/books/RayTracingTheNextWeek.html)

4. [Ray-tracing in a Weekend with SYCL: Basic sphere tracing -- Georgi
Mirazchiyski](https://www.codeplay.com/portal/blogs/2020/05/19/ray-tracing-in-a-weekend-with-sycl-basic-sphere-tracing.html)

5. [Ray-tracing in a Weekend with SYCL Part 2: Pixel sampling and
   Material tracing -- Georgi
   Mirazchiyski](https://www.codeplay.com/portal/blogs/2020/06/19/ray-tracing-in-a-weekend-with-sycl-part-2-pixel-sampling-and-material-tracing.html)

6. [CppCon 2018: Mateusz Pusz, “Effective replacement of dynamic
    polymorphism with
    std::variant”](https://www.youtube.com/watch?v=gKbORJtnVu8)

7. [Bartek's coding blog: Runtime Polymorphism with std::variant and
   std::visit](https://www.bfilipek.com/2020/04/variant-virtual-polymorphism.html)

8. [Intersection of a Ray/Segment with a
   Triangle](http://geomalgorithms.com/a06-_intersect-2.html)
