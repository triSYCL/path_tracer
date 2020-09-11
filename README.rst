RAY TRACER
++++++++++

Clone the reposity with::
  
  git clone https://gitenterprise.xilinx.com/kranipet/Ray_Tracer.git

Compiling with CMake
--------------------

Create a ``build`` directory inside the cloned repo and jump into it. From there, run::

  cmake ..

Build the project with::

  cmake --build .

This creates the executable . Now you can run the SYCL version of Ray Tracer with::
  
  ./RT_SYCL/main >! result.ppn

This results in the image ``result.ppm`` produced by the Ray Tracer.