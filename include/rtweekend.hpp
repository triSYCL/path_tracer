#ifndef RT_SYCL_RTWEEKEND_HPP
#define RT_SYCL_RTWEEKEND_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <variant>
#include <vector>

#include <triSYCL/vendor/triSYCL/random/xorshift.hpp>

/// \todo Remove these global objects and move them into the kernel.
/// It cannot work with SYCL on device otherwise.
namespace {
std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
/// A fast random generator good on accelerator like FPGA
trisycl::vendor::trisycl::random::xorshift generator;
} // namespace

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;

// Utility Functions

inline float degrees_to_radians(float degrees) { return degrees * pi / 180.0f; }

inline float random_float() {
#ifndef USE_SYCL_COMPILER
  return distribution(generator);
#else
  return 0.5;
#endif
}

inline float random_float(float min, float max) {
  // Returns a random real in (min,max).
  return min + (max - min) * random_float();
}

// Common Headers
#include "ray.hpp"
#include "vec.hpp"

#endif
