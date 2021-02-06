#ifndef RT_SYCL_RTWEEKEND_HPP
#define RT_SYCL_RTWEEKEND_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <variant>
#include <vector>

#include <triSYCL/vendor/triSYCL/random/xorshift.hpp>

#include <build_parameters.hpp>

// Constants

constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385f;

// Utility Functions

inline float degrees_to_radians(float degrees) { return degrees * pi / 180.0f; }

struct LocalPseudoRNG {
  trisycl::vendor::trisycl::random::xorshift<> generator;
  std::uniform_real_distribution<float> distribution { 0.0f, 1.0f };
};

inline float random_float(LocalPseudoRNG& rng) {
  return rng.distribution(rng.generator);
}

inline float random_float(float min, float max, LocalPseudoRNG& rng) {
  // Returns a random real in (min,max).
  return min + (max - min) * random_float(rng);
}

// Common Headers
#include "ray.hpp"
#include "vec.hpp"

#endif
