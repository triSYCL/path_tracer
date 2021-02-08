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

#include <sycl.hpp>
#include <triSYCL/vendor/triSYCL/random/xorshift.hpp>

#include <build_parameters.hpp>

// Constants

constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385f;

// type aliases for float3 - vec, point and color
using point = sycl::float3;
using color = sycl::float3;
using vec = sycl::float3;

// Utility Functions

inline float degrees_to_radians(float degrees) { return degrees * pi / 180.0f; }

class LocalPseudoRNG {
 public:
  inline LocalPseudoRNG(std::uint32_t init_state = trisycl::vendor::trisycl::random::xorshift<>::initial_state)
      : generator{init_state} {}

  // Returns a random float in 0., 1.
  inline float float_t() {
    constexpr float scale = 1./ (uint64_t{1} << 32);
    return generator() * scale;
  }

  // Returns a random float in min, max
  inline float float_t(float min, float max) {
    // TODO use FMA ?
    return min + (max - min) * float_t();
  }

  // Returns a random vector with coordinates in 0., 1.
  inline vec vec_t() { return { float_t(), float_t(), float_t() }; }

  // Returns a random vec with coordinates in min, max
  inline vec vec_t(float min, float max) {
    auto scale = max - min;
    return vec_t() * scale + min;
  }

  // Returns a random unit vector
  inline vec unit_vec() {
    auto x = float_t(-1., 1.);
    auto maxy = sycl::sqrt(1 - x * x);
    auto y = float_t(-maxy, maxy);
    auto absz = sycl::sqrt(maxy * maxy - y * y);
    auto z = (float_t() > 0.5) ? absz : -absz;
    return vec(x, y, z);
  }

  // Returns a random vector in the unit ball of usual norm
  inline vec in_unit_ball() {
    // Polar coordinates r, theta, phi
    auto r = float_t();
    auto theta = float_t(0, 2 * pi);
    auto phi = float_t(0, pi);

    auto plan_seed = r * sycl::sin(phi);
    auto z = r * sycl::cos(phi);

    return { plan_seed * sycl::cos(theta), plan_seed * sycl::sin(theta), z };
  }

  // Return a random vector in the unit disk of usual norm in plane x, y
  inline vec in_unit_disk() {
    auto x = float_t(-1., 1.);
    auto maxy = sycl::sqrt(1 - x * x);
    auto y = float_t(-maxy, maxy);
    return { x, y, 0. };
  }

 private:
  trisycl::vendor::trisycl::random::xorshift<> generator;
};

// Common Headers
#include "ray.hpp"
#include "vec.hpp"

#endif
