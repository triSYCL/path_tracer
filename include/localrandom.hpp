#ifndef LOCALRANDOM_HPP
#define LOCALRANDOM_HPP

#include <xorshift.hpp>

#include "primitives.hpp"

namespace raytracer::random {
class LocalPseudoRNG {
 public:
  inline LocalPseudoRNG(std::uint32_t init_state = xorshift<>::initial_state)
      : generator { init_state } {}

  // Returns a random float in 0.f 1.
  inline real_t real() {
    constexpr real_t scale = 1.f / (uint64_t { 1 } << 32);
    return generator() * scale;
  }

  // Returns a random float in min, max
  inline real_t real(real_t min, real_t max) {
    // TODO use FMA ?
    return min + (max - min) * real();
  }

  // Returns a random vector with coordinates in 0.f 1.
  inline vec vec_t() { return { real(), real(), real() }; }

  // Returns a random vec with coordinates in min, max
  inline vec vec_t(real_t min, real_t max) {
    auto scale = max - min;
    return vec_t() * scale + min;
  }

  // Returns a random unit vector
  inline vec unit_vec() {
    auto x = real(-1.f, 1.f);
    auto maxy = sycl::sqrt(1 - x * x);
    auto y = real(-maxy, maxy);
    auto absz = sycl::sqrt(maxy * maxy - y * y);
    auto z = (real() > 0.5) ? absz : -absz;
    return vec(x, y, z);
  }

  // Returns a random vector in the unit ball of usual norm
  inline vec in_unit_ball() {
    // Polar coordinates r, theta, phi
    auto r = real();
    auto theta = real(0, 2 * pi);
    auto phi = real(0, pi);

    auto plan_seed = r * sycl::sin(phi);
    auto z = r * sycl::cos(phi);

    return { plan_seed * sycl::cos(theta), plan_seed * sycl::sin(theta), z };
  }

  // Return a random vector in the unit disk of usual norm in plane x, y
  inline vec in_unit_disk() {
    auto x = real(-1.f, 1.f);
    auto maxy = sycl::sqrt(1 - x * x);
    auto y = real(-maxy, maxy);
    return { x, y, 0.f };
  }

 private:
  xorshift<> generator;
};
} // namespace raytracer::random

#endif