#ifndef RT_SYCL_VEC_HPP
#define RT_SYCL_VEC_HPP

#include "rtweekend.hpp"
#include <cmath>
#include <iostream>

#include "sycl.hpp"
using real_t = float;

// type aliases for float3 - vec, point and color
using point = sycl::float3;
using color = sycl::float3;
using vec = sycl::float3;

// vec Utility Functions
float length_squared(const vec& v) {
  return sycl::fma(v.x(), v.x(),
                   sycl::fma(v.y(), v.y(), fma(v.z(), v.z(), 0.0f)));
}

vec randomvec(LocalPseudoRNG& rng) {
  return vec(random_float(rng), random_float(rng), random_float(rng));
}

vec randomvec(float min, float max, LocalPseudoRNG& rng) {
  return vec(random_float(min, max, rng), random_float(min, max, rng),
             random_float(min, max, rng));
}

inline std::ostream& operator<<(std::ostream& out, const vec& v) {
  return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

// Missing operator from the SYCL specification for now
vec operator-(const vec& u) { return vec(-u.x(), -u.y(), -u.z()); }

// Compute a unit vector from a non-null vector
inline vec unit_vector(const vec& v) { return v / sycl::length(v); }

// Make a random unit vector
vec random_unit_vector(LocalPseudoRNG& rng) {
  auto a = random_float(0, 2 * pi, rng);
  auto z = random_float(-1, 1, rng);
  auto r = sycl::sqrt(1 - z * z);
  return vec(r * sycl::cos(a), r * sycl::sin(a), z);
}

// Compute a random point inside a unit sphere at origin
vec random_in_unit_sphere(LocalPseudoRNG& rng) {
  while (true) {
    auto p = randomvec(-1, 1, rng);
    if (length_squared(p) >= 1)
      continue;
    return p;
  }
}

// Compute reflected ray's direction
vec reflect(const vec& v, const vec& n) { return v - 2 * sycl::dot(v, n) * n; }

// Compute random point in a unit disk
vec random_in_unit_disk(LocalPseudoRNG& rng) {
  while (true) {
    auto p = vec(random_float(-1, 1, rng), random_float(-1, 1, rng), 0);
    if (length_squared(p) >= 1)
      continue;
    return p;
  }
}

// Computes refracted ray's direction based on refractive index
vec refract(const vec& uv, const vec& n, float etai_over_etat) {
  auto cos_theta = sycl::fmin(-sycl::dot(uv, n), 1.0f);
  vec r_out_perp = etai_over_etat * (uv + cos_theta * n);
  vec r_out_parallel =
      -sycl::sqrt(sycl::fabs(1.0f - length_squared(r_out_perp))) * n;
  return r_out_perp + r_out_parallel;
}

#endif
