#ifndef RT_SYCL_VEC_HPP
#define RT_SYCL_VEC_HPP

#include "rtweekend.hpp"
#include <cmath>
#include <iostream>

using real_t = float;

// vec Utility Functions
inline float length_squared(const vec& v) {
  return sycl::fma(v.x(), v.x(), sycl::fma(v.y(), v.y(), v.z() * v.z()));
}

inline std::ostream& operator<<(std::ostream& out, const vec& v) {
  return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

// Missing operator from the SYCL specification for now
vec operator-(const vec& u) { return vec(-u.x(), -u.y(), -u.z()); }

// Compute a unit vector from a non-null vector
inline vec unit_vector(const vec& v) { return v / sycl::length(v); }

// Compute reflected ray's direction
vec reflect(const vec& v, const vec& n) { return v - 2 * sycl::dot(v, n) * n; }

// Computes refracted ray's direction based on refractive index
vec refract(const vec& uv, const vec& n, float etai_over_etat) {
  auto cos_theta = sycl::fmin(-sycl::dot(uv, n), 1.0f);
  vec r_out_perp = etai_over_etat * (uv + cos_theta * n);
  vec r_out_parallel =
      -sycl::sqrt(sycl::fabs(1.0f - length_squared(r_out_perp))) * n;
  return r_out_perp + r_out_parallel;
}

#endif
