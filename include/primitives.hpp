#ifndef PRIMITIVES_HPP
#define PRIMITIVES_HPP

#include "sycl.hpp"

namespace raytracer {
using real_t = float;

// Constants

constexpr real_t infinity = std::numeric_limits<real_t>::infinity();
constexpr real_t pi = 3.1415926535897932385f;

// type aliases for float3 - vec, point and color
using real_vec = sycl::float3;

using point = real_vec;
using color = real_vec;
using vec = real_vec;

// Utility Functions
inline real_t degrees_to_radians(real_t degrees) { return degrees * pi / 180.0f; }

// vec Utility Functions
inline real_t length_squared(const vec& v) {
  return sycl::fma(v.x(), v.x(), sycl::fma(v.y(), v.y(), v.z() * v.z()));
}

// Missing operator from the SYCL specification for now
vec operator-(const vec& u) { return vec(-u.x(), -u.y(), -u.z()); }

// Compute a unit vector from a non-null vector
inline vec unit_vector(const vec& v) { return v / sycl::length(v); }

// Compute reflected ray's direction
vec reflect(const vec& v, const vec& n) { return v - 2 * sycl::dot(v, n) * n; }

// Computes refracted ray's direction based on refractive index
vec refract(const vec& uv, const vec& n, real_t etai_over_etat) {
  auto cos_theta = sycl::fmin(-sycl::dot(uv, n), 1.0f);
  vec r_out_perp = etai_over_etat * (uv + cos_theta * n);
  vec r_out_parallel =
      -sycl::sqrt(sycl::fabs(1.0f - length_squared(r_out_perp))) * n;
  return r_out_perp + r_out_parallel;
}

/* Computes normalised values of theta and phi. The input vector p
corresponds to a vector passing through the centre of the a sphere
and the hipoint on the surface of the sphere */
std::pair<real_t, real_t> mercator_coordinates(const vec& p) {
  // phi is the angle around the axis
  auto phi = sycl::atan2(p.z(), p.x());
  // theta is the angle down from the pole
  auto theta = sycl::asin(p.y());
  // theta and phi together constitute the spherical coordinates
  // phi is between -pi and pi , u is between 0 and 1
  auto u = 1 - (phi + pi) / (2 * pi);
  // theta is between -pi/2 and pi/2 , v is between 0 and 1
  auto v = (theta + pi / 2) / pi;
  return { u, v };
}

} // namespace raytracer

#endif