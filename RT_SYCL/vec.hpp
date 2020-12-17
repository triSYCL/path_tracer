#ifndef RT_SYCL_VEC_HPP
#define RT_SYCL_VEC_HPP

#include "rtweekend.hpp"
#include <cmath>
#include <iostream>

#include <SYCL/sycl.hpp>
using real_t = double;

//type aliases for double3 - vec, point and color
using point = sycl::double3;
using color = sycl::double3;
using vec = sycl::double3;

// vec Utility Functions
double length_squared(const vec& v)
{
    return sycl::fma(v.x(), v.x(), sycl::fma(v.y(), v.y(), fma(v.z(), v.z(), 0)));
}

vec randomvec()
{
    return vec(random_double(), random_double(), random_double());
}

vec randomvec(double min, double max)
{
    return vec(random_double(min, max), random_double(min, max), random_double(min, max));
}

inline std::ostream& operator<<(std::ostream& out, const vec& v)
{
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

// Missing operator from the SYCL specification for now
vec operator-(const vec& u) { return vec(-u.x(), -u.y(), -u.z()); }

// Compute a unit vector from a non-null vector
inline vec unit_vector(vec v)
{
    return v / sycl::length(v);
}

// Make a random unit vector
vec random_unit_vector()
{
    auto a = random_double(0, 2 * pi);
    auto z = random_double(-1, 1);
    auto r = sycl::sqrt(1 - z * z);
    return vec(r * sycl::cos(a), r * sycl::sin(a), z);
}

// Compute a random point inside a unit sphere at origin
vec random_in_unit_sphere()
{
    while (true) {
        auto p = randomvec(-1, 1);
        if (length_squared(p) >= 1)
            continue;
        return p;
    }
}

// Compute reflected ray's direction
vec reflect(const vec& v, const vec& n)
{
    return v - 2 * sycl::dot(v, n) * n;
}

// Compute random point in a unit disk
vec random_in_unit_disk()
{
    while (true) {
        auto p = vec(random_double(-1, 1), random_double(-1, 1), 0);
        if (length_squared(p) >= 1)
            continue;
        return p;
    }
}

// Computes refracted ray's direction based on refractive index
vec refract(const vec& uv, const vec& n, double etai_over_etat)
{
    auto cos_theta = sycl::fmin(sycl::dot(-uv, n), 1.0);
    vec r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec r_out_parallel = -sycl::sqrt(sycl::fabs(1.0 - length_squared(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

#endif
