#ifndef RT_SYCL_VEC_HPP
#define RT_SYCL_VEC_HPP

#include <cmath>
#include <iostream>

#include <SYCL/sycl.hpp>
using real_t = double;

//type aliases for double3 - vec3, point and color
using point3 = sycl::double3;
using color = sycl::double3;
using vec3 = sycl::double3;

// vec3 Utility Functions
double length_squared(const vec3& v)
{
    return sycl::fma(v.x(), v.x(), sycl::fma(v.y(), v.y(), fma(v.z(), v.z(), 0)));
}

vec3 randomvec3()
{
    return vec3(random_double(), random_double(), random_double());
}

vec3 randomvec3(double min, double max)
{
    return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
}

inline std::ostream& operator<<(std::ostream& out, const vec3& v)
{
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

vec3 operator-(const vec3& u) { return vec3(-u.x(), -u.y(), -u.z()); }

// Compute a unit vector from a non-null vector
inline vec3 unit_vector(vec3 v)
{
    return v / sycl::length(v);
}

// Make a random unit vector
vec3 random_unit_vector()
{
    auto a = random_double(0, 2 * pi);
    auto z = random_double(-1, 1);
    auto r = sycl::sqrt(1 - z * z);
    return vec3(r * sycl::cos(a), r * sycl::sin(a), z);
}

//return a random point inside a unit sphere at origin
vec3 random_in_unit_sphere()
{
    while (true) {
        auto p = randomvec3(-1, 1);
        if (length_squared(p) >= 1)
            continue;
        return p;
    }
}

//returns reflected ray about the normal
vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2 * sycl::dot(v, n) * n;
}

//returns random point in a unit disk
vec3 random_in_unit_disk()
{
    while (true) {
        auto p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
        if (length_squared(p) >= 1)
            continue;
        return p;
    }
}

//returns refracted ray based on refractive index
vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat)
{
    auto cos_theta = sycl::fmin(sycl::dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sycl::sqrt(sycl::fabs(1.0 - length_squared(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

#endif
