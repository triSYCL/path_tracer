#ifndef RT_SYCL_VEC_HPP
#define RT_SYCL_VEC_HPP

#include <cmath>
#include <iostream>

#include <SYCL/sycl.hpp>
using real_t = double;
/*class vec3 {
public:
    vec3()
        : e { 0, 0, 0 }
    {
    } //empty constructor
    vec3(double e0, double e1, double e2)
        : e { e0, e1, e2 }
    {
    }

    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    double operator[](int i) const { return e[i]; }
    double& operator[](int i) { return e[i]; }

    vec3& operator+=(const vec3& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vec3& operator*=(const double t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3& operator/=(const double t)
    {
        return *this *= 1 / t;
    }

    double length() const
    {
        return sycl::sqrt(length_squared());
    }

    double length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

public:
    double e[3];
};*/

//type aliases for vec3 -  point and color
using point3 = sycl::double3;
using color = sycl::double3;
using vec3 = sycl::double3;

// vec3 Utility Functions

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

/*inline vec3 operator+(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator*(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator*(double t, const vec3& v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline vec3 operator*(const vec3& v, double t)
{
    return t * v;
}

inline vec3 operator/(vec3 v, double t)
{
    return (1 / t) * v;
}

inline double dot(const vec3& u, const vec3& v)
{
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3& u, const vec3& v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}*/

inline vec3 unit_vector(vec3 v)
{
    return v / sycl::length(v);
}

vec3 random_unit_vector()
{
    auto a = random_double(0, 2 * pi);
    auto z = random_double(-1, 1);
    auto r = sycl::sqrt(1 - z * z);
    return vec3(r * cos(a), r * sin(a), z);
}

vec3 random_in_unit_sphere()
{
    while (true) {
        auto p = randomvec3(-1, 1);
        if (sycl::length(p)*sycl::length(p) >= 1)
            continue;
        return p;
    }
}

vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2 * sycl::dot(v, n) * n;
}

vec3 random_in_unit_disk()
{
    while (true) {
        auto p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
        if (sycl::length(p)*sycl::length(p) >= 1)
            continue;
        return p;
    }
}

vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat)
{
    auto cos_theta = fmin(sycl::dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sycl::sqrt(fabs(1.0 - sycl::length(r_out_perp)*sycl::length(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

#endif
