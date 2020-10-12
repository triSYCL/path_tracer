#ifndef RT_SYCL_RAY_HPP
#define RT_SYCL_RAY_HPP

#include "vec3.hpp"

class ray {
public:
    ray() = default;

    ray(const point3& origin, const vec3& direction)
        : orig { origin }
        , dir { direction }
    {
    }

    point3 origin() const { return orig; }
    vec3 direction() const { return dir; }

    //returns point along the ray at distance t from ray's origin
    point3 at(double t) const
    {
        return orig + t * dir;
    }

public:
    //To store the origin and direction of the ray
    point3 orig;
    vec3 dir;
};

#endif
