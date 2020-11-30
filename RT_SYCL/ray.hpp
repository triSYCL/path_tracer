#ifndef RT_SYCL_RAY_HPP
#define RT_SYCL_RAY_HPP

#include "vec.hpp"

class ray {
public:
    ray() = default;

    ray(const point& origin, const vec& direction)
        : orig { origin }
        , dir { direction }
    {
    }

    point origin() const { return orig; }
    vec direction() const { return dir; }

    //returns point along the ray at distance t from ray's origin
    //the ray P(t) = Origin + t*direction
    point at(double t) const
    {
        return orig + t * dir;
    }

public:
    //To store the origin and direction of the ray
    point orig;
    vec dir;
};

#endif
