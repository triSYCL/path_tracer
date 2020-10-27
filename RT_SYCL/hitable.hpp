#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.hpp"
#include "rtweekend.hpp"
#include "texture.hpp"
#include "vec3.hpp"

class hit_record {
public:
    double t;
    point3 p;
    vec3 normal;
    double u;
    double v;
};

#endif
