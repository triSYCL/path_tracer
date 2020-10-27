#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.hpp"
#include "rtweekend.hpp"
#include "texture.hpp"
#include "vec3.hpp"

enum class material_t { Lambertian,
    Metal,
    Dielectric };

class hit_record {
public:
    
    vec3 center;
    real_t radius;

    double t;
    point3 p;
    vec3 normal;
    double u;
    double v;

    // material properties
    //Material_t material_type;
};

#endif
