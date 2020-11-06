#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.hpp"
#include "rtweekend.hpp"
#include "texture.hpp"
#include "vec.hpp"

class hit_record {
public:
    double t;
    point p;
    vec normal;
    bool front_face;
    double u;
    double v;

    // To set if the hit point is on the front face  
    void set_face_normal(const ray& r, const vec& outward_normal){
        front_face = dot(r.direction(),outward_normal)<0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

#endif
