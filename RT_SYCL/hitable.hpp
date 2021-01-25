#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.hpp"
#include "rtweekend.hpp"
#include "vec.hpp"

class hit_record {
public:
    float t; //
    point p; // hit point
    vec normal; // normal at hit point
    bool front_face; // to check if hit point is on the outer surface
    /*local coordinates for rectangles 
    and mercator coordintes for spheres */
    float u;
    float v;

    // To set if the hit point is on the front face
    void set_face_normal(const ray& r, const vec& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : vec{}-outward_normal;
    }
};

#endif
