#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "ray.hpp"
#include "rtweekend.hpp"
#include "texture.hpp"
#include "material.hpp"
#include "vec.hpp"


class triangle{
public:
    triangle() = default;
    triangle(real_t _x0, real_t _y0, real_t _x1, real_t _y1, real_t _x2, real_t _y2, material_t mat_type)
    : x0 { _x0 }
    , x1 { _x1 }
    , x2 { _x2 }
    , y0 { _y0 }
    , y1 { _y1 }
    , y2 { _y2 }
    , material_type { mat_type }
    {}

    bool hit(const ray& r, real_t min, real_t max, hit_record& rec, material_t& hit_material_type) const
    {
        hit_material_type = material_type;


    }

    real_t x0,x1,x2,y0,y1,y2;
    material_t material_type;
};
#endif
