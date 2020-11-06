#ifndef RECT_HPP
#define RECT_HPP

#include "ray.hpp"
#include "rtweekend.hpp"
#include "texture.hpp"
#include "material.hpp"
#include "vec.hpp"

class xy_rect{
public:
    xy_rect() = default;

    xy_rect(double _x0, double _x1, double _y0, double _y1, double _k,material_t mat_type)
    : x0 { _x0 }
    , x1 { _x1 }
    , y0 { _y0 }
    , y1 { _y1 }
    , k { _k }
    , material_type { mat_type }
    {}

    bool hit(const ray& r, real_t min, real_t max, hit_record& rec, material_t& hit_material_type) const
    {
        hit_material_type = material_type;

        auto t = (k-r.origin().z()) / r.direction().z();
        if (t < min || t > max)
            return false;
        auto x = r.origin().x() + t*r.direction().x();
        auto y = r.origin().y() + t*r.direction().y();
        if(x < x0 || x > x1 || y < y0 || y > y1)
            return false;
        rec.u = (x - x0) / (x1 - x0);
        rec.v = (y - y0) / (y1 - y0);
        rec.t = t;
        rec.p = r.at(rec.t);
        vec outward_normal = vec(0,0,1);
        rec.set_face_normal(r,outward_normal);
        return true;
    }
    real_t x0, x1, y0, y1, k;
    material_t material_type;
};

#endif
