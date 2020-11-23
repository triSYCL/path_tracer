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
    triangle(point _v0, point _v1, point _v2, material_t mat_type)
    : v0 { _v0 }
    , v1 { _v1 }
    , v2 { _v2 }
    , material_type { mat_type }
    {}

    bool hit(const ray& r, real_t min, real_t max, hit_record& rec, material_t& hit_material_type) const
    {
        hit_material_type = material_type;

        // Get triangle edge vectors and plae normal
        auto u = v1-v0;
        auto v = v2-v0;
        vec outward_normal = sycl::cross(u, v);
        rec.set_face_normal(r,outward_normal);

        auto w0 = r.origin() - v0;
        auto a = -sycl::dot(outward_normal,w0);
        auto b = sycl::dot(outward_normal,r.direction());
        if (sycl::fabs(b) < 0.000001) return false; // ray is parallel to triangle plane

        // intersection point of ray with traingle
        real_t length = a/b;
        if(length<0) return false;
        else if (length < min || length > max)
            return false;

        vec hit_pt = r.at(rec.t);
        auto uu = sycl::dot(u, u);
        auto uv = sycl::dot(u, v);
        auto vv = sycl::dot(v, v);
        auto  w = hit_pt - v0;
        auto wu = sycl::dot(w, u);
        auto wv = sycl::dot(w, v);
        auto D = uv * uv - uu * vv;

        auto s = (uv * wv - vv * wu) / D;
        auto t = (uv * wu - uu * wv) / D;
        if(s < 0.0 || s > 1.0 || t < 0.0 || (s+t) > 1.0)
            return false;

        rec.set_face_normal(r,outward_normal);
        rec.t = length;
        rec.p = hit_pt;
        return true;

    }

    point v0,v1,v2;
    material_t material_type;
};
#endif
