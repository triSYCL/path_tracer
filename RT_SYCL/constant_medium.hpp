#ifndef CONSTANT_MEDIUM_HPP
#define CONSTANT_MEDIUM_HPP

#include "sphere.hpp"
#include "box.hpp"
#include "material.hpp"
#include "texture.hpp"

using hittableVolume_t = std::variant<sphere, box>;

class constant_medium{
public:
    constant_medium(const hittableVolume_t& b, real_t d, texture_t& a)
        : boundary { b }
        , neg_inv_density { -1/d }
        , phase_function { isotropic_material { a } }
    {
    }

    constant_medium(const hittableVolume_t& b, real_t d, const color& a)
        : boundary { b }
        , neg_inv_density { -1/d }
        , phase_function { isotropic_material { a } }
    {
    }

    bool hit(const ray& r, real_t min, real_t max, hit_record& rec, material_t& hit_material_type) const
    {
        hit_material_type = phase_function;
        material_t temp_material_type;
        hit_record rec1, rec2;
        if (!std::visit([&](auto&& arg) { return arg.hit(r, -infinity, infinity, rec1, temp_material_type); }, boundary)) {
            return false;
        }

        if (!std::visit([&](auto&& arg) { return arg.hit(r, rec1.t+0.0001, infinity, rec2, temp_material_type); }, boundary)) {
            return false;
        }

        if (rec1.t < min) rec1.t = min;
        if (rec2.t > max) rec2.t = max;
        if (rec1.t >= rec2.t)
            return false;
        if (rec1.t < 0)
            rec1.t = 0;
        
        const auto ray_length = sycl::length(r.direction());
        const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
        const auto hit_distance = neg_inv_density * log(random_double());

        if (hit_distance > distance_inside_boundary)
        return false;

        rec.t = rec1.t + hit_distance / ray_length;
        rec.p = r.at(rec.t);

        rec.normal = vec { 1, 0, 0 };  // arbitrary
        rec.front_face = true;     // also arbitrary
        return true;
    }

    hittableVolume_t boundary;
    real_t neg_inv_density;
    material_t phase_function;
};
#endif
