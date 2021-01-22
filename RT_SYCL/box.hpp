#ifndef BOX_HPP
#define BOX_HPP

#include "rectangle.hpp"
#include "rtweekend.hpp"
#include "visit.hpp"

/// This class implements a axis aligned cuboid using 6 rectangles
class box {
public:
    box() = default;

    /// p0 = { x0, y0, z0 } and p1 = { x1, y1. z1 }
    /// where x0 <= x1, y0 <= y1 and z0 <= z1
    box(const point& p0, const point& p1, const material_t& mat_type)
        : box_min { p0 }
        , box_max { p1 }
        , material_type { mat_type }
    {
        /// Add six sides of the box based on box_min and box_max to sides
        sides[0] = xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), mat_type);
        sides[1] = xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), mat_type);
        sides[2] = xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), mat_type);
        sides[3] = xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), mat_type);
        sides[4] = yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), mat_type);
        sides[5] = yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), mat_type);
    }

    /// Compute ray interaction with the box
    bool hit(const ray& r, real_t min, real_t max, hit_record& rec, material_t& hit_material_type) const
    {
        hit_record temp_rec;
        material_t temp_material_type;
        auto hit_anything = false;
        auto closest_so_far = max;
        // Checking if the ray hits any of the sides
        for (const auto& side : sides) {
            if (dev_visit([&](auto&& arg) { return arg.hit(r, min, closest_so_far, temp_rec, temp_material_type); }, side)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                hit_material_type = temp_material_type;
            }
        }
        return hit_anything;
    }

    point box_min;
    point box_max;
    material_t material_type;
    std::array<rectangle_t, 6> sides;
};

#endif
