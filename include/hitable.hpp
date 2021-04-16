#ifndef HITTABLE_H
#define HITTABLE_H

#include <variant>

#include "hit_record.hpp"
#include "material.hpp"

#include "box.hpp"
#include "constant_medium.hpp"
#include "ray.hpp"
#include "rectangle.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

namespace raytracer::scene {
using hittable_t = std::variant<std::monostate, sphere, xy_rect, triangle, box,
                                constant_medium>;
}
#endif
