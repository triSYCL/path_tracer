#ifndef HITTABLE_H
#define HITTABLE_H

#include "box.hpp"
#include "constant_medium.hpp"
#include "ray.hpp"
#include "rectangle.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

using hittable_t = std::variant<std::monostate, sphere, xy_rect, triangle, box,
                                constant_medium>;
#endif
