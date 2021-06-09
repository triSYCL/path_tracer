#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.hpp"
#include "sphere.hpp"
#include "triangle.hpp"
#include "box.hpp"
#include "constant_medium.hpp"

using hittable_t = std::variant<sphere, triangle, constant_medium, box>;
#endif
