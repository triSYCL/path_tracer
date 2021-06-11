#ifndef HITTABLE_H
#define HITTABLE_H

#include "box.hpp"
#include "constant_medium.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

using hittable_t = std::variant<sphere, triangle, constant_medium, box>;
#endif
