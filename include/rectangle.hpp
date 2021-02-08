#ifndef RECT_HPP
#define RECT_HPP

#include "material.hpp"
#include "ray.hpp"
#include "rtweekend.hpp"
#include "texture.hpp"
#include "vec.hpp"

/** The Following classes implement:

        -
   https://raytracing.github.io/books/RayTracingTheNextWeek.html#rectanglesandlights/creatingrectangleobjectsa
*/

class xy_rect {
 public:
  xy_rect() = default;

  /// x0 <= x1 and y0 <= y1
  xy_rect(real_t _x0, real_t _x1, real_t _y0, real_t _y1, real_t _k,
          const material_t& mat_type)
      : x0 { _x0 }
      , x1 { _x1 }
      , y0 { _y0 }
      , y1 { _y1 }
      , k { _k }
      , material_type { mat_type } {}

  /// Compute ray interaction with rectangle
  bool hit(const ray& r, real_t min, real_t max, hit_record& rec,
           material_t& hit_material_type) const {
    hit_material_type = material_type;

    auto t = (k - r.origin().z()) / r.direction().z();
    if (t < min || t > max)
      return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
      return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    rec.p = r.at(rec.t);
    vec outward_normal = vec(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    return true;
  }
  real_t x0, x1, y0, y1, k;
  material_t material_type;
};

class xz_rect {
 public:
  xz_rect() = default;

  /// x0 <= x1 and z0 <= z1
  xz_rect(real_t _x0, real_t _x1, real_t _z0, real_t _z1, real_t _k,
          const material_t& mat_type)
      : x0 { _x0 }
      , x1 { _x1 }
      , z0 { _z0 }
      , z1 { _z1 }
      , k { _k }
      , material_type { mat_type } {}

  /// Compute ray interaction with rectangle
  bool hit(const ray& r, real_t min, real_t max, hit_record& rec,
           material_t& hit_material_type) const {
    hit_material_type = material_type;

    auto t = (k - r.origin().y()) / r.direction().y();
    if (t < min || t > max)
      return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)
      return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    rec.p = r.at(rec.t);
    vec outward_normal = vec(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    return true;
  }
  real_t x0, x1, z0, z1, k;
  material_t material_type;
};

class yz_rect {
 public:
  yz_rect() = default;

  /// y0 <= y1 and z0 <= z1
  yz_rect(real_t _y0, real_t _y1, real_t _z0, real_t _z1, real_t _k,
          const material_t& mat_type)
      : y0 { _y0 }
      , y1 { _y1 }
      , z0 { _z0 }
      , z1 { _z1 }
      , k { _k }
      , material_type { mat_type } {}

  /// Compute ray interaction with rectangle
  bool hit(const ray& r, real_t min, real_t max, hit_record& rec,
           material_t& hit_material_type) const {
    hit_material_type = material_type;

    auto t = (k - r.origin().x()) / r.direction().x();
    if (t < min || t > max)
      return false;
    auto y = r.origin().y() + t * r.direction().y();
    auto z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)
      return false;
    rec.u = (y - y0) / (y1 - y0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    rec.p = r.at(rec.t);
    vec outward_normal = vec(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    return true;
  }
  real_t y0, y1, z0, z1, k;
  material_t material_type;
};

using rectangle_t = std::variant<xy_rect, xz_rect, yz_rect>;

#endif
