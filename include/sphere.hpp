#ifndef SPHERE_H
#define SPHERE_H

#include "material.hpp"
#include "primitives.hpp"
#include "texture.hpp"

namespace raytracer::scene {
class sphere {
 public:
  sphere() = default;

  sphere(const point& cen, real_t r, const material_t& mat_type)
      : center0 { cen }
      , center1 { cen }
      , radius { r }
      , time0 { 0 }
      , time1 { 0 }
      , material_type { mat_type } {}

  /// Simulates moving spheres from center0 to
  /// center1 between time0 and time1
  sphere(const point& cen0, const point& cen1, real_t _time0, real_t _time1,
         real_t r, const material_t& mat_type)
      : center0 { cen0 }
      , center1 { cen1 }
      , radius { r }
      , time0 { _time0 }
      , time1 { _time1 }
      , material_type { mat_type } {}

  /// Computes center of the sphere based on
  /// the time information stored in the ray
  point center(real_t time) const {
    if (time0 == time1)
      return center0;
    else
      return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
  }

  // Geometry properties
  point center0, center1;
  real_t radius;

  // Time of start and end of motion of the sphere
  real_t time0, time1;

  // Material properties
  material_t material_type;
};
} // namespace raytracer::scene

#endif
