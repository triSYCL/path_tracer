#ifndef BOX_HPP
#define BOX_HPP

#include <array>

#include "primitives.hpp"
#include "rectangle.hpp"

namespace raytracer::scene {
/// This class implements a axis aligned cuboid using 6 rectangles
class box {
 public:
  box() = default;

  /// p0 = { x0, y0, z0 } and p1 = { x1, y1. z1 }
  /// where x0 <= x1, y0 <= y1 and z0 <= z1
  box(const point& p0, const point& p1, const material_t& mat_type)
      : box_min { p0 }
      , box_max { p1 }
      , material_type { mat_type } {
    /// Add six sides of the box based on box_min and box_max to sides
    sides[0] = xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), material_type);
    sides[1] = xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), material_type);
    sides[2] = xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), material_type);
    sides[3] = xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), material_type);
    sides[4] = yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), material_type);
    sides[5] = yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), material_type);
  }

  point box_min;
  point box_max;
  material_t material_type;
  std::array<rectangle_t, 6> sides;
};
} // namespace raytracer::scene

#endif
