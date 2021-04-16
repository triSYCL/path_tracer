#ifndef RECT_HPP
#define RECT_HPP

#include "material.hpp"
#include "primitives.hpp"

/** The Following classes implement:

        -
   https://raytracing.github.io/books/RayTracingTheNextWeek.html#rectanglesandlights/creatingrectangleobjectsa
*/
namespace raytracer::scene {

template <bool on_x_plane, bool on_y_plane, bool on_z_plane> class rect {
  static_assert((on_x_plane && !on_y_plane && !on_z_plane) ||
                    (!on_x_plane && on_y_plane && !on_z_plane) ||
                    (!on_x_plane && !on_y_plane && on_z_plane),
                "Only one parameter should be set to true");

 public:
  real_t a0, a1, b0, b1, k;
  material_t material_type;
  rect(real_t _a0, real_t _a1, real_t _b0, real_t _b1, real_t _k,
       const material_t& _mat_type)
      : a0 { _a0 }
      , a1 { _a1 }
      , b0 { _b0 }
      , b1 { _b1 }
      , k { _k }
      , material_type { _mat_type } {}
    
  rect() = default;

  static inline real_t first_planar_coord(vec const& values) {
    if constexpr (on_x_plane) {
      return values.y();
    } else if constexpr (on_y_plane) {
      return values.z();
    } else {
      return values.x();
    }
  }

  static inline real_t second_planar_coord(vec const& values) {
    if constexpr (on_x_plane) {
      return values.z();
    } else if constexpr (on_y_plane) {
      return values.x();
    } else {
      return values.y();
    }
  }

  static inline real_t normal_coord(vec const& values) {
    if constexpr (on_x_plane) {
      return values.x();
    } else if constexpr (on_y_plane) {
      return values.y();
    } else {
      return values.z();
    }
  }

  static inline vec plan_basis(vec const& values) {
    return { first_planar_coord(values), second_planar_coord(values),
             normal_coord(values) };
  }
};

using xy_rect = rect<false, false, true>;
using xz_rect = rect<false, true, false>;
using yz_rect = rect<true, false, false>;

using rectangle_t = std::variant<xy_rect, xz_rect, yz_rect>;
} // namespace raytracer::scene

#endif
