#ifndef CONSTANT_MEDIUM_HPP
#define CONSTANT_MEDIUM_HPP

#include "box.hpp"
#include "material.hpp"
#include "sphere.hpp"
#include "texture.hpp"
#include "visit.hpp"

using hittableVolume_t = std::variant<sphere, box>;

/**
 * A ray going through the volume can either make it all the way through
 * or be scattered at some point on or inside the volume.
 */
class constant_medium {
 public:
  constant_medium(const hittableVolume_t& b, real_t d, texture_t& a)
      : boundary { b }
      , neg_inv_density { -1 / d }
      , phase_function { isotropic_material { a } } {}

  constant_medium(const hittableVolume_t& b, real_t d, const color& a)
      : boundary { b }
      , neg_inv_density { -1 / d }
      , phase_function { isotropic_material { a } } {}

  bool hit(const ray& r, real_t min, real_t max, hit_record& rec,
           material_t& hit_material_type) const {
    hit_material_type = phase_function;
    material_t temp_material_type;
    hit_record rec1, rec2;
    if (!dev_visit(
            [&](auto&& arg) {
              return arg.hit(r, -infinity, infinity, rec1, temp_material_type);
            },
            boundary)) {
      return false;
    }
    if (!dev_visit(
            [&](auto&& arg) {
              return arg.hit(r, rec1.t + 0.0001f, infinity, rec2,
                             temp_material_type);
            },
            boundary)) {
      return false;
    }

    if (rec1.t < min)
      rec1.t = min;
    if (rec2.t > max)
      rec2.t = max;
    if (rec1.t >= rec2.t)
      return false;
    if (rec1.t < 0)
      rec1.t = 0;

    const auto ray_length = sycl::length(r.direction());
    /// Distance between the two hitpoints affect of probability
    /// of the ray hitting a smoke particle
    const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    auto rng = LocalPseudoRNG { toseed(r.direction()) };
    const auto hit_distance = neg_inv_density * sycl::log(rng.real());

    /// With lower density, hit_distance has higher probabilty
    /// of being greater than distance_inside_boundary
    if (hit_distance > distance_inside_boundary)
      return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    rec.normal = vec { 1, 0, 0 }; // arbitrary
    rec.front_face = true;        // also arbitrary
    return true;
  }

  hittableVolume_t boundary;
  real_t neg_inv_density;
  material_t phase_function;
};
#endif
