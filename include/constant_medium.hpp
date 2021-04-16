#ifndef CONSTANT_MEDIUM_HPP
#define CONSTANT_MEDIUM_HPP

#include "box.hpp"
#include "material.hpp"
#include "sphere.hpp"
#include "texture.hpp"

namespace raytracer::scene {

using hittableVolume_t = std::variant<std::monostate, sphere, box>;

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

  hittableVolume_t boundary;
  real_t neg_inv_density;
  material_t phase_function;
};
} // namespace raytracer::scene
#endif
