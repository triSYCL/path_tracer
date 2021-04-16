#ifndef RT_SYCL_MATERIAL_HPP
#define RT_SYCL_MATERIAL_HPP

#include <variant>

#include "primitives.hpp"
#include "texture.hpp"

namespace raytracer::scene {
struct lambertian_material {
  lambertian_material() = default;
  lambertian_material(const color& a)
      : albedo { solid_texture { a } } {}
  lambertian_material(const texture_t& a)
      : albedo { a } {}
  texture_t albedo;
};

struct metal_material {
  metal_material() = default;
  metal_material(const color& a, real_t f)
      : albedo { a }
      , fuzz { std::clamp(f, 0.0f, 1.0f) } {}

  color albedo;
  real_t fuzz;
};

struct dielectric_material {
  dielectric_material() = default;
  dielectric_material(real_t ri, const color& albedo)
      : ref_idx { ri }
      , albedo { albedo } {}

  // Schlick's approximation for reflectance
  real_t reflectance(real_t cosine, real_t ref_idx) const {
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 *= r0;
    return r0 + (1 - r0) * sycl::pow((1 - cosine), 5.0f);
  }

  // Refractive index of the glass
  real_t ref_idx;
  // Color of the glass
  color albedo;
};

struct lightsource_material {
  lightsource_material() = default;
  lightsource_material(const texture_t& a)
      : emit { a } {}
  lightsource_material(const color& a)
      : emit { solid_texture { a } } {}

  texture_t emit;
};

struct isotropic_material {
  isotropic_material(const color& a)
      : albedo { solid_texture { a } } {}
  isotropic_material(texture_t& a)
      : albedo { a } {}
  texture_t albedo;
};

using material_t =
    std::variant<std::monostate, lambertian_material, metal_material,
                 dielectric_material, lightsource_material, isotropic_material>;
} // namespace raytracer::scene

#endif
