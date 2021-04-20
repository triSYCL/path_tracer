#ifndef RT_SYCL_MATERIAL_HPP
#define RT_SYCL_MATERIAL_HPP

#include <iostream>

#include "hit_record.hpp"
#include "texture.hpp"
#include "vec.hpp"
#include "visit.hpp"

struct lambertian_material {
  lambertian_material() = default;
  lambertian_material(const color& a)
      : albedo { solid_texture { a } } {}
  lambertian_material(const texture_t& a)
      : albedo { a } {}

  bool scatter(auto& ctx, const ray& r_in, const hit_record& rec,
               color& attenuation, ray& scattered) const {
    auto& rng = ctx.rng;
    vec scatter_direction = rec.normal + rng.unit_vec();
    scattered = ray(rec.p, scatter_direction, r_in.time());
    // Attenuation of the ray hitting the object is modified based on the color
    // at hit point
    attenuation *= dev_visit(
        monostate_dispatch([&](auto&& t) { return t.value(ctx, rec); },
                            color { 0.f, 0.f, 0.f }),
        albedo);
    return true;
  }
  color emitted(auto&, const hit_record& rec) { return color(0.f, 0.f, 0.f); }
  texture_t albedo;
};


struct lightsource_material {
  lightsource_material() = default;
  lightsource_material(const texture_t& a)
      : emit { a } {}
  lightsource_material(const color& a)
      : emit { solid_texture { a } } {}

  template <typename... T> bool scatter(T&...) const { return false; }

  color emitted(auto& ctx, const hit_record& rec) {
    return dev_visit(
        monostate_dispatch([&](auto&& t) { return t.value(ctx, rec); },
                           color { 0.f, 0.f, 0.f }),
        emit);
  }

  texture_t emit;
};

struct isotropic_material {
  isotropic_material(const color& a)
      : albedo { solid_texture { a } } {}
  isotropic_material(texture_t& a)
      : albedo { a } {}

  bool scatter(auto& ctx, const ray& r_in, const hit_record& rec,
               color& attenuation, ray& scattered) const {
    auto& rng = ctx.rng;
    scattered = ray(rec.p, rng.in_unit_ball(), r_in.time());
    attenuation *=
        dev_visit(monostate_dispatch([&](auto&& t) { return t.value(ctx, rec); },
                                     color { 0.f, 0.f, 0.f }),
                  albedo);
    return true;
  }

  color emitted(auto&, const hit_record& rec) { return color(0, 0, 0); }

  texture_t albedo;
};

using material_t =
    std::variant<std::monostate, lambertian_material, lightsource_material, isotropic_material>;
#endif
