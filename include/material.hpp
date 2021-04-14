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
    attenuation *= dev_visit(texture_value_visitor(ctx, rec), albedo);
    return true;
  }
  color emitted(auto&, const hit_record& rec) { return color(0, 0, 0); }
  texture_t albedo;
};

struct metal_material {
  metal_material() = default;
  metal_material(const color& a, real_t f)
      : albedo { a }
      , fuzz { std::clamp(f, 0.0f, 1.0f) } {}

  bool scatter(auto& ctx, const ray& r_in, const hit_record& rec,
               color& attenuation, ray& scattered) const {
    auto& rng = ctx.rng;
    vec reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz * rng.in_unit_ball(), r_in.time());
    // Attenuation of the ray hitting the object is modified based on the color
    // at hit point
    attenuation *= albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }

  color emitted(auto&, const hit_record& rec) { return color(0, 0, 0); }
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

  bool scatter(auto& ctx, const ray& r_in, const hit_record& rec,
               color& attenuation, ray& scattered) const {
    // Attenuation of the ray hitting the object is modified based on the color
    // at hit point
    auto& rng = ctx.rng;
    attenuation *= albedo;
    real_t refraction_ratio = rec.front_face ? (1.0f / ref_idx) : ref_idx;
    vec unit_direction = unit_vector(r_in.direction());
    real_t cos_theta = sycl::fmin(-sycl::dot(unit_direction, rec.normal), 1.0f);
    real_t sin_theta = sycl::sqrt(1.0f - cos_theta * cos_theta);
    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    vec direction;
    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rng.real())
      direction = reflect(unit_direction, rec.normal);
    else
      direction = refract(unit_direction, rec.normal, refraction_ratio);

    scattered = ray(rec.p, direction, r_in.time());
    return true;
  }

  color emitted(auto&, const hit_record& rec) { return color(0, 0, 0); }
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

  template <typename... T> bool scatter(T&...) const { return false; }

  color emitted(auto& ctx, const hit_record& rec) {
    return dev_visit(texture_value_visitor(ctx, rec), emit);
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
    attenuation *= dev_visit(texture_value_visitor(ctx, rec), albedo);
    return true;
  }

  color emitted(auto&, const hit_record& rec) { return color(0, 0, 0); }

  texture_t albedo;
};

using material_t =
    std::variant<std::monostate, lambertian_material, metal_material,
                 dielectric_material, lightsource_material, isotropic_material>;

struct material_emitted_visitor {
 private:
  task_context& ctx;
  const hit_record& rec;

 public:
  material_emitted_visitor(task_context& ctx, hit_record& rec)
      : ctx { ctx }
      , rec { rec } {}

  template <typename M> color operator()(M&& material) {
    return material.emitted(ctx, rec);
  }

  color operator()(std::monostate) {
    assert(false && "unreachable");
    return { 0.f, 0.f, 0.f };
  }
};

struct material_scatter_visitor {
 private:
  task_context& ctx;
  const ray& r_in;
  const hit_record& rec;
  color& attenuation;
  ray& scattered;

 public:
  material_scatter_visitor(auto& ctx, const ray& r_in, const hit_record& rec,
                           color& attenuation, ray& scattered)
      : ctx { ctx }
      , r_in { r_in }
      , rec { rec }
      , attenuation { attenuation }
      , scattered { scattered } {}

  template <typename M> bool operator()(M&& material) {
    return material.scatter(ctx, r_in, rec, attenuation, scattered);
  }

  bool operator()(std::monostate) {
    assert(false && "unreachable");
    return false;
  }
};

#endif
