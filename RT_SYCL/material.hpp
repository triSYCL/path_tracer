#ifndef RT_SYCL_MATERIAL_HPP
#define RT_SYCL_MATERIAL_HPP

#include <iostream>

#include "hitable.hpp"
#include "texture.hpp"
#include "vec.hpp"
#include "visit.hpp"

struct lambertian_material {
  lambertian_material() = default;
  lambertian_material(const color& a)
      : albedo { solid_texture { a } } {}
  lambertian_material(const texture_t& a)
      : albedo { a } {}

  bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
               ray& scattered) const {
    vec scatter_direction = rec.normal + random_unit_vector();
    scattered = ray(rec.p, scatter_direction, r_in.time());
    // Attenuation of the ray hitting the object is modified based on the color
    // at hit point
    attenuation *=
        dev_visit([&](auto&& arg) { return arg.value(rec); }, albedo);
    return true;
  }
  color emitted(const hit_record& rec) { return color(0, 0, 0); }
  texture_t albedo;
};

struct metal_material {
  metal_material() = default;
  metal_material(const color& a, float f)
      : albedo { a }
      , fuzz { std::clamp(f, 0.0f, 1.0f) } {}

  bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
               ray& scattered) const {
    vec reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered =
        ray(rec.p, reflected + fuzz * random_in_unit_sphere(), r_in.time());
    // Attenuation of the ray hitting the object is modified based on the color
    // at hit point
    attenuation *= albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }

  color emitted(const hit_record& rec) { return color(0, 0, 0); }
  color albedo;
  float fuzz;
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

  bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
               ray& scattered) const {
    // Attenuation of the ray hitting the object is modified based on the color
    // at hit point
    attenuation *= albedo;
    float refraction_ratio = rec.front_face ? (1.0f / ref_idx) : ref_idx;
    vec unit_direction = unit_vector(r_in.direction());
    float cos_theta = sycl::fmin(-sycl::dot(unit_direction, rec.normal), 1.0f);
    float sin_theta = sycl::sqrt(1.0f - cos_theta * cos_theta);
    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    vec direction;
    if (cannot_refract ||
        reflectance(cos_theta, refraction_ratio) > random_float())
      direction = reflect(unit_direction, rec.normal);
    else
      direction = refract(unit_direction, rec.normal, refraction_ratio);

    scattered = ray(rec.p, direction, r_in.time());
    return true;
  }

  color emitted(const hit_record& rec) { return color(0, 0, 0); }
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

  bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
               ray& scattered) const {
    return false;
  }

  color emitted(const hit_record& rec) {
    return dev_visit([&](auto&& arg) { return arg.value(rec); }, emit);
  }

  texture_t emit;
};

struct isotropic_material {
  isotropic_material(const color& a)
      : albedo { solid_texture { a } } {}
  isotropic_material(texture_t& a)
      : albedo { a } {}

  bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
               ray& scattered) const {
    scattered = ray(rec.p, random_in_unit_sphere(), r_in.time());
    attenuation *=
        dev_visit([&](auto&& arg) { return arg.value(rec); }, albedo);
    return true;
  }

  color emitted(const hit_record& rec) { return color(0, 0, 0); }

  texture_t albedo;
};

using material_t =
    std::variant<lambertian_material, metal_material, dielectric_material,
                 lightsource_material, isotropic_material>;

#endif
