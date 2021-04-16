#ifndef MATERIAL_VISITOR_HPP
#define MATERIAL_VISITOR_HPP

#include "material.hpp"
#include "primitives.hpp"
#include "texture_visitor.hpp"
#include "visit.hpp"

namespace raytracer::visitor {
struct material_emitted {
 private:
  task_context& ctx;
  const hit_record& rec;

 public:
  material_emitted(task_context& ctx, hit_record& rec)
      : ctx { ctx }
      , rec { rec } {}

  template <typename M> color operator()(M&&) {
    return { 0.f, 0.f, 0.f };
  }

  color operator()(scene::lightsource_material& mat) {
      return dev_visit(texture_value(ctx, rec), mat.emit);
  }

  color operator()(std::monostate) {
    assert(false && "unreachable");
    return { 0.f, 0.f, 0.f };
  }
};

struct material_scatter {
 private:
  task_context& ctx;
  const ray& r_in;
  const hit_record& rec;
  color& attenuation;
  ray& scattered;

 public:
  material_scatter(auto& ctx, const ray& r_in, const hit_record& rec,
                           color& attenuation, ray& scattered)
      : ctx { ctx }
      , r_in { r_in }
      , rec { rec }
      , attenuation { attenuation }
      , scattered { scattered } {}

  bool operator()(scene::lambertian_material& mat) const {
    vec scatter_direction = rec.normal + ctx.rng.unit_vec();
    scattered = ray(rec.p, scatter_direction, r_in.time());
    // Attenuation of the ray hitting the object is modified based on the color
    // at hit point
    attenuation *= dev_visit(texture_value(ctx, rec), mat.albedo);
    return true;
  }

  bool operator()(scene::metal_material& mat) {
    vec reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered =
        ray(rec.p, reflected + mat.fuzz * ctx.rng.in_unit_ball(), r_in.time());
    // Attenuation of the ray hitting the object is modified based on the color
    // at hit point
    attenuation *= mat.albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }

  bool operator()(scene::dielectric_material& mat) {
    // Attenuation of the ray hitting the object is modified based on the color
    // at hit point
    attenuation *= mat.albedo;
    real_t refraction_ratio =
        rec.front_face ? (1.0f / mat.ref_idx) : mat.ref_idx;
    vec unit_direction = unit_vector(r_in.direction());
    real_t cos_theta = sycl::fmin(-sycl::dot(unit_direction, rec.normal), 1.0f);
    real_t sin_theta = sycl::sqrt(1.0f - cos_theta * cos_theta);
    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    vec direction;
    if (cannot_refract ||
        mat.reflectance(cos_theta, refraction_ratio) > ctx.rng.real())
      direction = reflect(unit_direction, rec.normal);
    else
      direction = refract(unit_direction, rec.normal, refraction_ratio);

    scattered = ray(rec.p, direction, r_in.time());
    return true;
  }

  bool operator()(scene::lightsource_material&) {return false;}

  bool operator()(scene::isotropic_material& mat) {
    scattered = ray(rec.p, ctx.rng.in_unit_ball(), r_in.time());
    attenuation *= dev_visit(texture_value(ctx, rec), mat.albedo);
    return true;
  }

  bool operator()(std::monostate) {
    assert(false && "unreachable");
    return false;
  }
};
} // namespace raytracer::visitor
#endif