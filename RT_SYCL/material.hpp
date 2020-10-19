#ifndef RT_SYCL_MATERIAL_HPP
#define RT_SYCL_MATERIAL_HPP
#include "hittable.hpp"
#include "vec3.hpp"
#include <iostream>
#include <variant>
#include <vector>

struct lambertian_material {
    lambertian() = default;
    lambertian(const color& a)
        : albedo { a }
    {
    }

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, rat& scattered) const
    {
        vec3 scatter_direction = rec.normal + random_unit_vector();
        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
    color albedo;
};

struct metal_material {
    metal() = default;
    metal(const color& a, double f)
        : albedo { a }
        , fuzz { std::clamp(f, 0.0, 1.0) }
    {
    }

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, rat& scattered) const
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    color albedo;
    double fuzz;
};

struct dielectric_material{
    dielectric_material() = default;
    dielectric(double ri) : ref_idx {ri}{}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, rat& scattered) const
    {
        attenuation = color{0.1,0.1,0.1};
        double etai_over_etat = ref_idx;
        vec3 unit_direction = unit_vector(r_in.direction());
        vec3 refracted = refract(unit_direction, rec.normal, etai_over_etat);
        scattered = ray(rec.p, refracted);
        return true;
    }
    double ref_idx;
};

//using material_t = std::variant<lambertian_material, metal_material, dielectric_material>;

#endif
