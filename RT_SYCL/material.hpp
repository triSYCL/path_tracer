#ifndef RT_SYCL_MATERIAL_HPP
#define RT_SYCL_MATERIAL_HPP
#include "hitable.hpp"
#include "texture.hpp"
#include "vec.hpp"
#include <iostream>
#include <variant>
#include <vector>

struct lambertian_material {
    lambertian_material() = default;
    lambertian_material(const color& a)
        : albedo { solid_texture { a } }
    {
    }
    lambertian_material(texture_t& a)
        : albedo { a }
    {
    }

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const
    {
        vec scatter_direction = rec.normal + random_unit_vector();
        scattered = ray(rec.p, scatter_direction);
        attenuation *= std::visit([&](auto&& arg) { return arg.value(rec); }, albedo);
        return true;
    }
    color emitted(const hit_record& rec)
    {
        return color(0, 0, 0);
    }
    texture_t albedo;
};

struct metal_material {
    metal_material() = default;
    metal_material(const color& a, double f)
        : albedo { a }
        , fuzz { std::clamp(f, 0.0, 1.0) }
    {
    }

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const
    {
        vec reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
        attenuation *= albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

    color emitted(const hit_record& rec)
    {
        return color(0, 0, 0);
    }
    color albedo;
    double fuzz;
};

struct dielectric_material {
    dielectric_material() = default;
    dielectric_material(real_t ri, color albedo)
        : ref_idx { ri }
        , albedo { albedo }
    {
    }

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const
    {
        attenuation *= albedo;
        double refraction_ratio = rec.front_face ? (1.0 / ref_idx) : ref_idx;
        vec unit_direction = unit_vector(r_in.direction());
        double cos_theta = sycl::fmin(sycl::dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sycl::sqrt(1.0 - cos_theta * cos_theta);
        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec direction;
        if (cannot_refract)
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

    color emitted(const hit_record& rec)
    {
        return color(0, 0, 0);
    }
    // Refractive index of the glass
    real_t ref_idx;
    // Color of the glass
    color albedo;
};

struct lightsource_material {
    lightsource_material() = default;
    lightsource_material(texture_t& a)
        : emit { a }
    {
    }
    lightsource_material(const color& a)
        : emit { solid_texture { a } }
    {
    }

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const
    {
        return false;
    }

    color emitted(const hit_record& rec)
    {
        return std::visit([&](auto&& arg) { return arg.value(rec); }, emit);
    }

    texture_t emit;
};

using material_t = std::variant<lambertian_material, metal_material, dielectric_material, lightsource_material>;

#endif
