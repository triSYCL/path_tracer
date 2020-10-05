#ifndef HITTABLE_H
#define HITTABLE_H

#include "rtweekend.hpp"
#include "ray.hpp"
#include "vec3.hpp"

enum class material_t { Lambertian,
    Metal,
    Dielectric };

class hit_record {
public:
    hit_record() = default;

    bool scatter_material(const ray& r_in, vec3& attenuation, ray& scattered)
    {
        switch (material_type) {
        case material_t::Lambertian: {
            vec3 scatter_direction = normal + random_unit_vector();
            scattered = ray(p, scatter_direction);
            attenuation = albedo;
            return true;
        }
        case material_t::Metal: {
            vec3 reflected = reflect(unit_vector(r_in.direction()), normal);
            scattered = ray(p, reflected + fuzz * random_in_unit_sphere());
            attenuation = albedo;
            return (sycl::dot(scattered.direction(), normal) > 0);
        }
        case material_t::Dielectric: {
        }
        default:
            return false;
        }
    }
    vec3 center;
    real_t radius;

    double t;
    point3 p;
    vec3 normal;
    double u;
    double v;

    // material properties
    material_t material_type;
    vec3 albedo;
    real_t fuzz;
    real_t refraction_index;
};

template <class derived>
struct crtp {
    derived& underlying()
    {
        return static_cast<derived&>(*this);
    }
    const derived& underlying() const
    {
        return static_cast<const derived&>(*this);
    }
};

template <class geometry>
class hitable : crtp<geometry> {
public:
    bool hit(const ray& r, real_t min, real_t max, hit_record& rec) const
    {
        return this->underlying().hit(r, min, max, rec);
    }
};

class sphere : public hitable<sphere> {
public:
    sphere() = default;
    sphere(const vec3& cen, real_t r)
        : center(cen)
        , radius(r)
    {
        material_type = material_t::Lambertian;
    }
    sphere(const vec3& cen, real_t r, material_t mat_type, const vec3& color)
        : center(cen)
        , radius(r)
        , material_type(mat_type)
        , albedo(color)
    {
    }
    sphere(const vec3& cen, real_t r, material_t mat_type, const vec3& mat_color, real_t f)
        : center(cen)
        , radius(r)
        , material_type(mat_type)
        , albedo(mat_color)
        , fuzz(std::clamp(f, 0.0, 1.0))
    {
    }

    sphere(const vec3& cen, real_t r, material_t mat_type, real_t ref_idx)
        : center(cen)
        , radius(r)
        , material_type(mat_type)
        , refraction_index(ref_idx)
    {
    }

    bool hit(const ray& r, real_t min, real_t max, hit_record& rec) const
    {
        //$#rec.material_type = material_t::Lambertian;
        rec.material_type = material_type;
        rec.center = center;
        rec.radius = radius;
        if (material_type == material_t::Lambertian) {
            rec.albedo = albedo;
        }
        if (material_type == material_t::Metal) {
            rec.albedo = albedo;
            rec.fuzz = fuzz;
        }
        if (material_type == material_t::Dielectric) {
            rec.refraction_index = refraction_index;
        }
        vec3 oc = r.origin() - center;
        auto a = sycl::dot(r.direction(), r.direction());
        auto b = sycl::dot(oc, r.direction());
        auto c = sycl::dot(oc, oc) - radius * radius;
        auto discriminant = b * b - a * c;
        //std::cout<<"center "<<center<<std::endl;
        if (discriminant > 0) {
            auto temp = (-b - sycl::sqrt(discriminant)) / a;
            if (temp < max && temp > min) {
                rec.t = temp;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                return true;
            }
            temp = (-b + sycl::sqrt(discriminant)) / a;
            if (temp < max && temp > min) {
                rec.t = temp;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                return true;
            }
        }
        return false;
    }

    // geometry properties
    vec3 center;
    real_t radius;

    //material properties
    material_t material_type;
    vec3 albedo;
    real_t fuzz;
    real_t refraction_index;
};

#endif